"""Verifiable same-world boundary snapshots; no HF-only or automatic restart."""
import copy
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def digest(value):
    h = hashlib.sha256()
    def visit(v):
        if isinstance(v, torch.Tensor):
            h.update(str((str(v.dtype), tuple(v.shape))).encode())
            raw = v.detach().contiguous().reshape(-1).view(torch.uint8)
            for block in raw.split(8 * 1024**2):
                h.update(block.cpu().numpy().tobytes())
        elif isinstance(v, np.ndarray):
            h.update(str((str(v.dtype), v.shape)).encode()); h.update(v.tobytes())
        elif isinstance(v, dict):
            for k in sorted(v, key=repr):
                visit(k); visit(v[k])
        elif isinstance(v, (tuple, list)):
            h.update(type(v).__name__.encode())
            for x in v: visit(x)
        else:
            h.update(repr((type(v).__name__, v)).encode())
    visit(value)
    return h.hexdigest()


def finite(value):
    if isinstance(value, torch.Tensor) and value.is_floating_point():
        for block in value.detach().reshape(-1).split(4 * 1024**2):
            if not torch.isfinite(block).all():
                raise ValueError('Nonfinite complete training state')
    elif isinstance(value, dict):
        for v in value.values(): finite(v)
    elif isinstance(value, (list, tuple)):
        for v in value: finite(v)
    elif isinstance(value, float) and not np.isfinite(value):
        raise ValueError('Nonfinite complete training scalar')


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''): h.update(block)
    return h.hexdigest()


def atomic_save(value, path, json_format=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + f'.pending-{os.getpid()}-{time.time_ns()}')
    try:
        with pending.open('xb') as f:
            if json_format:
                f.write(json.dumps(value, sort_keys=True, allow_nan=False).encode())
            else:
                torch.save(value, f)
            f.flush(); os.fsync(f.fileno())
        os.link(pending, path)  # Existing snapshot/receipt is never overwritten.
    finally:
        pending.unlink(missing_ok=True)
    return file_sha(path)


def rng_state(cuda=False):
    value = dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state())
    if cuda:
        value['cuda'] = dict(device=torch.cuda.current_device(), state=torch.cuda.get_rng_state(torch.cuda.current_device()))
    return cpu_copy(value)


def restore_rng(value):
    random.setstate(value['python']); np.random.set_state(value['numpy'])
    torch.set_rng_state(value['torch'])
    if 'cuda' in value:
        if value['cuda']['device'] != torch.cuda.current_device(): raise ValueError('CUDA RNG device changed')
        torch.cuda.set_rng_state(value['cuda']['state'], torch.cuda.current_device())


class CheckpointBatchSampler:
    """No prefetch: cursor advances only when the driver requests the next batch."""
    def __init__(self, rows, batch_size, seed, identity, shuffle=True):
        if rows < batch_size or batch_size < 1: raise ValueError('Invalid sampler dimensions')
        self.rows, self.batch_size, self.identity, self.shuffle = int(rows), int(batch_size), identity, bool(shuffle)
        self.generator = torch.Generator().manual_seed(seed)
        self.order = None; self.cursor = 0; self.epoch = -1

    def __len__(self): return self.rows // self.batch_size

    def __iter__(self):
        limit = len(self) * self.batch_size
        if self.order is None or self.cursor >= limit:
            self.order = (torch.randperm(self.rows, generator=self.generator).tolist()
                          if self.shuffle else list(range(self.rows)))
            self.cursor = 0; self.epoch += 1
        while self.cursor < limit:
            start = self.cursor; self.cursor += self.batch_size
            yield self.order[start:self.cursor]

    def state_dict(self):
        return cpu_copy(dict(rows=self.rows, batch_size=self.batch_size, identity=self.identity,
                             shuffle=self.shuffle, order=self.order, cursor=self.cursor,
                             epoch=self.epoch, generator=self.generator.get_state()))

    def load_state_dict(self, state):
        for key in ('rows', 'batch_size', 'identity', 'shuffle'):
            if state[key] != getattr(self, key): raise ValueError('Sampler identity/layout changed')
        order = state['order']; cursor = state['cursor']
        if (order is None and (cursor != 0 or state['epoch'] != -1)) or (order is not None and state['epoch'] < 0):
            raise ValueError('Incoherent sampler epoch/order/cursor')
        if order is not None and (len(order) != self.rows or set(order) != set(range(self.rows))):
            raise ValueError('Incomplete sampler permutation')
        if not 0 <= cursor <= len(self) * self.batch_size or cursor % self.batch_size:
            raise ValueError('Invalid sampler cursor')
        self.order = copy.deepcopy(order); self.cursor = cursor; self.epoch = state['epoch']
        self.generator.set_state(state['generator'])


def worker_snapshot(module, optimizer, scheduler, identity, cuda=False):
    if cuda: torch.cuda.synchronize(torch.cuda.current_device())
    value = dict(version=2, identity=copy.deepcopy(identity),
                 local_shards={n: cpu_copy(p._local_shard) for n, p in module.named_parameters() if hasattr(p, "_local_shard")},
                 parameters={n: cpu_copy(p) for n, p in module.named_parameters()},
                 gradients={n: None if p.grad is None else cpu_copy(p.grad) for n, p in module.named_parameters()},
                 buffers={n: cpu_copy(p) for n, p in module.named_buffers()},
                 optimizer=cpu_copy(optimizer.state_dict()), scheduler=cpu_copy(scheduler.state_dict()),
                 training=module.training, rng=rng_state(cuda))
    finite(value)
    return value


def restore_worker(module, optimizer, scheduler, saved, identity, cold=False):
    if saved['version'] != 2 or saved['identity'] != identity: raise ValueError('Worker snapshot identity changed')
    finite(saved)
    parameters = dict(module.named_parameters()); buffers = dict(module.named_buffers())
    local_shards = {n: p._local_shard for n, p in parameters.items() if hasattr(p, '_local_shard')}
    if local_shards.keys() != saved['local_shards'].keys(): raise ValueError('FSDP local shard coverage changed')
    for name, target in local_shards.items():
        source = saved['local_shards'][name]
        if source.shape != target.shape or source.dtype != target.dtype:
            raise ValueError('FSDP local shard shape/dtype changed')
    current_groups = optimizer.param_groups
    saved_groups = saved['optimizer']['param_groups']
    if len(current_groups) != len(saved_groups): raise ValueError('Optimizer group layout changed')
    current_states = optimizer.state_dict()['state']
    if cold and current_states:
        raise ValueError('Cold restore requires an untouched empty optimizer')
    if cold and set(saved['optimizer']['state']) != {
            key for group in saved_groups for key in group['params']}:
        raise ValueError('Cold Adam state coverage incomplete')
    if not cold and saved['optimizer']['state'].keys() != current_states.keys():
        raise ValueError('Initialized Adam state coverage changed during same-world restore')
    parameter_by_id = {}
    for current, group in zip(current_groups, saved_groups):
        if len(current['params']) != len(group['params']): raise ValueError('Optimizer parameter layout changed')
        for parameter, key in zip(current['params'], group['params']):
            if key in parameter_by_id: raise ValueError('Duplicate optimizer parameter identity')
            parameter_by_id[key] = parameter
    for key, state in saved['optimizer']['state'].items():
        if key not in parameter_by_id: raise ValueError('Unknown optimizer parameter identity')
        parameter = parameter_by_id[key]
        for name in ('exp_avg', 'exp_avg_sq'):
            moment = state.get(name)
            if not isinstance(moment, torch.Tensor) or moment.shape != parameter.shape or moment.dtype != parameter.dtype:
                raise ValueError('Adam moment shard shape/dtype changed')
        step = state.get('step')
        if not isinstance(step, torch.Tensor) or step.numel() != 1 or step.item() < 0 or step.item() != int(step.item()):
            raise ValueError('Invalid Adam step')
    for key, current in [('parameters', parameters), ('buffers', buffers)]:
        if current.keys() != saved[key].keys(): raise ValueError('Worker parameter/buffer names changed')
        for name, target in current.items():
            source = saved[key][name]
            if source.shape != target.shape or source.dtype != target.dtype:
                raise ValueError('Worker shard shape/dtype changed')
    if parameters.keys() != saved['gradients'].keys(): raise ValueError('Gradient shard names changed')
    for name, parameter in parameters.items():
        gradient = saved['gradients'][name]
        if gradient is not None and (gradient.shape != parameter.shape or gradient.dtype != parameter.dtype):
            raise ValueError('Gradient shard shape/dtype changed')
    with torch.no_grad():
        for key, current in [('parameters', parameters), ('buffers', buffers)]:
            for name, target in current.items(): target.copy_(saved[key][name].to(target.device))
        for name, target in local_shards.items():
            target.copy_(saved['local_shards'][name].to(target.device))
        for name, parameter in parameters.items():
            gradient = saved['gradients'][name]
            parameter.grad = None if gradient is None else gradient.to(parameter.device).clone()
    optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
    module.train(saved['training']); restore_rng(saved['rng'])


def migrate_sampler_batch(sampler, saved, batch_size):
    """Keep consumed questions exactly; only change subsequent batch boundaries.

    A mixed-batch run cannot infer its cursor from outer count alone. Callers
    must retain parent provenance and the saved cursor in recovery receipts.
    """
    if sampler.batch_size != batch_size or batch_size < 1:
        raise ValueError('Recovery sampler batch mismatch')
    for key in ('rows', 'identity', 'shuffle'):
        if saved[key] != getattr(sampler, key):
            raise ValueError('Recovery sampler identity changed')
    order, cursor = saved['order'], saved['cursor']
    if order is None or len(order) != sampler.rows or set(order) != set(range(sampler.rows)):
        raise ValueError('Incomplete recovery permutation')
    if not 0 <= cursor <= sampler.rows or cursor % batch_size:
        raise ValueError('Recovery cursor does not align with new batch')
    migrated = cpu_copy(saved)
    migrated['batch_size'] = batch_size
    sampler.load_state_dict(migrated)
    return dict(parent_batch_size=saved['batch_size'], batch_size=batch_size,
                consumed_cursor=cursor, epoch=saved['epoch'],
                permutation_sha256=digest(order), exact_executor_resume=False)


def snapshot_worker_to_file(module, optimizer, scheduler, path, identity, verify=False, cuda=False):
    started = time.monotonic()
    saved = worker_snapshot(module, optimizer, scheduler, identity, cuda)
    logical = digest(saved); sha = atomic_save(saved, path)
    del saved
    if verify:
        loaded = torch.load(path, map_location='cpu', weights_only=False)
        if digest(loaded) != logical: raise ValueError('Serialized worker state differs')
        restore_worker(module, optimizer, scheduler, loaded, identity)
        del loaded
        actual = worker_snapshot(module, optimizer, scheduler, identity, cuda)
        if digest(actual) != logical: raise ValueError('Reloaded worker state differs')
        del actual
    return dict(path=str(path), sha256=sha, logical_state_sha256=logical,
                bytes=Path(path).stat().st_size, identity=identity,
                full_worker_state_finite=True, exact_same_world_reload_verified=verify,
                checkpoint_and_reload_seconds=time.monotonic()-started,
                scheduler_last_epoch=scheduler.last_epoch,
                optimizer_state_entries=len(optimizer.state), optimizer_lrs=[g['lr'] for g in optimizer.param_groups])


def driver_snapshot(trainer, identity):
    controller = trainer.baseline_controller
    return cpu_copy(dict(version=1, identity=identity, completed_step=trainer.global_steps,
                         next_step=trainer.global_steps+1, actual_outer_updates=trainer.actual_outer_updates,
                         sampler=trainer.checkpoint_sampler.state_dict(), rng=rng_state(),
                         algorithm=dict(method=controller.method, stage=controller.stage.state_dict(),
                                        tree_rng=None if controller.tree_rng is None else controller.tree_rng.getstate())))


def restore_driver(trainer, saved, identity):
    if saved['version'] != 1 or saved['identity'] != identity:
        raise ValueError('Driver snapshot identity changed')
    if (saved['completed_step'] != saved['actual_outer_updates']
            or saved['next_step'] != saved['completed_step']+1
            or saved['completed_step'] < 0):
        raise ValueError('Incoherent driver completed/next/actual counters')
    controller = trainer.baseline_controller
    if saved['algorithm']['method'] != controller.method:
        raise ValueError('Algorithm identity changed')
    sampler = saved['sampler']; steps = saved['completed_step']
    batches = sampler['rows'] // sampler['batch_size']
    if steps and (sampler['epoch'] != (steps-1)//batches
                  or sampler['cursor'] != ((steps-1)%batches+1)*sampler['batch_size']):
        raise ValueError('Sampler cursor disagrees with actual completed updates')
    trainer.checkpoint_sampler.load_state_dict(saved['sampler'])
    controller.stage.load_state_dict(saved['algorithm']['stage'])
    if controller.tree_rng is not None:
        if saved['algorithm']['tree_rng'] is None: raise ValueError('Missing Tree RNG')
        controller.tree_rng.setstate(saved['algorithm']['tree_rng'])
    trainer.global_steps = saved['completed_step']; trainer.actual_outer_updates = saved['actual_outer_updates']
    restore_rng(saved['rng'])


def wait_for_acceptance(directory, ready, timeout_seconds=43200):
    """Only a matching independent receipt can release the still-live driver."""
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    ready_sha = atomic_save(ready, directory / 'ready.json', json_format=True)
    start = time.monotonic()
    while True:
        receipt = directory / 'approved.json'
        if receipt.exists():
            receipt_bytes = receipt.read_bytes()
            receipt_sha = hashlib.sha256(receipt_bytes).hexdigest()
            approval = json.loads(receipt_bytes)
            if approval.get('status') != 'PASS' or approval.get('ready_sha256') != ready_sha:
                raise ValueError('Independent acceptance receipt does not match ready state')
            for key in ('job_id', 'code_commit', 'config_sha256', 'data_identity', 'model_identity'):
                if approval.get(key) != ready.get(key): raise ValueError('Acceptance identity changed: ' + key)
            if not all(approval.get(key) is True for key in (
                    'full_state_reload_accepted', 'full_hf_reload_accepted', 'rank_tree_numerics_accepted')):
                raise ValueError('Independent complete-state/HF/Tree acceptance missing')
            if 'full_state_manifest' in ready:
                manifest_path = ready['full_state_manifest']
                manifest_bytes = Path(manifest_path).read_bytes()
                if (approval.get('full_state_manifest_sha256') != ready['full_state_manifest_sha256']
                        or hashlib.sha256(manifest_bytes).hexdigest() != ready['full_state_manifest_sha256']):
                    raise ValueError('Complete-state manifest identity changed')
                manifest = json.loads(manifest_bytes)
                for record in manifest['worker_states'] + [manifest['driver']]:
                    if file_sha(record['path']) != record['sha256']:
                        raise ValueError('Complete-state shard changed after acceptance')
                from .select_teacher import model_manifest
                if (approval.get('hf_identity_sha256') != ready['hf_identity']['sha256']
                        or model_manifest(ready['hf_checkpoint']) != ready['hf_identity']):
                    raise ValueError('HF checkpoint identity changed after acceptance')
            if receipt.read_bytes() != receipt_bytes:
                raise ValueError('Acceptance receipt changed during verification')
            atomic_save(dict(status='RELEASED_FOR_SAME_JOB_6_TO_300', ready_sha256=ready_sha,
                             approval_sha256=receipt_sha, pause_seconds=time.monotonic()-start),
                        directory / 'released.json', json_format=True)
            return approval
        if time.monotonic() - start > timeout_seconds:
            raise TimeoutError('Independent step5 gate timed out; no sixth batch was fetched')
        time.sleep(2)
