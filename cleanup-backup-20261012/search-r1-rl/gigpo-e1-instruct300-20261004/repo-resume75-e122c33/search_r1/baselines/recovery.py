"""Explicit training-state recovery; HF weights alone are never exact resume."""
import hashlib
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Sampler


def cpu_snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_snapshot(item) for item in value)
    return value


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state() if torch.cuda.is_initialized() else None,
                cuda_device=torch.cuda.current_device() if torch.cuda.is_initialized() else None)


def validate_rng(state):
    random.Random().setstate(state['python'])
    np.random.RandomState(0).set_state(state['numpy'])
    torch.Generator().set_state(state['torch'])
    if state['cuda'] is not None:
        if (not torch.cuda.is_initialized()
                or state['cuda_device'] != torch.cuda.current_device()):
            raise ValueError('RNG CUDA device layout changed')
        current = torch.cuda.get_rng_state()
        if (not isinstance(state['cuda'], torch.Tensor)
                or state['cuda'].dtype != torch.uint8
                or state['cuda'].device.type != 'cpu'
                or state['cuda'].shape != current.shape):
            raise ValueError('Invalid CUDA RNG state')


def restore_rng(state):
    validate_rng(state)
    if state['cuda'] is not None and (not torch.cuda.is_initialized()
            or state['cuda_device'] != torch.cuda.current_device()):
        raise ValueError('RNG CUDA device layout changed')
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'])


def atomic_torch_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.partial')
    with temp.open('wb') as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def rebuild_scheduler_at_logical_step(scheduler, logical_step):
    """Keep the original LambdaLR horizon; reconstruct its logical position."""
    if logical_step < 0 or not isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR):
        raise ValueError('Expected nonnegative position and LambdaLR scheduler')
    rates = [base * fn(logical_step)
             for base, fn in zip(scheduler.base_lrs, scheduler.lr_lambdas)]
    scheduler.last_epoch = logical_step
    scheduler._step_count = logical_step + 1
    scheduler._last_lr = rates
    for group, rate in zip(scheduler.optimizer.param_groups, rates):
        group['lr'] = rate


class RecoverableSampler(Sampler):
    """Persist an actual permutation and cursor, with no worker prefetch."""
    def __init__(self, size, shuffle=True, seed=42):
        if size <= 0:
            raise ValueError('Dataset must be nonempty')
        self.size = size
        self.shuffle = shuffle
        self.generator = torch.Generator().manual_seed(seed)
        self.order = None
        self.cursor = 0
        self.epoch = -1

    def __len__(self):
        return self.size

    def __iter__(self):
        if self.order is None or self.cursor == self.size:
            self.order = (torch.randperm(self.size, generator=self.generator).tolist()
                          if self.shuffle else list(range(self.size)))
            self.cursor = 0
            self.epoch += 1
        while self.cursor < self.size:
            index = self.order[self.cursor]
            self.cursor += 1
            yield index

    def state_dict(self):
        return dict(size=self.size, shuffle=self.shuffle, epoch=self.epoch,
                    order=None if self.order is None else list(self.order),
                    cursor=self.cursor, generator=self.generator.get_state().clone())

    def load_state_dict(self, state):
        if state['size'] != self.size or state['shuffle'] != self.shuffle:
            raise ValueError('Dataset or shuffle configuration changed')
        order = state['order']
        if order is not None and sorted(order) != list(range(self.size)):
            raise ValueError('Invalid saved data permutation')
        if not 0 <= state['cursor'] <= self.size:
            raise ValueError('Invalid data cursor')
        if order is None and state['cursor'] != 0:
            raise ValueError('Cursor without permutation')
        self.order = None if order is None else list(order)
        self.cursor = state['cursor']
        self.epoch = state['epoch']
        self.generator.set_state(state['generator'])


def parameter_layout(module, optimizer, rank, world_size):
    names = {id(parameter): name for name, parameter in module.named_parameters()}
    parameters = []
    for name, parameter in module.named_parameters():
        entry = dict(name=name, shape=list(parameter.shape), dtype=str(parameter.dtype),
                     numel=parameter.numel(), requires_grad=parameter.requires_grad)
        for attr in ('_fqns', '_shapes', '_numels', '_numels_with_padding', '_is_padding_mask'):
            value = getattr(parameter, attr, None)
            if value is not None:
                entry[attr] = [list(item) if isinstance(item, torch.Size) else item for item in value]
        infos = getattr(parameter, '_shard_param_infos', None)
        if infos is not None:
            entry['shard_param_infos'] = [dict(item._asdict()) for item in infos]
        entry['shard_numel_padded'] = getattr(parameter, '_shard_numel_padded', None)
        parameters.append(entry)
    flat_layouts = []
    for name, submodule in module.named_modules():
        handle = getattr(submodule, '_handle', None)
        if handle is None:
            continue
        flat = handle.flat_param
        flat_layouts.append(dict(
            module=name, fqns=list(flat._fqns), shapes=[list(shape) for shape in flat._shapes],
            numels=list(flat._numels), numels_with_padding=list(flat._numels_with_padding),
            shard_param_infos=[dict(item._asdict()) for item in flat._shard_param_infos],
            shard_numel_padded=flat._shard_numel_padded, local_numel=flat.numel(),
            dtype=str(flat.dtype), use_orig_params=submodule._use_orig_params))
    groups = []
    for group in optimizer.param_groups:
        groups.append(dict(names=[names[id(parameter)] for parameter in group['params']],
                           hyperparameters={key: value for key, value in group.items()
                                            if key not in ('params', 'lr')}))
    return dict(rank=rank, world_size=world_size,
                use_orig_params=getattr(module, '_use_orig_params', None),
                parameters=parameters, flat_layouts=flat_layouts, optimizer_groups=groups)


def tensor_digest(tensors):
    digest = hashlib.sha256()
    for name, tensor in tensors:
        digest.update(str((name, tuple(tensor.shape), str(tensor.dtype))).encode())
        raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        view = memoryview(raw).cast('B')
        for start in range(0, len(view), 8 * 1024 * 1024):
            digest.update(view[start:start + 8 * 1024 * 1024])
    return digest.hexdigest()


def checkpoint_pause_identity(module, optimizer, scheduler, rank, sharding_manager=None, complete_internal=False):
    start_ticks = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19] if Path('/proc/self/stat').exists() else None
    moments = [(str(index) + '/' + key, value)
               for index, parameter in enumerate(module.parameters())
               for key, value in sorted(optimizer.state.get(parameter, {}).items())
               if isinstance(value, torch.Tensor)]
    generator_states = [('torch_cpu', torch.get_rng_state())]
    if torch.cuda.is_initialized():
        generator_states.append(('torch_cuda', torch.cuda.get_rng_state()))
    if sharding_manager is not None:
        generator_states.extend((key, value) for key in ('gen_random_states', 'torch_random_states')
                                if (value := getattr(sharding_manager, key, None)) is not None)
    identity = dict(rank=rank, pid=os.getpid(), start_ticks=start_ticks,
                scheduler_step=scheduler.last_epoch, learning_rates=scheduler.get_last_lr(),
                model_shards_sha256=tensor_digest(module.named_parameters()),
                optimizer_tensors_sha256=tensor_digest(moments),
                rng_tensors_sha256=tensor_digest(generator_states))
    if complete_internal:
        identity['local_shards_sha256'] = tensor_digest(
            (name, parameter._local_shard) for name, parameter in module.named_parameters()
            if hasattr(parameter, '_local_shard'))
        identity['buffers_sha256'] = tensor_digest(module.named_buffers())
    return identity


def worker_training_state(module, optimizer, scheduler, rank, world_size, logical_step,
                          sharding_manager=None, complete_internal=False):
    state = dict(schema=1, logical_step=logical_step,
                layout=parameter_layout(module, optimizer, rank, world_size),
                pause_identity=checkpoint_pause_identity(module, optimizer, scheduler, rank, sharding_manager, complete_internal),
                model_shards={name: cpu_snapshot(parameter)
                              for name, parameter in module.named_parameters()},
                optimizer=cpu_snapshot(optimizer.state_dict()),
                scheduler=cpu_snapshot(scheduler.state_dict()), rng=rng_state(),
                sharding_rng={key: cpu_snapshot(getattr(sharding_manager, key, None))
                              for key in ('gen_random_states', 'torch_random_states')},
                engine_restart_exact=False)
    if complete_internal:
        state['complete_internal'] = True
        state['local_shards'] = {name: cpu_snapshot(parameter._local_shard)
                                 for name, parameter in module.named_parameters()
                                 if hasattr(parameter, '_local_shard')}
        state['buffers'] = {name: cpu_snapshot(buffer) for name, buffer in module.named_buffers()}
    return state


def load_worker_training_state(state, module, optimizer, scheduler, rank, world_size,
                               sharding_manager=None):
    if state['schema'] != 1 or state['layout'] != parameter_layout(module, optimizer, rank, world_size):
        raise ValueError('Training-state parameter/shard/optimizer layout changed')
    saved = state['model_shards']
    live = dict(module.named_parameters())
    if saved.keys() != live.keys():
        raise ValueError('Model shard identity changed')
    # Validate every tensor before changing any live state.
    for name, parameter in live.items():
        tensor = saved[name]
        if tensor.shape != parameter.shape or tensor.dtype != parameter.dtype or not torch.isfinite(tensor).all():
            raise ValueError('Invalid model shard: ' + name)
    local_shards = buffers = {}
    if state.get('complete_internal', False):
        local_shards = {name: parameter._local_shard for name, parameter in live.items()
                        if hasattr(parameter, '_local_shard')}
        buffers = dict(module.named_buffers())
        for key, targets in [('local_shards', local_shards), ('buffers', buffers)]:
            if targets.keys() != state[key].keys():
                raise ValueError('Incomplete internal FSDP/buffer coverage')
            for name, target in targets.items():
                source = state[key][name]
                if source.shape != target.shape or source.dtype != target.dtype or not torch.isfinite(source).all():
                    raise ValueError('Invalid FSDP local shard or buffer: ' + name)
    saved_groups = state['optimizer']['param_groups']
    if len(saved_groups) != len(optimizer.param_groups):
        raise ValueError('Optimizer group count changed')
    slots = state['optimizer']['state']
    param_ids = []
    for saved_group, group in zip(saved_groups, optimizer.param_groups):
        if len(saved_group['params']) != len(group['params']):
            raise ValueError('Optimizer parameter group size changed')
        if ({key: value for key, value in saved_group.items() if key not in ('params', 'lr')}
                != {key: value for key, value in group.items() if key not in ('params', 'lr')}):
            raise ValueError('Optimizer parameter group hyperparameters changed')
        for identity, parameter in zip(saved_group['params'], group['params']):
            param_ids.append(identity)
            slot = slots.get(identity)
            if slot is None:  # valid for a parameter that has never received gradients
                continue
            required = {'step', 'exp_avg', 'exp_avg_sq'}
            if group.get('amsgrad', False):
                required.add('max_exp_avg_sq')
            if set(slot) != required:
                raise ValueError('Unexpected AdamW state fields')
            step = slot['step']
            if (not isinstance(step, torch.Tensor) or step.numel() != 1
                    or not torch.isfinite(step).all() or step.item() < 0
                    or step.item() != int(step.item())):
                raise ValueError('Invalid AdamW step')
            for key in required - {'step'}:
                tensor = slot[key]
                if (not isinstance(tensor, torch.Tensor) or tensor.shape != parameter.shape
                        or tensor.dtype != parameter.dtype or not torch.isfinite(tensor).all()):
                    raise ValueError('Invalid AdamW moment: ' + key)
    if len(set(param_ids)) != len(param_ids) or not set(slots).issubset(param_ids):
        raise ValueError('Invalid optimizer parameter identity mapping')
    schedule = state['scheduler']
    if (schedule.get('last_epoch') != state['logical_step']
            or len(schedule.get('_last_lr', [])) != len(optimizer.param_groups)):
        raise ValueError('Scheduler logical step or group count mismatch')
    import math
    expected_rates = [base * fn(state['logical_step'])
                      for base, fn in zip(scheduler.base_lrs, scheduler.lr_lambdas)]
    if (schedule.get('base_lrs') != scheduler.base_lrs
            or schedule.get('_step_count') != state['logical_step'] + 1
            or schedule.get('_last_lr') != expected_rates
            or any(not isinstance(rate, (float, int)) or not math.isfinite(rate) or rate < 0
                   for rate in schedule['_last_lr'])
            or [group.get('lr') for group in saved_groups] != expected_rates):
        raise ValueError('Invalid scheduler learning rate or position')
    validate_rng(state['rng'])
    for key, value in state['sharding_rng'].items():
        if value is None:
            if sharding_manager is not None and getattr(sharding_manager, key, None) is not None:
                raise ValueError('Missing saved sharding RNG')
            continue
        if sharding_manager is None:
            raise ValueError('Sharding RNG requires its original manager')
        current = getattr(sharding_manager, key, None)
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.uint8
                or value.device.type != 'cpu' or current is None or value.shape != current.shape):
            raise ValueError('Invalid sharding RNG')
    with torch.no_grad():
        for name, parameter in live.items():
            parameter.copy_(saved[name].to(parameter.device))
        for name, target in local_shards.items():
            target.copy_(state['local_shards'][name].to(target.device))
        for name, target in buffers.items():
            target.copy_(state['buffers'][name].to(target.device))
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    for group, rate in zip(optimizer.param_groups, scheduler.get_last_lr()):
        group['lr'] = rate
    if sharding_manager is not None:
        for key, value in state['sharding_rng'].items():
            setattr(sharding_manager, key, cpu_snapshot(value))
    restore_rng(state['rng'])
    return state['logical_step']


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json_save(value, path):
    import json
    path = Path(path)
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('w') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return dict(path=str(path), sha256=file_sha256(path))


def publish_checkpoint_manifest(checkpoint, worker_receipts, driver_receipt, source, step, actual,
                                recovery_mode='weights_only_original200', extra_receipts=()):
    checkpoint = Path(checkpoint)
    if not isinstance(worker_receipts, list) or len(worker_receipts) != 8:
        raise ValueError('Expected all eight worker training-state receipts')
    expected = {str(checkpoint / 'training_state' / f'rank_{rank}.pt') for rank in range(8)}
    if {receipt['path'] for receipt in worker_receipts} != expected:
        raise ValueError('Checkpoint rank receipts missing or duplicated')
    files = list(worker_receipts) + [driver_receipt] + list(extra_receipts)
    for path in sorted(checkpoint.iterdir()):
        if path.is_file():
            files.append(dict(path=str(path), bytes=path.stat().st_size, sha256=file_sha256(path)))
    for receipt in files:
        if (not Path(receipt['path']).is_file()
                or Path(receipt['path']).stat().st_size != receipt['bytes']
                or file_sha256(receipt['path']) != receipt['sha256']):
            raise ValueError('Checkpoint file missing or wrong size')
    state = dict(schema=1, job_id=os.environ.get('SLURM_JOB_ID'),
                 source=source, logical_step=step, actual_new_updates=actual,
                 recovery_mode=recovery_mode, engine_restart_exact=False,
                 code_commit=os.environ.get('SEARCH_R1_BASELINE_COMMIT'),
                 config_sha256=os.environ.get('SEARCH_R1_BASELINE_CONFIG_SHA256'),
                 data_identity=os.environ.get('SEARCH_R1_BASELINE_DATA_IDENTITY'),
                 files=files)
    receipt = atomic_json_save(state, checkpoint / 'training_state' / 'manifest.json')
    return dict(identity=state, **receipt)


def wait_for_independent_approval(manifest, approval_path, wait_seconds):
    import json
    import time
    if not approval_path or wait_seconds <= 0:
        raise ValueError('Recovery requires a finite independent-approval wait')
    ready_path = Path(approval_path).with_name('resume205-ready.json')
    atomic_json_save(manifest, ready_path)
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        if Path(approval_path).exists():
            approval = json.loads(Path(approval_path).read_text())
            expected = dict(job_id=manifest['identity']['job_id'],
                            source=manifest['identity']['source'], logical_step=205,
                            actual_new_updates=5, manifest_sha256=manifest['sha256'])
            if any(approval.get(key) != value for key, value in expected.items()):
                raise RuntimeError('Independent approval identity does not match this recovery')
            review = Path(approval['review_path'])
            if file_sha256(review) != approval['review_sha256']:
                raise RuntimeError('Independent review SHA mismatch')
            accepted = json.loads(review.read_text())
            if accepted.get('status') != 'PASS' or any(accepted.get(key) != value for key, value in expected.items()):
                raise RuntimeError('Independent review rejected or bound to another trajectory')
            return
        time.sleep(1)
    raise TimeoutError('Recovery paused at205: independent review did not authorize continuation')
