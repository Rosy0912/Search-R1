"""Remote-only durable state and parameter isolation for the actor-probe variant."""

import hashlib
import json
import os
import random
import uuid
from pathlib import Path


def tree_digest(value):
    import torch
    h = hashlib.sha256()
    def visit(item):
        import numpy as np
        if isinstance(item, torch.Tensor):
            t = item.detach().cpu().contiguous()
            h.update(json.dumps([str(t.dtype), list(t.shape)]).encode())
            h.update(memoryview(t.reshape(-1).view(torch.uint8).numpy()))
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            h.update(json.dumps([str(array.dtype), list(array.shape)]).encode())
            h.update(memoryview(array))
        elif isinstance(item, dict):
            for key in sorted(item, key=repr):
                h.update(repr(key).encode()); visit(item[key])
        elif isinstance(item, (tuple, list)):
            h.update(type(item).__name__.encode())
            for child in item: visit(child)
        else:
            h.update(json.dumps(item, sort_keys=True, allow_nan=False).encode())
    visit(value)
    return h.hexdigest()


def isolation_state(module, optimizer=None, scheduler=None):
    import torch
    # The existing offload helper uses nonblocking GPU-to-CPU copies. Do not
    # hash their CPU destinations before this worker's stream has finished.
    if torch.cuda.is_initialized(): torch.cuda.synchronize()
    result = dict(parameters_sha256=tree_digest(dict(module.named_parameters())),
        gradients_sha256=tree_digest({name: p.grad for name, p in module.named_parameters()}),
        module_modes_sha256=tree_digest({name: m.training for name, m in module.named_modules()}))
    if optimizer is not None:
        result['optimizer_sha256'] = tree_digest(optimizer.state_dict())
    if scheduler is not None:
        result['scheduler_sha256'] = tree_digest(scheduler.state_dict())
    return result


def rng_state():
    import numpy as np
    import torch
    return dict(python=random.getstate(), numpy=np.random.get_state(),
        torch_cpu=torch.get_rng_state(),
        torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])


def validate_rng(state):
    import numpy as np
    import torch
    assert set(state) == {'python', 'numpy', 'torch_cpu', 'torch_cuda'}
    random.Random(0).setstate(state['python'])
    np.random.RandomState(0).set_state(state['numpy'])
    torch.Generator(device='cpu').set_state(state['torch_cpu'])
    assert isinstance(state['torch_cuda'], list)
    assert bool(state['torch_cuda']) == torch.cuda.is_initialized(), 'Saved/current CUDA RNG topology differs'
    if state['torch_cuda']:
        assert torch.cuda.is_initialized(), 'Restore cannot create a different CUDA RNG topology'
        actual = torch.cuda.get_rng_state_all()
        assert len(actual) == len(state['torch_cuda'])
        assert all(saved.dtype == torch.uint8 and saved.shape == current.shape
                   for saved, current in zip(state['torch_cuda'], actual))


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'])
    if state['torch_cuda']: torch.cuda.set_rng_state_all(state['torch_cuda'])


def cpu_state(value):
    import torch
    if isinstance(value, torch.Tensor): return value.detach().cpu()
    if isinstance(value, dict): return {k: cpu_state(v) for k, v in value.items()}
    if isinstance(value, list): return [cpu_state(v) for v in value]
    if isinstance(value, tuple): return tuple(cpu_state(v) for v in value)
    return value


def durable_torch_save(payload, target, base):
    import torch
    from reproduction.common8.topology import validate_base
    base = Path(base).resolve(); validate_base(base, payload['identity'])
    target = Path(target)
    assert target.parent.resolve().is_relative_to(base), 'Full state must remain in this remote namespace'
    assert not target.exists(), 'Never overwrite a full-state checkpoint'
    tmp = target.with_name(target.name + '.tmp-' + uuid.uuid4().hex)
    with tmp.open('xb') as f:
        torch.save(cpu_state(payload), f); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, target)
    fd = os.open(target.parent, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try: os.fsync(fd)
    finally: os.close(fd)
    h = hashlib.sha256()
    with target.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''): h.update(chunk)
    return dict(path=str(target), bytes=target.stat().st_size, sha256=h.hexdigest())


def worker_payload(worker, run, completed_outer):
    import torch
    if torch.cuda.is_initialized(): torch.cuda.synchronize()
    evidence = worker._outer_scheduler_evidence
    assert evidence.completed_outer == completed_outer and evidence.pending is None and evidence.prepared is None
    assert not evidence.rejected and worker.actor_lr_scheduler.last_epoch == completed_outer
    actor = worker.actor
    assert actor._complete_outers == completed_outer
    assert actor._last_attempt_index == evidence.last_attempt_index
    assert actor._used_attempt_ids == evidence.used_attempt_ids
    return dict(schema_version=1, role='actor_probe_full_worker_state',
        identity={key: run[key] for key in ('run_id', 'execution_id', 'source_commit',
            'execution_backend', 'training_world_size', 'profile', 'slurm_job_id')},
        rank=worker.rank, completed_outer=completed_outer, torch_version=torch.__version__,
        optimizer_writes=worker.actor._optimizer_writes,
        actor_accounting=dict(completed_outer=actor._complete_outers,
            last_attempt_index=actor._last_attempt_index, used_attempt_ids=sorted(actor._used_attempt_ids),
            policy_rejections=getattr(actor, '_policy_rejections', 0)),
        flat_parameter_layout=[dict(name=name, shape=list(p.shape), dtype=str(p.dtype))
                               for name, p in worker.actor_module_fsdp.named_parameters()],
        optimizer=worker.actor_optimizer.state_dict(), scheduler=worker.actor_lr_scheduler.state_dict(),
        outer_accounting=dict(actual_writes=evidence.actual_writes,
            last_attempt_index=evidence.last_attempt_index, used_attempt_ids=sorted(evidence.used_attempt_ids),
            completed_outer=completed_outer, pending=None, prepared=None, rejected=False),
        rollout_rng=dict(training=worker.rollout_sharding_manager.torch_random_states,
                         generation=worker.rollout_sharding_manager.gen_random_states),
        rng=rng_state())


def validate_worker_payload(worker, run, payload):
    """Validate actual parameter/Adam layout before any model or state mutation."""
    import torch
    from reproduction.common8.scheduler import expected_lr
    identity = {key: run[key] for key in ('run_id', 'execution_id', 'source_commit',
        'execution_backend', 'training_world_size', 'profile', 'slurm_job_id')}
    assert payload['identity'] == identity and payload['rank'] == worker.rank
    assert payload['role'] == 'actor_probe_full_worker_state'
    assert payload['torch_version'] == torch.__version__, 'Optimizer shard runtime changed'
    layout = [dict(name=name, shape=list(p.shape), dtype=str(p.dtype))
              for name, p in worker.actor_module_fsdp.named_parameters()]
    assert layout == payload['flat_parameter_layout'], 'FSDP flat optimizer layout differs'
    outer = payload['outer_accounting']; completed = payload['completed_outer']
    assert type(completed) is int and 1 <= completed <= 200
    assert outer['completed_outer'] == completed and outer['pending'] is None
    assert outer['prepared'] is None and outer['rejected'] is False
    assert outer['actual_writes'] == payload['optimizer_writes'] > 0
    assert outer['last_attempt_index'] >= completed
    assert len(set(outer['used_attempt_ids'])) == outer['last_attempt_index']
    actor = payload['actor_accounting']
    assert actor['completed_outer'] == completed and actor['last_attempt_index'] == outer['last_attempt_index']
    assert actor['used_attempt_ids'] == outer['used_attempt_ids']
    assert type(actor['policy_rejections']) is int and actor['policy_rejections'] >= 0
    assert payload['scheduler']['last_epoch'] == completed
    assert payload['scheduler']['_step_count'] == completed + 1
    assert all(group['lr'] == expected_lr(completed) for group in payload['optimizer']['param_groups'])
    assert payload['optimizer']['state']
    groups = payload['optimizer']['param_groups']
    current_groups = worker.actor_optimizer.state_dict()['param_groups']
    assert len(groups) == len(current_groups) == len(worker.actor_optimizer.param_groups)
    parameters = {}
    for saved, current, actual in zip(groups, current_groups, worker.actor_optimizer.param_groups):
        assert {k:v for k,v in saved.items() if k != 'lr'} == {k:v for k,v in current.items() if k != 'lr'}, \
            'Adam parameter groups or hyperparameters changed'
        assert len(saved['params']) == len(actual['params'])
        parameters.update(zip(saved['params'], actual['params']))
    assert set(payload['optimizer']['state']) == set(parameters), 'Incomplete saved Adam parameter state'
    for index, values in payload['optimizer']['state'].items():
        for key in ('exp_avg', 'exp_avg_sq'):
            assert values[key].shape == parameters[index].shape and values[key].dtype == parameters[index].dtype, \
                'Saved Adam moment shape or precision differs from the actual parameter'
    for values in payload['optimizer']['state'].values():
        assert values['step'] == payload['optimizer_writes']
        for value in values.values():
            if isinstance(value, torch.Tensor): assert torch.isfinite(value).all()
    validate_rng(payload['rng'])
    for key in ('training', 'generation'):
        assert payload['rollout_rng'][key].dtype == torch.uint8
        actual = (worker.rollout_sharding_manager.torch_random_states if key == 'training'
                  else worker.rollout_sharding_manager.gen_random_states)
        assert payload['rollout_rng'][key].shape == actual.shape, 'Rollout RNG device geometry changed'
    evidence = worker._outer_scheduler_evidence
    assert evidence.pending is None and evidence.prepared is None and not evidence.rejected
    return outer, actor


def restore_worker_payload(worker, run, payload):
    """Restore validated state; canonical weights and parent binding are separate."""
    outer, actor = validate_worker_payload(worker, run, payload)
    evidence = worker._outer_scheduler_evidence
    worker.actor_optimizer.load_state_dict(payload['optimizer'])
    worker.actor_lr_scheduler.load_state_dict(payload['scheduler'])
    worker.actor._optimizer_writes = payload['optimizer_writes']
    worker.actor._complete_outers = actor['completed_outer']
    worker.actor._last_attempt_index = actor['last_attempt_index']
    worker.actor._used_attempt_ids = set(actor['used_attempt_ids'])
    worker.actor._policy_rejections = actor['policy_rejections']
    for key in ('actual_writes', 'last_attempt_index', 'completed_outer'):
        setattr(evidence, key, outer[key])
    evidence.used_attempt_ids = set(outer['used_attempt_ids'])
    worker.rollout_sharding_manager.torch_random_states = payload['rollout_rng']['training'].clone()
    worker.rollout_sharding_manager.gen_random_states = payload['rollout_rng']['generation'].clone()
    restore_rng(payload['rng'])
