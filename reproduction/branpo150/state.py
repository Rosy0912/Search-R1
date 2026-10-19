"""CPU tensor records used both before submission and after real native reload."""
from pathlib import Path
import gc
import hashlib
import json
import math
import random
import time
from .contract import sha, canonical


def tensor_record(value):
    import torch
    from torch.distributed.tensor import DTensor
    if isinstance(value, DTensor):
        local = value._local_tensor
        assert len(value.placements) == 1
        place = value.placements[0]
        assert place.is_shard() or place.is_replicate()
        layout = dict(kind='shard' if place.is_shard() else 'replicate',
                      mesh=value.device_mesh.mesh.tolist(), global_shape=list(value.shape),
                      mesh_names=list(value.device_mesh.mesh_dim_names))
        if place.is_shard(): layout['dim'] = place.dim
    else:
        assert torch.is_tensor(value)
        local = value
        layout = dict(kind='plain')
    local = local.detach().to('cpu').contiguous()
    assert torch.isfinite(local).all().item(), 'Nonfinite tensor'
    raw = local.reshape(-1).view(torch.uint8).numpy()
    return dict(shape=list(local.shape), dtype=str(local.dtype), numel=local.numel(),
                sha256=hashlib.sha256(memoryview(raw).cast('B')).hexdigest(), layout=layout)


def rng_record(rng):
    import numpy as np
    import torch
    assert set(rng) == {'cpu', 'cuda', 'numpy', 'random'}
    for key in ('cpu', 'cuda'):
        assert torch.is_tensor(rng[key]) and rng[key].dtype == torch.uint8
        assert rng[key].numel() > 0 and rng[key].ndim == 1
    torch.Generator(device='cpu').set_state(rng['cpu'].cpu())
    np.random.RandomState(0).set_state(rng['numpy'])
    random.Random(0).setstate(rng['random'])
    return dict(cpu=tensor_record(rng['cpu']), cuda=tensor_record(rng['cuda']),
                numpy=hashlib.sha256(rng['numpy'][1].tobytes()+repr((rng['numpy'][0],*rng['numpy'][2:])).encode()).hexdigest(),
                random=hashlib.sha256(repr(rng['random']).encode()).hexdigest())


def optimizer_record(optim, actual, count):
    import torch
    assert set(optim) == {'state', 'param_groups'}
    groups = optim['param_groups']
    params = [p for g in groups for p in g['params']]
    assert len(params) == len(set(params)) == count
    assert set(optim['state']) == set(params), 'Missing Adam state'
    assert all(math.isclose(float(g['lr']), 1e-6, rel_tol=1e-12) for g in groups)
    states = {}
    for key, value in optim['state'].items():
        assert set(value) == {'step', 'exp_avg', 'exp_avg_sq'}
        step = value['step'].item() if torch.is_tensor(value['step']) else value['step']
        assert math.isfinite(float(step)) and float(step)==int(step) and step>0
        if actual is not None:assert float(step) == actual, 'Adam counter mismatch'
        assert value['exp_avg'].shape == value['exp_avg_sq'].shape
        assert value['exp_avg'].ndim == 1, 'Frozen FSDP flat optimizer layout changed'
        assert value['exp_avg_sq'].ge(0).all().item()
        states[str(key)] = dict(step=int(step), exp_avg=tensor_record(value['exp_avg']),
                                exp_avg_sq=tensor_record(value['exp_avg_sq']))
    # The real frozen optimizer retains OmegaConf ListConfig for betas. Resolve
    # containers only in this observation record, never in the live optimizer.
    def plain(value):
        from omegaconf import OmegaConf
        if OmegaConf.is_config(value):value=OmegaConf.to_container(value,resolve=True)
        if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)):return [plain(v) for v in value]
        return value
    # All actual group settings, including epsilon/betas/weight decay, are bound.
    return json.loads(canonical(dict(param_groups=plain(groups), state=states)))


def validate_model_records(models, shapes):
    assert len(models) == 8 and all(set(m) == set(shapes) for m in models)
    for name, shape in shapes.items():
        rows = [m[name] for m in models]; layout = rows[0]['layout']
        assert all(r['layout'] == layout and r['dtype'] == rows[0]['dtype'] for r in rows)
        assert layout['mesh'] == list(range(8)) and layout['mesh_names'] == ['fsdp']
        assert layout['global_shape'] == list(shape)
        if layout['kind'] == 'replicate':
            assert all(r['shape'] == list(shape) and r['sha256'] == rows[0]['sha256'] for r in rows)
        else:
            assert layout['kind'] == 'shard' and layout['dim'] == 0
            assert all(r['shape'][1:] == list(shape)[1:] for r in rows)
            assert sum(r['shape'][0] for r in rows) == shape[0]


def audit_native(actor, shapes, physical_numel, actual, size):
    """Read all real24 files; never initialize CUDA, merge, or resume training."""
    import torch
    assert actual in (140, 150) and size in ('3b', '7b')
    assert not torch.cuda.is_initialized() and not torch.distributed.is_initialized()
    actor = Path(actor)
    assert actor.name == 'actor' and actor.parent.name == 'global_step_'+str(actual+100)
    files, models, workers = {}, [], []
    paths = [actor/f'{kind}_world_size_8_rank_{r}.pt' for r in range(8)
             for kind in ('model', 'optim', 'extra_state')]
    before = {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in paths}
    assert all(v[0] > 0 for v in before.values())
    started = time.monotonic()
    for rank in range(8):
        assert time.monotonic()-started < 2700, 'Bounded native audit deadline'
        paths_rank = {k: actor/f'{k}_world_size_8_rank_{rank}.pt' for k in ('model','optim','extra_state')}
        for k,p in paths_rank.items():
            assert p.is_file() and not p.is_symlink()
            files[p.name] = dict(path=str(p), bytes=before[p.name][0], mtime_ns=before[p.name][1], sha256=sha(p))
        model = torch.load(paths_rank['model'], map_location='cpu', weights_only=False)
        assert set(model) == set(shapes)
        models.append({k:tensor_record(v) for k,v in model.items()})
        del model;gc.collect()
        optim = torch.load(paths_rank['optim'], map_location='cpu', weights_only=False)
        optimizer = optimizer_record(optim, None, 37 if size=='3b' else 29)
        # Scheduler/progress counts iterations. Adam calls are measured from the
        # real native state and must not be inferred from that iteration count.
        measured_steps={v['step'] for v in optimizer['state'].values()}
        assert len(measured_steps)==1
        del optim;gc.collect()
        extra = torch.load(paths_rank['extra_state'], map_location='cpu', weights_only=False)
        assert set(extra) == {'lr_scheduler','rng'}
        scheduler = extra['lr_scheduler']
        assert scheduler['last_epoch'] == actual and scheduler['_step_count'] == actual+1
        assert scheduler['_last_lr'] == [1e-6] and scheduler['base_lrs'] == [1e-6]
        workers.append(dict(rank=rank, optimizer=optimizer, actual_adam_calls=next(iter(measured_steps)),
                            scheduler=scheduler, rng=rng_record(extra['rng'])))
        del extra;gc.collect()
        assert not torch.cuda.is_initialized()
    validate_model_records(models, shapes)
    assert len({w['actual_adam_calls'] for w in workers})==1
    flat_numel = sum(s['exp_avg']['numel'] for w in workers for s in w['optimizer']['state'].values())
    flat_count = 37 if size=='3b' else 29
    assert physical_numel <= flat_numel <= physical_numel+7*flat_count, 'Incomplete optimizer coverage'
    for p in paths:
        assert (p.stat().st_size,p.stat().st_mtime_ns) == before[p.name], 'Checkpoint changed during audit'
    return dict(status='FULL24_PHYSICAL_SHA_MODEL_ADAM_SAVED_RNG_SCHEDULER_CPU_ACCEPTED',
                size=size,actual_updates=actual,label=actual+100,world_size=8,files=files,
                model_shards=models,workers=workers,physical_numel=physical_numel,
                optimizer_numel_with_padding=flat_numel,cuda_initialized=False,
                no_cold_gpu_restore_claim=True,elapsed_seconds=time.monotonic()-started)
