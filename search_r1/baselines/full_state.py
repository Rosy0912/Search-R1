"""Same-process checkpoint roundtrip primitives; never a weights-only resume."""
import hashlib
import os
import random
from pathlib import Path

import numpy as np
import torch


def rng_state(cuda_device=None):
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=[] if cuda_device is None else
                [dict(device=int(cuda_device),state=torch.cuda.get_rng_state(cuda_device))])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        if len(state['cuda'])!=1 or state['cuda'][0]['device']!=torch.cuda.current_device():
            raise ValueError('Rank-local CUDA RNG device changed')
        torch.cuda.set_rng_state(state['cuda'][0]['state'],state['cuda'][0]['device'])


def state_digest(value):
    """Exact shard/content witness with DTensor global layout and mesh identity.

    Metadata and local bytes only: no redistribute/full_tensor/all-gather, RNG
    calls, or private mesh mutation. CPU-offloaded DTensors retain their mesh.
    """
    h = hashlib.sha256()
    def add(x):
        if hasattr(x, 'local_shards'):
            h.update(str(x.metadata()).encode())
            for shard in x.local_shards():
                h.update(str(shard.metadata).encode()); add(shard.tensor)
        elif hasattr(x, 'to_local') and hasattr(x, 'placements'):
            mesh=x.device_mesh
            h.update(b'DTENSOR_LAYOUT_V2')
            add(dict(global_shape=tuple(x.shape),global_stride=tuple(x.stride()),
                     global_dtype=str(x.dtype),placements=tuple(str(p) for p in x.placements),
                     mesh_device_type=mesh.device_type,mesh_ranks=mesh.mesh,
                     mesh_dim_names=mesh.mesh_dim_names,mesh_coordinate=mesh.get_coordinate()))
            add(x.to_local())
        elif torch.is_tensor(x):
            t = x.detach().cpu().contiguous()
            h.update(str((str(t.dtype), tuple(t.shape))).encode())
            h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x, np.ndarray):
            h.update(str((str(x.dtype),x.shape)).encode());h.update(x.tobytes())
        elif isinstance(x, dict):
            for k in sorted(x, key=lambda k: (str(type(k)),repr(k))): add(k);add(x[k])
        elif isinstance(x, (tuple,list)):
            h.update(str(type(x)).encode())
            for item in x:add(item)
        else:
            h.update(str((type(x).__name__,repr(x))).encode())
    add(value)
    return h.hexdigest()


def atomic_save(value, path):
    path = Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    torch.save(value,temporary);os.replace(temporary,path)


class StatefulSampler(torch.utils.data.Sampler):
    """Explicit epoch permutation/cursor; num_workers=0 avoids unrecorded prefetch."""
    def __init__(self, data_source, seed=None, shuffle=True):
        self.size=len(data_source);self.shuffle=shuffle;self.epoch=0;self.cursor=0
        self.seed=seed;self.generator=torch.Generator().manual_seed(0 if seed is None else seed);self.order=None
    def __iter__(self):
        if self.order is None:
            if self.shuffle and self.seed is None:
                # Match torch RandomSampler(generator=None), preserving prior data order.
                self.generator.manual_seed(int(torch.empty((),dtype=torch.int64).random_().item()))
            self.order=torch.randperm(self.size,generator=self.generator).tolist() if self.shuffle else list(range(self.size))
        while self.cursor < self.size:
            i=self.order[self.cursor];self.cursor+=1;yield i
        self.epoch+=1;self.cursor=0;self.order=None
    def __len__(self):return self.size
    def state_dict(self):
        return dict(size=self.size,shuffle=self.shuffle,seed=self.seed,epoch=self.epoch,cursor=self.cursor,
                    order=self.order,generator=self.generator.get_state())
    def load_state_dict(self, state):
        if state['size']!=self.size or state['shuffle']!=self.shuffle or state['seed']!=self.seed:
            raise ValueError('Sampler dataset or shuffle changed')
        if not 0<=state['cursor']<=self.size:raise ValueError('Invalid sampler cursor')
        order=state['order']
        if order is not None and (len(order)!=self.size or set(order)!=set(range(self.size))):
            raise ValueError('Invalid sampler permutation')
        self.epoch=state['epoch'];self.cursor=state['cursor'];self.order=order
        self.generator.set_state(state['generator'])


def resolve_warmup_steps(optim_config):
    explicit=optim_config.get('num_warmup_steps')
    steps=int(optim_config.get('lr_warmup_steps_ratio',0.)*optim_config.get('total_training_steps',0)) if explicit is None else explicit
    if type(steps) is not int or steps<0:raise ValueError('num_warmup_steps must be a nonnegative integer')
    return steps
