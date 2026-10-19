"""Explicit CPU/CUDA RNG transition around the frozen strict flat-state core.

Only the tiny tested format is supported. Production identity/guard accounting
and actual7B layouts are intentionally not implemented by this probe.
"""
import copy
import hashlib
import random

import numpy as np
import torch

POLICY = 'tiny-native-cuda8to16-parent-digest-domain-separated-v1'


def seed(parent_digest, rank, stream):
    text = '|'.join((POLICY, parent_digest, 'world16', str(rank), stream))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'little')


def rng_transition(parents, rank, local_device, digest, private_cuda_state):
    if len(parents) != 8 or not 0 <= rank < 16 or local_device != rank % 8:
        raise ValueError('Only homogeneous2x8 tiny CUDA transition supported')
    shape = None
    for r, value in enumerate(parents):
        if set(value) != {'python','numpy','torch','cuda'} or set(value['cuda']) != {'device','state'}:
            raise ValueError('Complete original CPU/CUDA RNG fields required')
        state = value['cuda']['state']
        if (value['cuda']['device'] != r or not isinstance(state, torch.Tensor)
                or state.device.type != 'cpu' or state.dtype != torch.uint8
                or state.ndim != 1 or state.numel() == 0):
            raise ValueError('Malformed source local-device CUDA state')
        if shape is None: shape = state.shape
        if state.shape != shape: raise ValueError('Inconsistent CUDA state byte lengths')
    if rank < 8:
        return copy.deepcopy(parents[rank])
    parent_digest = digest(parents)
    cpu_seed, cuda_seed = seed(parent_digest, rank, 'cpu'), seed(parent_digest, rank, 'cuda')
    state = private_cuda_state(cuda_seed)
    if state.device.type != 'cpu' or state.dtype != torch.uint8 or state.shape != shape:
        raise ValueError('Target CUDA generator state format differs')
    return dict(python=random.Random(cpu_seed).getstate(),
                numpy=np.random.RandomState(cpu_seed % 2**32).get_state(),
                torch=torch.Generator().manual_seed(cpu_seed).get_state(),
                cuda=dict(device=local_device, state=state.clone()))


def transform(source, source_layouts, target_layout, target_identity, core, digest,
              local_device, private_cuda_state):
    for item in source:
        for state in item['optimizer']['state'].values():
            if torch.any(state['exp_avg_sq'] < 0):
                raise ValueError('Negative Adam second moment')
    parents = [s['rng'] for s in source]
    # Keep the frozen model/Adam/scheduler/shard verifier byte-identical. Its
    # CPU-only RNG mapping is replaced by the explicitly registered CUDA policy.
    cpu_source = copy.deepcopy(source)
    for item in cpu_source: item['rng'].pop('cuda', None)
    result = core.transform(cpu_source, source_layouts, target_layout, target_identity, digest)
    result['rng'] = rng_transition(parents, target_identity['rank'], local_device,
                                   digest, private_cuda_state)
    core.finite(result)
    return result
