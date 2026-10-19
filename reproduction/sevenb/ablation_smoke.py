"""Eight-device compiler smoke in a disposable process before policy training."""
import json
import os
from pathlib import Path
import sys
import time
import triton
import triton.language as tl
from .ablations import COMPILER, sha


@triton.jit
def add_seven(x, y, N: tl.constexpr, B: tl.constexpr):
    offsets = tl.program_id(0) * B + tl.arange(0, B)
    tl.store(y + offsets, tl.load(x + offsets, offsets < N, other=0) + 7, offsets < N)


def run(path):
    import torch
    from triton.backends.nvidia.compiler import _path_to_binary
    assert os.environ['TRITON_PTXAS_PATH'] == COMPILER['path'] and sha(COMPILER['path']) == COMPILER['sha256']
    selected, version = _path_to_binary('ptxas'); assert selected == COMPILER['path'] and version == '12.4'
    assert torch.cuda.device_count() == 8
    rng = torch.get_rng_state().clone(); devices = []
    for index in range(8):
        with torch.cuda.device(index):
            x = torch.arange(257, dtype=torch.float32, device='cuda'); y = torch.empty_like(x)
            add_seven[(triton.cdiv(257, 128),)](x, y, 257, B=128)
            torch.cuda.synchronize(); assert torch.equal(y.cpu(), torch.arange(257, dtype=torch.float32) + 7)
            devices.append(dict(device=index, name=torch.cuda.get_device_name(index), capability=torch.cuda.get_device_capability(index)))
            del x, y
    assert torch.equal(rng, torch.get_rng_state())
    with Path(path).open('x') as f:
        json.dump(dict(at=time.time(), status='PASSED', compiler=COMPILER, devices=devices, cpu_rng_unchanged=True, no_policy_or_optimizer_loaded=True), f, indent=2)


if __name__ == '__main__':
    run(sys.argv[1])
