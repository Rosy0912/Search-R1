"""Tiny compiler/device verification in a disposable process, no policy/model/RNG."""
import json
import os
from pathlib import Path
import sys
import time
from .pj_entry import COMPILER
from .contract import sha


def run(output):
    assert os.environ['TRITON_PTXAS_PATH']==COMPILER['path']
    assert sha(COMPILER['path'])==COMPILER['sha256']
    import torch
    import triton
    import triton.language as tl
    from triton.backends.nvidia.compiler import _path_to_binary,ptx_get_version
    selected,version=_path_to_binary('ptxas')
    assert selected==COMPILER['path'] and version=='12.4' and ptx_get_version(version)==84
    assert torch.cuda.device_count()==8
    cpu_rng=torch.get_rng_state().clone()
    @triton.jit
    def add_seven(x,y,N:tl.constexpr,B:tl.constexpr):
        offsets=tl.program_id(0)*B+tl.arange(0,B)
        values=tl.load(x+offsets,offsets<N,other=0)
        tl.store(y+offsets,values+7,offsets<N)
    checked=[]
    for device in range(8):
        with torch.cuda.device(device):
            x=torch.arange(257,dtype=torch.float32,device='cuda');y=torch.empty_like(x)
            add_seven[(triton.cdiv(x.numel(),128),)](x,y,x.numel(),B=128)
            torch.cuda.synchronize();assert torch.equal(y.cpu(),torch.arange(257,dtype=torch.float32)+7)
            checked.append(dict(device=device,name=torch.cuda.get_device_name(device),capability=torch.cuda.get_device_capability(device)))
            del x,y
    assert torch.equal(cpu_rng,torch.get_rng_state())
    v=dict(at=time.time(),status='PASSED',compiler=COMPILER,selected_version=version,devices=checked,no_policy_model_loaded=True,no_training_updates=True,cpu_rng_unchanged=True)
    with Path(output).open('x') as f:json.dump(v,f,indent=2)


if __name__=='__main__':run(sys.argv[1])
