"""Real four-rank NCCL/BF16/JIT and collective no-write failure admission."""
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
from search_g1.numerics import require_finite

assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0,1,2,3', 'Only physical GPUs0–3 are authorized for this preflight'
rank=int(os.environ['LOCAL_RANK']);assert rank in range(4);torch.cuda.set_device(rank)
dist.init_process_group('nccl')
try:
    assert dist.get_world_size()==4
    x=torch.tensor([float(rank)],device='cuda');dist.all_reduce(x);assert x.item()==6
    a=torch.randn(128,128,device='cuda',dtype=torch.bfloat16)
    assert torch.isfinite(a@a).all()
    def computation(z):return z.sin()+z.cos()
    z=torch.arange(1024,device='cuda',dtype=torch.float32)/1000
    compiled=torch.compile(computation,fullgraph=True)(z);torch.cuda.synchronize()
    assert torch.allclose(compiled,computation(z),rtol=1e-5,atol=1e-5)
    parameter=torch.nn.Parameter(torch.tensor([1.],device='cuda'))
    optimizer=torch.optim.Adam([parameter],lr=.01)
    original=parameter.detach().clone()
    parameter.grad=torch.tensor([float('nan') if rank==2 else 1.],device='cuda')
    rejected=False
    try:
        require_finite({'gradient':parameter.grad},stage='gpu-admission-reject-one-rank')
        optimizer.step()
    except RuntimeError:rejected=True
    assert rejected and torch.equal(parameter,original) and not optimizer.state
    parameter.grad=torch.ones_like(parameter)
    require_finite({'gradient':parameter.grad},stage='gpu-admission-valid-write')
    optimizer.step();assert torch.isfinite(parameter).all() and not torch.equal(parameter,original)
    # Empty rank must still participate in the same rejection collective.
    rejected=False
    try:require_finite({} if rank==1 else {'value':torch.ones(1,device='cuda')},stage='gpu-admission-empty-rank')
    except RuntimeError:rejected=True
    assert rejected
    dist.barrier()
    base=Path(os.environ['SEARCH_G1_BASE'])
    run=json.loads((base/'logs/run-manifest.json').read_text())
    result=dict(run_id=run['run_id'],rank=rank,world_size=4,physical_devices=os.environ['CUDA_VISIBLE_DEVICES'],
                nccl_bf16=True,real_cuda_jit=True,nonfinite_rank_collectively_rejected_before_write=True,
                empty_rank_collectively_rejected=True,finite_adam_write=True,torch_version=torch.__version__)
    (base/'logs'/f'gpu-preflight-rank{rank}.json').write_text(json.dumps(result,indent=2))
    if rank==0:print('SEARCH_G1_FOUR_GPU_ADMISSION_PASSED='+run['run_id'],flush=True)
finally:dist.destroy_process_group()
