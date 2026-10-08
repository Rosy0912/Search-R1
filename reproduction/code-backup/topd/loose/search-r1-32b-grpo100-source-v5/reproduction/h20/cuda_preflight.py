import os
import torch
import torch.distributed as dist

rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(rank)
dist.init_process_group('nccl')
x = torch.tensor([float(rank)], device='cuda')
dist.all_reduce(x)
assert x.item() == 28.0, x.item()
a = torch.randn(128, 128, device='cuda', dtype=torch.bfloat16)
assert torch.isfinite(a @ a).all()
if rank == 0:
    print('EIGHT_GPU_NCCL_BF16_OK', torch.__version__, flush=True)
dist.destroy_process_group()
