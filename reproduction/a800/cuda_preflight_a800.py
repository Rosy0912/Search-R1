import os
import torch
import torch.distributed as dist

rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(rank)
dist.init_process_group('nccl')
assert dist.get_world_size() == 4
x = torch.tensor([float(rank)], device='cuda')
dist.all_reduce(x)
assert x.item() == 6.0, x.item()
a = torch.randn(128, 128, device='cuda', dtype=torch.bfloat16)
assert torch.isfinite(a @ a).all()
if rank == 0:
    print('FOUR_GPU_NCCL_BF16_OK', torch.__version__, flush=True)
dist.destroy_process_group()
