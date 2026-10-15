"""Bounded eight-rank hardware probe, without model weights or optimizer updates."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import time
from datetime import timedelta

import torch
import torch.distributed as dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    assert int(os.environ['WORLD_SIZE']) == 8
    assert torch.cuda.device_count() == 8
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl', timeout=timedelta(seconds=120))
    x = torch.tensor([rank], dtype=torch.float32, device='cuda')
    dist.all_reduce(x)
    assert x.item() == 28
    a = torch.ones((256, 256), dtype=torch.bfloat16, device='cuda', requires_grad=True)
    y = a @ a.T
    assert torch.all(y == 256).item()
    y.float().mean().backward()
    assert torch.isfinite(a.grad).all().item()
    torch.cuda.synchronize()
    prop = torch.cuda.get_device_properties(local_rank)
    row = dict(rank=rank, local_rank=local_rank, host=socket.gethostname(),
               device=prop.name, memory_bytes=prop.total_memory,
               capability=list(torch.cuda.get_device_capability(local_rank)),
               nccl_sum=x.item(), bf16_forward_backward_finite=True,
               torch=torch.__version__, cuda=torch.version.cuda,
               optimizer_updates=0, model_weights_loaded=False)
    rows = [None] * 8
    dist.all_gather_object(rows, row)
    if rank == 0:
        assert {r['rank'] for r in rows} == set(range(8))
        args.output.mkdir(parents=True, exist_ok=True)
        result = dict(at=time.time(), passed=True, ranks=rows,
                      packages={n: importlib.metadata.version(n) for n in
                                ('transformers', 'vllm', 'ray', 'flash-attn')},
                      source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      scope='Hardware and PyTorch/NCCL only; no model, rollout, optimizer, checkpoint restoration, or vLLM engine acceptance.')
        with (args.output / 'hardware.PASS.json').open('x') as f:
            json.dump(result, f, indent=2)
        print(json.dumps(result), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
