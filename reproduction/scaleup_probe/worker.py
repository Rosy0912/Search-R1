"""16-GPU NCCL and tiny fresh FSDP check; no native recovery/speedup claim."""
import datetime
import json
import os
from pathlib import Path
import socket
import sys
import time

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from .contract import read_plan, sha


def timings(group, count, loops=5):
    tensor = torch.ones(count, dtype=torch.float32, device='cuda')
    size = dist.get_world_size(group)
    values = []
    for i in range(3 + loops):
        tensor.fill_(1)
        torch.cuda.synchronize(); start = time.perf_counter()
        dist.all_reduce(tensor, group=group)
        torch.cuda.synchronize(); elapsed = time.perf_counter() - start
        assert bool(torch.all(tensor == size)), 'NCCL sum mismatch'
        if i >= 3: values.append(elapsed)
    del tensor
    return dict(payload_bytes=count * 4, participants=size, seconds=values,
                infrastructure_collective_only_not_training_throughput=True)


def main():
    pp = Path(sys.argv[1]); plan = read_plan(pp); root = Path(plan['root'])
    rank = int(os.environ['RANK']); local = int(os.environ['LOCAL_RANK'])
    assert int(os.environ['WORLD_SIZE']) == 16 and 0 <= local < 8
    torch.set_num_threads(1); torch.cuda.set_device(local)
    assert torch.cuda.device_count() == 8
    dist.init_process_group('nccl', timeout=datetime.timedelta(seconds=180))
    node = rank // 8
    hardware = json.loads((root / ('node-' + str(node)) / 'hardware.json').read_text())
    me = dict(rank=rank, local_rank=local, node=node, hostname=socket.gethostname(),
              uuid=hardware['devices'][local][0].strip(),
              name=torch.cuda.get_device_name(local), torch=torch.__version__,
              cuda=torch.version.cuda, plan_sha256=sha(pp))
    all_devices = [None] * 16; dist.all_gather_object(all_devices, me)
    assert len({v['uuid'] for v in all_devices}) == 16
    assert len({v['hostname'] for v in all_devices}) == 2
    assert len({v['name'] for v in all_devices}) == 1
    assert all('A100' in v['name'] for v in all_devices)
    # All ranks create groups in identical order, including non-members.
    g0 = dist.new_group(list(range(8)), backend='nccl')
    g1 = dist.new_group(list(range(8, 16)), backend='nccl')
    local_group = g0 if node == 0 else g1
    local_times = [timings(local_group, n) for n in (262144, 16777216)]
    dist.barrier()
    full_times = [timings(None, n) for n in (262144, 16777216)]
    torch.manual_seed(137)
    model = FSDP(torch.nn.Sequential(torch.nn.Linear(127, 257), torch.nn.Tanh(),
                                   torch.nn.Linear(257, 19)).cuda(),
                 device_id=local, use_orig_params=False)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-6, foreach=False)
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        loss = model(torch.ones(4, 127, device='cuda')).square().mean()
        loss.backward()
        assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        opt.step()
    steps = [int(s['step'].item()) for s in opt.state.values()]
    assert steps and set(steps) == {3} and bool(torch.isfinite(loss))
    assert all(bool(torch.isfinite(p).all()) for p in model.parameters())
    report = dict(**me, at=time.time(), local8_collectives=local_times,
                  cross_node16_collectives=full_times, tiny_fresh_fsdp_adam_steps=steps,
                  tiny_loss=float(loss.detach()), actual_native_cold_restore_test=False,
                  actual7b_or_training_throughput_test=False, production_migration=False)
    with (root / ('rank-' + str(rank) + '.json')).open('x') as f:
        json.dump(report, f, indent=2)
    dist.barrier()
    if rank == 0:
        with (root / 'all16-gpu-identities.json').open('x') as f: json.dump(all_devices, f, indent=2)
    dist.barrier(); dist.destroy_process_group()


if __name__ == '__main__':
    main()
