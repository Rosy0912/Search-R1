"""Bounded eight-GPU communication diagnostic; no model or training is loaded."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import socket
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if (os.getuid() != 1066 or os.environ.get('SLURM_JOB_NODELIST') != 'node6'
            or os.environ.get('SLURM_JOB_NAME') != 'rt16-seven-nccl-probe-n6-v1'
            or not os.environ.get('SLURM_JOB_ID', '').isdigit()
            or os.environ.get('WORLD_SIZE') != '8'
            or os.environ.get('NCCL_P2P_DISABLE') != '1'
            or os.environ.get('NCCL_IB_DISABLE') != '1'):
        raise ValueError('Requires the exact owned diagnostic allocation and bound transport')
    import torch
    import torch.distributed as dist
    rank = int(os.environ['RANK'])
    local = int(os.environ['LOCAL_RANK'])
    if rank != local or torch.cuda.device_count() != 8:
        raise ValueError('Expected eight distinct local CUDA ranks')
    torch.set_num_threads(1)
    torch.cuda.set_device(local)
    dist.init_process_group('nccl', timeout=timedelta(seconds=90))
    cases = []
    try:
        for mib in (1, 64, 256):
            data = torch.full((mib * 1024**2 // 4,), rank + 1.,
                              dtype=torch.float32, device=local)
            started = time.monotonic()
            # Reproduce the FSDP initialization primitive before testing reduction.
            dist._broadcast_coalesced(dist.group.WORLD, [data], 256 * 1024**2, 0)
            torch.cuda.synchronize()
            if not torch.all(data == 1).item():
                raise ValueError('Broadcast output differs')
            data.fill_(rank + 1)
            dist.all_reduce(data)
            torch.cuda.synchronize()
            if not torch.all(data == 36).item():
                raise ValueError('All-reduce output differs')
            cases.append(dict(mib=mib, seconds=time.monotonic()-started,
                              broadcast_and_reduce_values_verified=True))
            del data
        rows = [None] * 8
        dist.all_gather_object(rows, dict(rank=rank, local_rank=local,
                                         name=torch.cuda.get_device_name(local), cases=cases))
        if rank == 0:
            hardware = subprocess.check_output(['nvidia-smi',
                '--query-gpu=index,name,uuid', '--format=csv,noheader'], text=True)
            gpu_rows = [r.split(',') for r in hardware.splitlines() if r.strip()]
            if len(gpu_rows) != 8 or len({r[2].strip() for r in gpu_rows}) != 8:
                raise ValueError('Physical GPU identity is not eight distinct devices')
            if {r['rank'] for r in rows} != set(range(8)):
                raise ValueError('Incomplete rank coverage')
            result = dict(status='EIGHT_RANK_COMMUNICATION_PROBE_PASSED', at=time.time(),
                          job=os.environ['SLURM_JOB_ID'], host=socket.gethostname(),
                          source=os.environ['RECURSIVE_SOURCE_COMMIT'], ranks=rows,
                          hardware=hardware, transport=dict(NCCL_P2P_DISABLE='1', NCCL_IB_DISABLE='1'),
                          scope='Collective diagnostic only; full evaluator GPU execution remains unverified')
            with Path(args.output).open('x') as stream:
                json.dump(result, stream, indent=2, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            print(json.dumps(result), flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
