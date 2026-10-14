"""Bounded NCCL diagnostic on an otherwise idle eight-3090 allocation.

No model, optimizer, rollout or training checkpoint is loaded or modified.
This measures collectives only and is not an estimate of end-to-end RL speed.
"""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import socket
import statistics
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if (os.getuid() != 1066 or os.environ.get('SLURM_JOB_NODELIST') != 'node2'
            or socket.gethostname().split('.')[0] != 'gpu-3090-1'
            or not os.environ.get('SLURM_JOB_ID', '').isdigit()
            or int(os.environ.get('WORLD_SIZE', '0')) != 8):
        raise ValueError('Diagnostic requires the exact owned single-node eight-3090 allocation')
    import torch
    import torch.distributed as dist
    rank = int(os.environ['RANK'])
    local = int(os.environ['LOCAL_RANK'])
    if torch.cuda.device_count() != 8 or rank != local:
        raise ValueError('Need eight distinct local CUDA ranks')
    torch.set_num_threads(1)
    torch.cuda.set_device(local)
    dist.init_process_group('nccl', timeout=timedelta(seconds=90))
    device = torch.device('cuda', local)
    prop = torch.cuda.get_device_properties(local)
    samples = []
    started = time.time()
    try:
        for mib in (16, 64, 256):
            size = mib * 1024**2
            count = size // 4
            buffer = torch.empty(count, device=device, dtype=torch.float32)
            gathered = torch.empty(count * 8, device=device, dtype=torch.float32)
            for operation in ('all_reduce', 'all_gather_into_tensor'):
                elapsed = []
                for iteration in range(15):
                    buffer.fill_(rank + 1)
                    dist.barrier()
                    torch.cuda.synchronize()
                    begin = time.perf_counter()
                    if operation == 'all_reduce':
                        dist.all_reduce(buffer)
                    else:
                        dist.all_gather_into_tensor(gathered, buffer)
                    torch.cuda.synchronize()
                    seconds = time.perf_counter() - begin
                    if iteration >= 5:
                        elapsed.append(seconds)
                if operation == 'all_reduce':
                    correct = bool(torch.all(buffer == 36).item())
                else:
                    correct = all(bool(torch.all(gathered[i * count:(i + 1) * count] == i + 1).item())
                                  for i in range(8))
                if not correct:
                    raise ValueError('Actual collective output failed the full-tensor value check')
                samples.append(dict(operation=operation, input_bytes_per_rank=size,
                    output_bytes_per_rank=size if operation == 'all_reduce' else size * 8,
                    measured_seconds=elapsed, full_tensor_value_check=True))
                if rank == 0:
                    print(json.dumps(dict(event='COLLECTIVE_PROBE_CASE', operation=operation,
                        input_mib=mib, rank0_median_seconds=statistics.median(elapsed))), flush=True)
            del buffer, gathered
            torch.cuda.empty_cache()
        local_result = dict(rank=rank, local_rank=local, name=prop.name,
            total_memory=prop.total_memory, compute_capability=[prop.major, prop.minor],
            p2p_access=[bool(torch.cuda.can_device_access_peer(local, j)) if local != j else True for j in range(8)],
            cpu_affinity=sorted(os.sched_getaffinity(0)), cases=samples)
        results = [None] * 8
        dist.all_gather_object(results, local_result)
        if rank == 0:
            if {r['rank'] for r in results} != set(range(8)):
                raise ValueError('Incomplete physical collective rank coverage')
            summaries = []
            for index, case in enumerate(samples):
                synchronized = [max(r['cases'][index]['measured_seconds'][iteration] for r in results)
                                for iteration in range(10)]
                median = statistics.median(synchronized)
                summaries.append(dict(operation=case['operation'], input_bytes_per_rank=case['input_bytes_per_rank'],
                    median_rank_max_seconds=median, min_rank_max_seconds=min(synchronized),
                    max_rank_max_seconds=max(synchronized),
                    input_payload_GiB_per_second=case['input_bytes_per_rank'] / 1024**3 / median))
            value = dict(at=time.time(), started=started, job=os.environ['SLURM_JOB_ID'],
                host=socket.gethostname(), source_commit=os.environ['RECURSIVE_SOURCE_COMMIT'],
                status='EIGHT_RANK_COLLECTIVE_DIAGNOSTIC_COMPLETE', torch_version=torch.__version__,
                cuda_version=torch.version.cuda, ranks=results, summaries=summaries,
                gpu_identity=subprocess.check_output(['nvidia-smi',
                    '--query-gpu=index,name,uuid,memory.total,driver_version', '--format=csv,noheader,nounits'], text=True),
                topology=subprocess.check_output(['nvidia-smi', 'topo', '-m'], text=True),
                scope='NCCL collectives only: five warmups and ten measured calls, maximum rank duration per call, full output value checks. No model/optimizer/rollout, no RL ablation, no end-to-end speed or migration compatibility claim.')
            with Path(args.output).open('x') as stream:
                json.dump(value, stream, indent=2, allow_nan=False)
            print(json.dumps(dict(event=value['status'], output=args.output, summaries=summaries)), flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
