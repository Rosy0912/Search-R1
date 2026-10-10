"""Actual CPU-only Ray workers, with the same native-runtime environment as training."""
import json
import os
import pathlib
import sys
import time

BASE = pathlib.Path('/data/user/jxu957/experiments/topd-20261008')
sys.path.insert(0, str(BASE / 'runtime/tools/topd-instruct300-launch-v6/runtime'))
from allocator_policy import worker_environment
import ray

temp = BASE / 'environment/ray-preliminary'
temp.mkdir(exist_ok=True)
alias = pathlib.Path('/tmp') / ('topd-ray-probe-' + os.environ['SLURM_JOB_ID'])
alias.symlink_to(temp, target_is_directory=True)
try:
    ray.init(address='local', num_cpus=2, num_gpus=0, include_dashboard=False, _temp_dir=str(alias),
             object_store_memory=100 * 1024**2, runtime_env={'env_vars': worker_environment(os.environ)})
    @ray.remote
    def actual_worker(v):
        import torch, flash_attn_interface, flash_attn_3_cuda
        return {'value': v + 1, 'pid': os.getpid(), 'preload': os.environ.get('LD_PRELOAD'),
                'cuda_initialized': torch.cuda.is_initialized()}
    rows = []
    for v in [41, 42]:
        row = ray.get(actual_worker.remote(v), timeout=90)
        assert row['value'] == v + 1 and row['preload'] and not row['cuda_initialized']
        rows.append(row)
        print(json.dumps({'actual_task': len(rows), 'pid': row['pid'], 'epoch': time.time()}), flush=True)
    p = BASE / 'outputs/training-preparation/ray-preliminary-validation.json'
    p.write_text(json.dumps({'status': 'TWO_ACTUAL_RAY_CPU_TASKS_NATIVE_FA3_ENV_PASS',
                            'job': os.environ['SLURM_JOB_ID'], 'tasks': rows, 'gpu_used': False}, indent=2))
finally:
    ray.shutdown()
    alias.unlink()
