import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

root = Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-monitor-r6-20261004')
run = root / 'runs/bapo-instruct300-formal300-3177'
staging = run / 'logs/step5-independent-release-3177-reviewed-staging.json'
release = run / 'logs/step5-independent-release.json'
expected = 'bd087d435906ac44566cdc2e56200aa11861bd73ecb6bc7d73a7098f929c78f1'
assert not release.exists()
assert hashlib.sha256(staging.read_bytes()).hexdigest() == expected
slurm = subprocess.check_output(['scontrol', 'show', 'job', '-dd', '3177'])
assert b'JobState=RUNNING' in slurm and b'Restarts=0' in slurm
driver_stat = Path('/proc/3908944/stat').read_text()
env = os.environ.copy()
env.update(PYTHONPATH=str(root / 'repo'), CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='8',
    MKL_NUM_THREADS='8', PYTHONDONTWRITEBYTECODE='1')
code = 'from search_r1.baselines.checkpoint_gate import validate_release; import sys; validate_release(sys.argv[1],sys.argv[2],sys.argv[3],expected_driver_pid=3908944); print("ACTUAL_STEP5_STAGING_GATE_PASS",flush=True)'
started = time.monotonic()
check = subprocess.run([str(root / 'env/bin/python'), '-c', code, str(staging),
    str(run / 'checkpoints/full_state/global_step_5/manifest.json'), str(root / 'repo')],
    env=env, cwd=root / 'repo')
assert check.returncode == 0, 'Do not publish failed release'
assert hashlib.sha256(staging.read_bytes()).hexdigest() == expected
assert driver_stat.split(') ', 1)[1].split()[19] == Path('/proc/3908944/stat').read_text().split(') ', 1)[1].split()[19]
assert not release.exists()
os.link(staging, release)
print(json.dumps({'status': 'ACTUAL_GATE_PASS_ATOMIC_RELEASE_PUBLISHED', 'job_id': 3177,
    'driver_pid': 3908944, 'receipt_sha256': expected, 'gate_wall_seconds': time.monotonic()-started,
    'release': str(release)}), flush=True)
