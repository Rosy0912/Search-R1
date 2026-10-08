"""Collect only audit metadata/logs from one own same-node zero-write run."""
import base64
import gzip
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

job = sys.argv[1]
assert job.isdigit()
root = Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
run = root / 'runs' / ('tree-safety300-' + job)
files = {}
for location in (run, run / 'output', run / 'gpu-kl-proof', run / 'native-fullstate'):
    paths = location.iterdir() if location == run else location.rglob('*')
    for path in paths:
        if not path.is_file() or path.suffix not in ('.json', '.jsonl', '.log', '.txt', '.ready', '.yaml'):
            continue
        data = path.read_bytes()
        files[str(path.relative_to(run))] = dict(sha256=hashlib.sha256(data).hexdigest(),
             bytes=len(data), gzip_base64=base64.b64encode(gzip.compress(data)).decode())
for name in ('tree-safety110-calibration-' + job + '.out', 'tree-safety110-calibration-' + job + '.err'):
    path = root / 'logs' / name
    if path.exists():
        data = path.read_bytes()
        files['slurm-logs/' + name] = dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data),
             gzip_base64=base64.b64encode(gzip.compress(data)).decode())
commands = {}
for args in (['scontrol', 'show', 'job', job],
             ['sacct', '-j', job, '--format=JobID,State,ExitCode,Elapsed,AllocCPUS,AllocTRES', '-P'],
             ['nvidia-smi', '--query-gpu=uuid,memory.used,utilization.gpu', '--format=csv,noheader'],
             ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader']):
    result = subprocess.run(args, capture_output=True, text=True, timeout=15)
    commands[' '.join(args)] = dict(rc=result.returncode, stdout=result.stdout, stderr=result.stderr)
background = {}
for pid in (1680782, 1680783, 1680784, 1680785, 1696443, 1696444, 1696445, 1696446):
    path = Path('/proc') / str(pid) / 'stat'
    background[str(pid)] = path.read_text() if path.exists() else None
print(json.dumps(dict(job_id=int(job), epoch=time.time(), files=files, commands=commands,
      background_proc_stat=background,
      scope='Metadata and logs only; all model/index/data/trajectory tensors remain on h100-9'), allow_nan=False))
