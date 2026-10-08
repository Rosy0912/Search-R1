"""Collect own CPU position/action JSON metadata and actual scheduler state."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

job = sys.argv[1]
assert job.isdigit()
run = Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-safety300-3268')
commands = {}
for args in (['scontrol', 'show', 'job', job],
             ['sacct', '-j', job, '--format=JobID,State,ExitCode,Elapsed,AllocCPUS,AllocTRES', '-P']):
    p = subprocess.run(args, capture_output=True, text=True, timeout=15)
    commands[' '.join(args)] = dict(rc=p.returncode, stdout=p.stdout, stderr=p.stderr)
reports = {}
for prefix, suffix in [('position-analysis', 'json'), ('retained-actions', 'json'), ('layout-analysis', 'err')]:
    path = run / (prefix + '-' + job + '.' + suffix)
    if path.exists():
        payload = path.read_bytes()
        item = dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest(), text=payload.decode(errors='replace'))
        if suffix == 'json':
            try:
                item['result'] = json.loads(payload)
            except Exception as error:
                item['incomplete_or_error'] = type(error).__name__
        reports[path.name] = item
print(json.dumps(dict(epoch=time.time(), job_id=int(job), commands=commands, reports=reports,
      scope='Only own Slurm and read-only analysis JSON/text; no tensor/model/data assets exported'), allow_nan=False))
