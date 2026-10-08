"""Read-only CPU analysis reports, preserving partial-output and Slurm evidence."""
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
    result = subprocess.run(args, capture_output=True, text=True, timeout=15)
    commands[' '.join(args)] = dict(rc=result.returncode, stdout=result.stdout, stderr=result.stderr)
reports = {}
for prefix, suffix in (('raw-stats', 'json'), ('tree-trajectory', 'json'),
                       ('context-analysis', 'json'), ('context-analysis', 'err')):
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
print(json.dumps(dict(epoch=time.time(), job_id=int(job), reports=reports, commands=commands,
      scope='Audit report JSON/text only; no tensor or model assets transferred'), allow_nan=False))
