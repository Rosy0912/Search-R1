"""Launch one sequential study step inside our retained eight-GPU allocation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument('--base', default='/dataset/shaoshitong/recursive-ablation150-20261014')
p.add_argument('--job', default='3393')
p.add_argument('--commit')
a = p.parse_args()
b = Path(a.base)
assert socket.gethostname().split('.')[0] == 'h100-11'
assert str(b).startswith('/dataset/shaoshitong/recursive-ablation150-')
deployment = json.loads((b / 'deployment.json').read_text())
if a.commit:
    assert deployment['commit'] == a.commit
receipt = b / 'study-launch.json'
if receipt.exists():
    print(receipt.read_text(), flush=True)
    raise SystemExit(0)
job = subprocess.check_output(['scontrol', 'show', 'job', a.job], text=True)
assert 'JobState=RUNNING' in job and 'NodeList=h100-11' in job and 'UserId=root(0)' in job
apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader,nounits'], text=True)
assert not apps.strip(), 'GPU processes present; no foreign process is stopped'
gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free', '--format=csv,noheader,nounits'], text=True)
rows = [line.split(',') for line in gpu.splitlines()]
assert len(rows) == 8 and len({x[1].strip() for x in rows}) == 8 and all(int(x[2]) >= 60000 for x in rows)
source = Path(deployment['source'])
script = source / 'release/recursive_bran_a/ablation150_infra/study_run.sh'
assert script.is_file()
(b / 'logs').mkdir(exist_ok=True)
wrapper = b / 'study-wrapper.sh'
wrapper.write_text('''#!/bin/bash
set +e
export RBA_ABLATION_BASE="%s"
export RBA_ABLATION_SOURCE_COMMIT="%s"
export RBA_H100_ALLOCATION_ID="$SLURM_JOB_ID"
bash "%s"
rc=$?
python3 - "$rc" <<'PY'
import json,os,time
from pathlib import Path
b=Path(os.environ['RBA_ABLATION_BASE'])
(b/'study-actual-exit.json').write_text(json.dumps(dict(time=time.time(),rc=int(__import__('sys').argv[1]),job=os.environ['SLURM_JOB_ID'],source_commit=os.environ['RBA_ABLATION_SOURCE_COMMIT'])))
PY
exit "$rc"
''' % (str(b), deployment['commit'], str(script)))
env = dict(os.environ, RBA_ABLATION_BASE=str(b), RBA_ABLATION_SOURCE_COMMIT=deployment['commit'])
args = ['srun', '--jobid=' + a.job, '--overlap', '--nodes=1', '--ntasks=1', '--gres=gpu:h100:8', '--cpus-per-task=96', '--mem=768G', 'bash', str(wrapper)]
with (b / 'logs/study-srun.log').open('ab', buffering=0) as log:
    child = subprocess.Popen(args, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
proc = Path('/proc', str(child.pid))
tick = int((proc / 'stat').read_text().rsplit(')', 1)[1].split()[19])
result = dict(time=time.time(),pid=child.pid,uid=proc.stat().st_uid,starttick=tick,job=a.job,command=args,
              source_commit=deployment['commit'],wrapper_sha256=hashlib.sha256(wrapper.read_bytes()).hexdigest(),
              gpu_admission=rows,status='SUBMITTED_NOT_TRAINING_ACCEPTED')
receipt.write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
