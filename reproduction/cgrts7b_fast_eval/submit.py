"""One held-record-release milestone50 submission; fail closed on uncertainty."""
import json
import shlex
import subprocess
import time
from pathlib import Path
from .entry import JOB, verify
from .prepare import ROOT
from reproduction.official_eval_v2.driver import once
from reproduction.official_eval_v2.protocol import sha


def submit():
    path = ROOT / 'plans/cgrts50-r1.json'
    plan = json.loads(path.read_text()); root = Path(plan['root'])
    assert root == ROOT / 'cgrts50-r1' and plan['job'] == JOB
    assert json.loads((root / 'preflight.json').read_text())['plan_sha256'] == sha(path)
    verify(plan)
    assert not (root / 'submission-intent.json').exists()
    assert not (root / 'submission.json').exists() and not (root / 'output').exists()
    assert not subprocess.check_output(['squeue', '-h', '--name=' + JOB, '-o', '%i|%T'], text=True).strip()
    assert not subprocess.check_output(['sacct', '-X', '-n', '-P', '-S', '2026-10-19T00:00:00',
        '--name=' + JOB, '--format=JobIDRaw,JobName%70,State'], text=True).strip()
    env = '/home/jmzhang/Workspace/Search-G1-Question-Diversity-20261013-v12/env'
    q = shlex.quote
    command = [env + '/bin/python3', '-m', 'reproduction.cgrts7b_fast_eval.entry', str(path)]
    script = '\n'.join(['#!/bin/bash', 'set -euo pipefail',
        'export PATH=' + q(env + '/bin') + ':"$PATH"',
        'export PYTHONPATH=' + q(plan['source']) + ' PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1',
        'export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false',
        'export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 VLLM_ATTENTION_BACKEND=XFORMERS',
        'export VLLM_USE_V1=0 VLLM_WORKER_MULTIPROC_METHOD=spawn WANDB_MODE=disabled',
        'unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy PYTHONHOME RAY_ADDRESS',
        'cd ' + q(plan['source']), 'ulimit -n 65535', 'set +e', shlex.join(command),
        'rc=$?', 'echo "$rc" > ' + q(str(root / 'exit.code')), 'exit "$rc"', ''])
    cmd = ['sbatch', '--parsable', '--hold', '--no-requeue', '--account=lumia', '--qos=normal',
        '--ntasks=1', '--job-name=' + JOB, '--output=' + str(root / 'slurm-%j.log'),
        '--nodes=1', '--gres=gpu:8', '--cpus-per-task=64', '--mem=600G', '--time=1-00:00:00',
        '--partition=ADA6000,L40S,RTX4090,RTX3090']
    once(root / 'submission-intent.json', dict(at=time.time(), job=JOB,
        plan_sha256=sha(path), command=cmd, script=script, milestone50_not_final200=True))
    result = subprocess.run(cmd, input=script, text=True, capture_output=True)
    once(root / 'submission-response.json', dict(at=time.time(), returncode=result.returncode,
        stdout=result.stdout, stderr=result.stderr))
    result.check_returncode(); jid = result.stdout.strip().split(';')[0]; assert jid.isdigit()
    receipt = dict(at=time.time(), job_id=jid, job=JOB, kind=plan['kind'], repeat=1,
        root=str(root), plan=str(path), plan_sha256=sha(path), source_commit=plan['source_commit'],
        no_auto_retry=True, not_final200=True)
    once(root / 'submission.json', receipt)
    subprocess.run(['scontrol', 'release', jid], check=True)
    print('SUBMITTED=' + json.dumps(receipt), flush=True)


if __name__ == '__main__':
    submit()
