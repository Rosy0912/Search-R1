"""One held Slurm submission per immutable 7B plan; no automatic retries."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from .contract import ROOT, OLD
from .entry import verify
from reproduction.seven_dataset.contracts import sha


def write_once(path, value):
    with Path(path).open('x') as f:
        json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())


def submit(planfile):
    plan=json.loads(Path(planfile).read_text());root=verify(plan)
    proof=json.loads((root/'preflight.json').read_text())
    assert proof['plan_sha256']==sha(planfile) and proof['status']=='FULL_SHA_DATA_CONFIG_VERIFIED'
    assert not (root/'submission-intent.json').exists() and not (root/'submission.json').exists()
    job=plan['job']
    assert not subprocess.check_output(['squeue','-h','--name='+job,'-o','%i|%T'],text=True).strip()
    assert not subprocess.check_output(['sacct','-X','-n','-P','-S','2026-10-18T00:00:00','--name='+job,'--format=JobIDRaw,State'],text=True).strip()
    assert os.statvfs(ROOT).f_bavail*os.statvfs(ROOT).f_frsize>1_000_000_000_000
    tree=plan['kind']=='treegrpo'
    env=OLD/'official-treegrpo/env' if tree else Path('/home/jmzhang/Workspace/Search-G1-Question-Diversity-20261013-v12/env')
    py=str(env/'bin/python');q=shlex.quote;source=plan['source']
    command=[py,'-m','reproduction.sevenb.entry',str(planfile)]
    if not tree:command=[py,'-m','search_r1.baselines.preflight','--']+command
    script='\n'.join(['#!/bin/bash','set -euo pipefail',
        'export PATH='+q(str(env/'bin'))+':"$PATH"',
        'export PYTHONPATH='+q(source)+' PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1',
        'export RECURSIVE_SOURCE_COMMIT='+q(plan['evaluation_code_commit'])+' SEARCH_R1_EXECUTION_BACKEND=slurm_lumia',
        'export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false',
        'export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NCCL_DEBUG=WARN NCCL_NVLS_ENABLE=0',
        'export VLLM_USE_V1=0 VLLM_ATTENTION_BACKEND=XFORMERS VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_FLASHINFER_SAMPLER=0',
        'export TORCH_NCCL_AVOID_RECORD_STREAMS=1 SWANLAB_MODE=disabled WANDB_MODE=disabled',
        'export HF_HOME='+q(str(ROOT/'cache/hf'))+' XDG_CACHE_HOME='+q(str(ROOT/'cache'))+' TRITON_CACHE_DIR=/tmp/sevenb-triton-"$SLURM_JOB_ID"',
        'export C_INCLUDE_PATH=/home/jmzhang/Workspace/Search-G1-Question-Diversity-20261013-v12/bootstrap-tools/python-dev/usr/include/python3.10:/home/jmzhang/Workspace/Search-G1-Question-Diversity-20261013-v12/bootstrap-tools/python-dev/usr/include',
        'unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy PYTHONHOME RAY_ADDRESS',
        'cd '+q(source),'ulimit -n 65535','set +e',shlex.join(command),
        'rc=$?','echo "$rc" > '+q(str(root/'exit.code')),'exit "$rc"',''])
    cmd=['sbatch','--parsable','--hold','--no-requeue','--account=lumia','--qos=normal','--ntasks=1',
        '--job-name='+job,'--output='+str(root/'slurm-%j.log'),'--nodes=1','--gres=gpu:8',
        '--cpus-per-task=64','--mem=600G',
        '--time='+('7-00:00:00' if plan['kind']=='original' else '3-00:00:00')]
    if plan['node']=='node8':cmd+=['--partition=ADA6000','--nodelist=node8']
    elif plan['node']=='node11':cmd+=['--partition=L40S','--nodelist=node11']
    else:cmd+=['--partition=ADA6000,L40S']
    write_once(root/'submission-intent.json',dict(at=time.time(),job=job,plan_sha256=sha(planfile),command=cmd,script=script))
    r=subprocess.run(cmd,input=script,text=True,capture_output=True)
    write_once(root/'submission-response.json',dict(at=time.time(),returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))
    r.check_returncode();jid=r.stdout.strip().split(';')[0];assert jid.isdigit()
    sub=dict(at=time.time(),job=job,job_id=jid,plan=str(planfile),plan_sha256=sha(planfile),
        root=str(root),source=plan['evaluation_code_commit'],kind=plan['kind'],repeat_index=plan['repeat_index'],no_auto_retry=True)
    write_once(root/'submission.json',sub)
    subprocess.run(['scontrol','release',jid],check=True)
    print(json.dumps(sub),flush=True)


if __name__=='__main__':submit(sys.argv[1])
