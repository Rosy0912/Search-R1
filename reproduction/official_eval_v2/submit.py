"""Unique held-record-release submission; never automatically retry."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from .driver import once,verify
from .protocol import sha
from .prepare import ROOT

def submit(path):
 path=Path(path);plan=json.loads(path.read_text());root=Path(plan['root'])
 assert path.parent==ROOT/'plans' and root.parent==ROOT
 assert json.loads((root/'preflight.json').read_text())['plan_sha256']==sha(path)
 verify(plan)
 assert not (root/'submission-intent.json').exists() and not (root/'submission.json').exists()
 job=plan['job']
 assert not subprocess.check_output(['squeue','-h','--name='+job,'-o','%i|%T'],text=True).strip()
 assert not subprocess.check_output(['sacct','-X','-n','-P','-S','2026-10-19T00:00:00','--name='+job,'--format=JobIDRaw,State'],text=True).strip()
 env='/home/jmzhang/Workspace/Search-G1-Question-Diversity-20261013-v12/env';q=shlex.quote
 command=[env+'/bin/python3','-m','reproduction.official_eval_v2.driver',str(path)]
 script='\n'.join(['#!/bin/bash','set -euo pipefail',
  'export PATH='+q(env+'/bin')+':"$PATH"',
  'export PYTHONPATH='+q(plan['source'])+' PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1',
  'export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false',
  'export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 VLLM_ATTENTION_BACKEND=XFORMERS',
  'export VLLM_USE_V1=0 VLLM_WORKER_MULTIPROC_METHOD=spawn WANDB_MODE=disabled',
  'unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy PYTHONHOME RAY_ADDRESS',
  'cd '+q(plan['source']),'ulimit -n 65535','set +e',shlex.join(command),
  'rc=$?','echo "$rc" > '+q(str(root/'exit.code')),'exit "$rc"',''])
 cmd=['sbatch','--parsable','--hold','--no-requeue','--account=lumia','--qos=normal','--ntasks=1',
  '--job-name='+job,'--output='+str(root/'slurm-%j.log'),'--nodes=1','--gres=gpu:8',
  '--cpus-per-task=64','--mem=600G','--time=1-00:00:00','--partition=ADA6000,L40S,RTX4090,RTX3090']
 once(root/'submission-intent.json',dict(at=time.time(),job=job,plan_sha256=sha(path),command=cmd,script=script))
 result=subprocess.run(cmd,input=script,text=True,capture_output=True)
 once(root/'submission-response.json',dict(at=time.time(),returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
 result.check_returncode();jid=result.stdout.strip().split(';')[0];assert jid.isdigit()
 sub=dict(at=time.time(),job_id=jid,job=job,kind=plan['kind'],repeat=plan['repeat'],root=str(root),
  plan=str(path),plan_sha256=sha(path),source_commit=plan['source_commit'],no_auto_retry=True)
 once(root/'submission.json',sub)
 subprocess.run(['scontrol','release',jid],check=True)
 print(json.dumps(sub),flush=True)

if __name__=='__main__':submit(sys.argv[1])
