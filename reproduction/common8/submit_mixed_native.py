"""Once-only native submission after independently verified source deployment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time


def publish_exclusive(path,value):
    with path.open('x') as f:
        json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())


def submit(base,commit,expected_files):
    base=Path(base).resolve();assert str(base).startswith('/mnt/data/shaoshitong/')
    intent=base/'preparation/native-submit-intent.json';receipt=base/'preparation/native-submit-receipt.json'
    if intent.exists() or receipt.exists():
        raise RuntimeError('Submission already attempted; inspect intent, receipt, Slurm queue and accounting; never replay')
    for name,expected in expected_files.items():
        path=base/'repo'/name
        assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==expected, 'Deployed source differs: '+name
    assert not (base/'logs/run-manifest.json').exists()
    # Verify a fresh target and no duplicate native allocation before publishing intent.
    active=subprocess.check_output(['squeue','-h','-w','h100-8','-t','RUNNING,PENDING,COMPLETING','-o','%i|%j'],text=True)
    assert not active.strip(), 'Inspect existing node allocation before submitting'
    assets=Path('/mnt/data/shaoshitong/search-r1-rl/gigpo/assets-root/assets/wiki')
    assert (assets/'e5_Flat.index').is_file() and (assets/'wiki-18.jsonl').is_file()
    launcher=base/'preparation/launch-mixed-native.sh'
    text='#!/usr/bin/env bash\nset -euo pipefail\n'
    for key,value in dict(SEARCH_G1_BASE=str(base),SEARCH_G1_SOURCE_COMMIT=commit,
        SEARCH_G1_ALLOW_EXISTING_GPU_PROCESSES='1',SEARCH_G1_ROLLOUT_MEMORY_UTILIZATION='0.2',
        SEARCH_G1_LOCAL_RETRIEVAL_ASSETS=str(assets)).items():text+='export '+key+'='+shlex.quote(value)+'\n'
    text+='exec bash '+shlex.quote(str(base/'repo/reproduction/common8/launch_native.sh'))+'\n'
    with launcher.open('x') as f:f.write(text);f.flush();os.fsync(f.fileno())
    command=['sbatch','--parsable','--job-name=search-g1-mixed200','--partition=gpu',
        '--nodelist=h100-8','--nodes=1','--ntasks=1','--cpus-per-task=96','--gres=gpu:h100:8',
        '--mem=512G','--exclusive','--time=7-00:00:00','--signal=B:TERM@60',
        '--output='+str(base/'logs/native-%j.log'),'--error='+str(base/'logs/native-%j.log'),
        '--export=NONE',str(launcher)]
    publish_exclusive(intent,dict(created_unix=time.time(),base=str(base),source_commit=commit,
        verified_files=len(expected_files),command=command,shared_gpu_policy='preserve_bound_existing_processes'))
    r=subprocess.run(command,capture_output=True,text=True,timeout=30)
    result=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,checked_unix=time.time(),source_commit=commit)
    if r.returncode==0:
        job=r.stdout.strip().split(';')[0];assert job.isdecimal();result['slurm_job_id']=job
    publish_exclusive(receipt,result)
    print(json.dumps(result),flush=True)
    if r.returncode:raise RuntimeError('Native submission failed; inspect saved receipt')
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--base',required=True)
    parser.add_argument('--source-commit',required=True);parser.add_argument('--source-manifest',required=True)
    args=parser.parse_args();submit(args.base,args.source_commit,json.loads(Path(args.source_manifest).read_text()))
