"""Submit this task's one small GPU diagnosis once, only after actual717607 failure."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from reproduction.hpc3_qprobe.control import publish


def submit(commit):
    base=Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-20261014')
    failed=Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-4gpu-jxu957-recovery717554-20261014')
    assert os.getuid()==14090
    accounting=subprocess.check_output(['sacct','-nP','-j','717607','-X','--format=JobID,State,ExitCode,UID'],text=True,timeout=30).strip()
    assert accounting=='717607|FAILED|1:0|14090','Actual own diagnostic parent must be FAILED'
    assert (failed/'failed-job-717607/terminal-and-files.json').is_file()
    run=json.loads((failed/'logs/run-manifest.json').read_text());assert run['completed_updates']==0
    assert not (failed/'logs/process-ledger.json').exists()
    root=base/'preparation/gpu-inventory717607';root.mkdir(exist_ok=True)
    claim=root/'submit-claim.json';receipt=root/'submission.json'
    assert not claim.exists() and not receipt.exists(),'Diagnostic already attempted; inspect saved actual job'
    proof=json.loads((base/'preparation'/('source-'+commit+'-verification.json')).read_text())
    assert proof['status']=='PUBLISHED_SOURCE_FULL_SHA_PASS' and proof['commit']==commit
    source=base/'source'/commit
    for name,sha in proof['files'].items():assert hashlib.sha256((source/name).read_bytes()).hexdigest()==sha,name
    command=['sbatch','--parsable','--account=jxu957','--qos=formal-user','--partition=acd_ue',
        '--nodes=1','--ntasks=1','--gres=gpu:1','--cpus-per-task=8','--mem=16G','--time=00:10:00',
        '--job-name=qprobe-gpu-inventory','--export=NONE','--chdir='+str(source),
        '--output='+str(root/'diagnostic-%j.log'),'--error='+str(root/'diagnostic-%j.log'),
        str(source/'reproduction/hpc3_qprobe/gpu_inventory_diagnostic.sh'),str(base),str(source)]
    publish(claim,dict(command=command,commit=commit,actual_failed_parent_accounting=accounting,epoch=time.time()))
    result=subprocess.run(command,capture_output=True,text=True,timeout=30)
    value=dict(commit=commit,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,epoch=time.time())
    if result.returncode==0:
        value['job']=result.stdout.strip().split(';')[0];assert value['job'].isdecimal()
    publish(receipt,value)
    if result.returncode:raise RuntimeError('Actual diagnostic submission failed; inspect receipt')
    return value


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--commit',required=True)
    print(json.dumps(submit(parser.parse_args().commit)),flush=True)
