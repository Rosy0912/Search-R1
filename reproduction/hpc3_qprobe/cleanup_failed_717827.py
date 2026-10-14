"""Close this observed failed job's private tmpfs on its original node only.

Slurm must already have terminated the job. Preserve original logs/claims:
cleanup uses a shadow ledger and checks all recorded readers are quiescent.
"""
import json
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time
from reproduction.common8.native_allocation import fields
from reproduction.common8.stage_retrieval import assert_quiescent
from reproduction.hpc3_qprobe.control import publish

FAILED=Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-4gpu-jxu957-recovery717765-20261014')
SHADOW=FAILED/'failure-cleanup717827'


def main():
    assert os.getuid()==14090 and socket.gethostname().split('.')[0]=='ACD1-67'
    assert os.environ.get('SLURM_JOB_ID') not in (None,'717827')
    proof=json.loads((FAILED/'failed-job-717827/terminal-and-files.json').read_text())
    old=fields(proof['terminal']['stdout'])
    assert old['JobId']=='717827' and old['JobState']=='FAILED' and old['UserId']=='jxu957(14090)' and old['NodeList']=='ACD1-67'
    actual=subprocess.check_output(['sacct','-nP','-j','717827','-X','--format=JobID,State,ExitCode,UID,NodeList'],text=True).strip()
    assert actual=='717827|FAILED|1:0|14090|ACD1-67'
    assert SHADOW.is_dir() and not (SHADOW/'complete.json').exists()
    run=json.loads((FAILED/'logs/run-manifest.json').read_text())
    assert run['completed_updates']==0 and run['training_world_size']==4 and run['source_commit']=='a75e0c944e2bbae284e52ce9b9f8bf909ab89564'
    from reproduction.common8.temporary_cache import cleanup as clean_corpus
    from reproduction.common8.stage_retrieval import cleanup as clean_stage
    from reproduction.a800.owned_processes import OwnedProcesses
    for name in ('corpus-cache-claim.json','retrieval-staging-claim.json'):
        assert_quiescent(SHADOW,json.loads((SHADOW/'logs'/name).read_text()))
    owned=OwnedProcesses(SHADOW/'logs/process-ledger.json',run['run_id'])
    # No live process is eligible after terminal Slurm teardown. Fail rather
    # than signal any recovered PID when the kernel has no pidfd support.
    assert all(not owned.observe(n) for n in owned.jobs)
    owned.stop_all()
    corpus=clean_corpus(SHADOW);stage=clean_stage(SHADOW)
    value=dict(cleaned=True,run_id=run['run_id'],failed_job='717827',cleanup_job=os.environ['SLURM_JOB_ID'],node=socket.gethostname(),actual_terminal=actual,corpus=corpus,stage=stage,original_logs_unchanged=True,checked_unix=time.time())
    publish(SHADOW/'complete.json',value);print(json.dumps(value),flush=True)


def submit(commit):
    assert os.getuid()==14090
    root=FAILED.parents[0]/'qprobe-teacher-warmup200-20261014'
    source=root/'source'/commit
    verification=json.loads((root/'preparation'/('source-'+commit+'-verification.json')).read_text())
    assert verification['status']=='PUBLISHED_SOURCE_FULL_SHA_PASS' and verification['commit']==commit
    for n,h in verification['files'].items():assert hashlib.sha256((source/n).read_bytes()).hexdigest()==h
    proof=json.loads((FAILED/'failed-job-717827/terminal-and-files.json').read_text())
    assert subprocess.check_output(['sacct','-nP','-j','717827','-X','--format=JobID,State,ExitCode,UID,NodeList'],text=True).strip()=='717827|FAILED|1:0|14090|ACD1-67'
    for n,h in proof['files'].items():assert hashlib.sha256((FAILED/n).read_bytes()).hexdigest()==h['sha256']
    intent=FAILED/'preparation/cleanup717827-submit-claim.json';receipt=FAILED/'preparation/cleanup717827-submission.json'
    assert not SHADOW.exists() and not intent.exists() and not receipt.exists(),'Inspect existing cleanup claim instead of repeating'
    command=['sbatch','--parsable','--account=jxu957','--qos=formal-user','--partition=acd_ue',
        '--nodelist=ACD1-67','--nodes=1','--ntasks=1','--cpus-per-task=1','--mem=1G','--gres=none',
        '--time=00:10:00','--job-name=qprobe-cleanup717827','--export=NONE',
        '--output='+str(FAILED/'logs/cleanup717827-%j.log'),str(source/'reproduction/hpc3_qprobe/cleanup_failed_717827.sh'),str(source)]
    test=subprocess.run(command[:1]+['--test-only']+command[1:],capture_output=True,text=True,timeout=30)
    assert test.returncode==0,test.stderr
    publish(intent,dict(commit=commit,command=command,failed_job='717827',epoch=time.time()))
    (SHADOW/'logs').mkdir(parents=True)
    for n in ('run-manifest.json','process-ledger.json','corpus-cache-claim.json','retrieval-staging-claim.json'):
        shutil.copyfile(FAILED/'logs'/n,SHADOW/'logs'/n)
    result=subprocess.run(command,capture_output=True,text=True,timeout=30)
    v=dict(commit=commit,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,epoch=time.time())
    if result.returncode==0:
        v['job']=result.stdout.strip().split(';')[0];assert v['job'].isdecimal()
    publish(receipt,v);assert result.returncode==0,result.stderr
    return v


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--submit-commit');a=p.parse_args()
    if a.submit_commit:print(json.dumps(submit(a.submit_commit)),flush=True)
    else:main()
