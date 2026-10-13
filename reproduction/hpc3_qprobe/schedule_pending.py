"""Move only this still-pending owned job to an already eligible partition.

No submission/requeue/hold, source edit, resource reduction, model loading or
GPU allocation. Retain the original submission and record the single update.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

from reproduction.common8.native_allocation import fields


def show(*args):
    return subprocess.check_output(['scontrol','show']+list(args),universal_newlines=True)


def once(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())


def run(base):
    base=Path(base)
    if base!=Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-20261014') or base.resolve()!=base:
        raise ValueError('Scheduler helper requires the exact owned run namespace')
    submission=json.loads((base/'preparation/training-submission.json').read_text())
    claim=json.loads((base/'preparation/training-submit-claim.json').read_text())
    job=submission['job'];commit=submission['commit']
    if job!='716883' or submission['returncode']!=0 or claim['commit']!=commit:
        raise ValueError('Actual existing job/claim differs')
    source=base/'source'/commit;entry=source/'reproduction/hpc3_qprobe/train.sh'
    if (base/'repo').resolve()!=source or not entry.is_file():raise ValueError('Queued immutable source differs')
    groups=set(subprocess.check_output(['id','-nG'],universal_newlines=True).split())
    old=fields(show('partition','acd_u'));new=fields(show('partition','acd_ue'))
    for partition in (old,new):
        if partition['State']!='UP' or partition['RootOnly']!='NO' or partition['ReqResv']!='NO' or partition['PreemptMode']!='OFF':
            raise ValueError('Partition eligibility/preemption policy changed')
        if not groups.intersection(partition['AllowGroups'].split(',')):raise ValueError('Current user is not eligible')
        if partition['AllowAccounts']!='ALL' or partition['AllowQos']!='ALL':raise ValueError('Unexpected account/QOS restriction')
    if old['Nodes']!=new['Nodes'] or old['MaxTime']!=new['MaxTime'] or int(new['PriorityTier'])<=int(old['PriorityTier']):
        raise ValueError('Destination does not preserve the same nodes/time with higher configured tier')
    before_text=show('job',job);before=fields(before_text)
    if before['JobState']!='PENDING' or before['Partition']!='acd_u' or before['AllocTRES']!='(null)':
        raise ValueError('Inspect existing handle; only the original still-pending job can be updated')
    if before['UserId']!='mou176('+str(os.geteuid())+')' or before['Command']!=str(entry):
        raise ValueError('Job ownership or actual immutable entry differs')
    expected={'JobId':job,'JobName':'qprobe-warmup200','QOS':'normal','Account':'mou176',
        'NumNodes':'1-1','NumCPUs':'96','NumTasks':'1','CPUs/Task':'96','MinMemoryNode':'512G',
        'TresPerNode':'gres/gpu:8','TimeLimit':'7-00:00:00','OverSubscribe':'NO','BatchFlag':'1','Restarts':'0'}
    if any(before.get(k)!=v for k,v in expected.items()):raise ValueError('Original full eight-GPU resources changed')
    if (base/'logs/run-manifest.json').exists() or (base/'logs/training-716883.log').exists():
        raise ValueError('Native execution already began; do not change its scheduling')
    entry_sha=hashlib.sha256(entry.read_bytes()).hexdigest()
    intent=base/'preparation/scheduler-partition-update-716883-intent.json'
    receipt=base/'preparation/scheduler-partition-update-716883.json'
    if intent.exists() or receipt.exists():raise ValueError('Update already attempted; inspect actual state, never replay')
    command=['scontrol','update','JobId='+job,'Partition=acd_ue']
    once(intent,dict(job=job,source_commit=commit,entry_sha256=entry_sha,before=before_text,
        original_partition=old,destination_partition=new,current_groups=sorted(groups),
        command=command,epoch=time.time(),submission=False,requeue=False,resource_changes=False))
    if fields(show('job',job))['JobState']!='PENDING':raise ValueError('Job started before update; leave it alone')
    p=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=30)
    after_text=show('job',job);after=fields(after_text)
    result=dict(job=job,source_commit=commit,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,
        after=after_text,epoch=time.time(),submission=False,requeue=False,
        actual_job_state_after=after['JobState'],training_started=after['JobState'] in ('CONFIGURING','RUNNING','COMPLETING'))
    # A successful scheduler cycle can allocate this same job before readback.
    # Slurm then renders NumNodes as 1 rather than the pending request 1-1.
    persistent=tuple(k for k in expected if k!='NumNodes')+('Command','SubmitTime','ReqTRES')
    same=all(after.get(k)==before.get(k) for k in persistent) and after.get('NumNodes') in ('1','1-1')
    same=same and (base/'repo').resolve()==source and hashlib.sha256(entry.read_bytes()).hexdigest()==entry_sha
    result['immutable_source_and_original_resources_equal']=same
    result['status']='EXISTING_PENDING_JOB_PARTITION_UPDATED' if p.returncode==0 and same and after['Partition']=='acd_ue' else 'PARTITION_UPDATE_NOT_ACCEPTED'
    once(receipt,result)
    if result['status']!='EXISTING_PENDING_JOB_PARTITION_UPDATED':raise ValueError('Inspect recorded scheduler update; no automatic replay or replacement')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);a=p.parse_args()
    print(json.dumps(run(a.base)))
