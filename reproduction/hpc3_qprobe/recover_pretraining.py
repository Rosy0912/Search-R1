"""Once-only recovery of the actual pretraining allocation failure 717554.

Use a fresh target, preserve the terminal source/logs, and reject any prior
actor/teacher pipeline work. No live job, shared asset or old binding is edited.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from reproduction.hpc3_qprobe.control import prepare, link, publish
from reproduction.common8.native_allocation import fields

ROOT=Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-20261014')
FAILED=Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-4gpu-jxu957-20261014')
TARGET=Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-4gpu-jxu957-recovery717554-20261014')
FAILED_COMMIT='300734e78705936b9b3885248ac3ffe7f8cbd3aa'
JOB='717554'


def terminal():
    if os.getuid()!=14090:raise ValueError('Actual jxu957 owner required')
    detail=subprocess.check_output(['scontrol','show','job','-d',JOB],text=True,timeout=30)
    actual=fields(detail)
    expected=dict(JobId=JOB,JobState='FAILED',UserId='jxu957(14090)',Account='jxu957',
        QOS='formal-user',WorkDir=str(FAILED),Restarts='0',
        Command=str(ROOT/'source'/FAILED_COMMIT/'reproduction/hpc3_qprobe/train.sh'))
    if any(actual.get(k)!=v for k,v in expected.items()):raise ValueError('Not the actual failed own pretraining job')
    prior=json.loads((ROOT/'preparation/account-transition-submission.json').read_text())
    if prior['job']!=JOB or prior['returncode']!=0:raise ValueError('Missing original submission binding')
    run=json.loads((FAILED/'logs/run-manifest.json').read_text())
    if run['source_commit']!=FAILED_COMMIT or run['completed_updates']!=0 or run['training_world_size']!=4:
        raise ValueError('Actual failed run differs or already trained')
    for name in ('process-ledger.json','native-allocation-admission.json','training-launch-claim.json',
                 'pipeline-status.json','search-g1-common8-training.log'):
        if (FAILED/'logs'/name).exists():raise ValueError('Pipeline work already claimed; use full-state recovery instead')
    log=FAILED/'logs'/('training-'+JOB+'.log')
    if 'validate_native_four' not in log.read_text() or 'AssertionError' not in log.read_text():
        raise ValueError('Known actual allocation failure absent')
    evidence=FAILED/('failed-job-'+JOB)
    if not (evidence/'terminal-and-files.json').is_file():raise ValueError('Preserve failure evidence before recovery')
    preserved=json.loads((evidence/'terminal-and-files.json').read_text())
    for name,record in preserved['files'].items():
        if name.startswith('logs/') and hashlib.sha256((FAILED/name).read_bytes()).hexdigest()!=record['sha256']:
            raise ValueError('Preserved terminal log changed')
    return dict(actual=actual,detail=detail,prior=prior,failed_run_id=run['run_id'],checked_unix=time.time())


def fresh(commit):
    actual=terminal()
    intent=ROOT/'preparation/recovery717554-prepare-intent.json'
    if intent.exists() or TARGET.exists():raise ValueError('Recovery preparation already attempted; inspect instead of replaying')
    proof=json.loads((ROOT/'preparation'/('source-'+commit+'-verification.json')).read_text())
    source=ROOT/'source'/commit
    if proof['status']!='PUBLISHED_SOURCE_FULL_SHA_PASS' or proof['commit']!=commit:
        raise ValueError('Repair must be published and verified')
    for name,sha in proof['files'].items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest()!=sha:raise ValueError('Repair bytes differ: '+name)
    os.umask(0o007)
    publish(intent,dict(commit=commit,target=str(TARGET),failure=actual,epoch=time.time()))
    TARGET.mkdir(mode=0o770)
    prepare(TARGET);link(TARGET/'repo',source)
    (TARGET/'source').mkdir();link(TARGET/'source'/commit,source)
    link(TARGET/'question-features',ROOT/'question-features')
    for name in ('features-ready.json','official-model-manifest.json'):
        link(TARGET/'preparation'/name,ROOT/'preparation'/name)
    policy=json.loads((FAILED/'preparation/personal-storage-authorization.json').read_text())
    if policy['budget_bytes']!=2*10**12 or policy['scope']!='jxu957_native4_qprobe_only':raise ValueError('Actual user budget differs')
    publish(TARGET/'preparation/personal-storage-authorization.json',policy)
    publish(TARGET/'preparation/recovery-source-binding.json',dict(source_commit=commit,source=str(source),
        target=str(TARGET),failed_job=JOB,failure=actual,source_files=len(proof['files']),
        training_world_size=4,actor_micro_batch_per_rank=4,feature_cache_reused=True,
        optimizer_writes_before_recovery=0,teacher_labels_before_recovery=0))
    return dict(status='FRESH_FAILED_PRETRAINING_RECOVERY_PREPARED',target=str(TARGET),commit=commit)


def submit(commit):
    actual=terminal()
    binding=json.loads((TARGET/'preparation/recovery-source-binding.json').read_text())
    source=ROOT/'source'/commit
    if binding['source_commit']!=commit or (TARGET/'repo').resolve()!=source or (TARGET/'source'/commit).resolve()!=source:
        raise ValueError('Fresh recovery source binding differs')
    claim=TARGET/'preparation/training-submit-claim.json';receipt=TARGET/'preparation/training-submission.json'
    if claim.exists() or receipt.exists():raise ValueError('Recovery submission already attempted; never duplicate')
    jobs=subprocess.check_output(['squeue','-h','-u','jxu957','-o','%i|%j|%T'],text=True,timeout=30)
    if any('qprobe' in line for line in jobs.splitlines()):raise ValueError('Existing own qprobe job requires inspection')
    from search_g1.qprobe_features import FeatureCache
    features=FeatureCache(TARGET/'question-features');features.verify_all()
    if features.manifest['count']!=169028:raise ValueError('Actual feature cache incomplete')
    command=['sbatch','--parsable','--account=jxu957','--qos=formal-user','--partition=acd_ue',
        '--nodes=1','--ntasks=1','--gres=gpu:4','--cpus-per-task=48','--mem=512G','--time=7-00:00:00',
        '--job-name=qprobe-warmup200','--export=NONE','--chdir='+str(TARGET),
        '--output='+str(TARGET/'logs/training-%j.log'),'--error='+str(TARGET/'logs/training-%j.log'),
        str(source/'reproduction/hpc3_qprobe/train.sh'),str(TARGET),commit]
    test=subprocess.run(command[:1]+['--test-only']+command[1:],capture_output=True,text=True,timeout=30)
    publish(TARGET/'preparation/training-test-only.json',dict(command=command,exit=test.returncode,stdout=test.stdout,stderr=test.stderr,epoch=time.time()))
    if test.returncode:raise ValueError('Actual recovery scheduler preflight failed')
    publish(claim,dict(commit=commit,command=command,recovery_of_failed_job=JOB,failure=actual,epoch=time.time()))
    result=subprocess.run(command,capture_output=True,text=True,timeout=30)
    value=dict(commit=commit,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,epoch=time.time(),recovery_of_failed_job=JOB)
    if result.returncode==0:
        value['job']=result.stdout.strip().split(';')[0]
        if not value['job'].isdecimal():raise ValueError('Invalid actual recovery job')
    publish(receipt,value)
    if result.returncode:raise RuntimeError('Recovery submission failed; inspect saved receipt')
    return value


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('prepare','submit'));parser.add_argument('--commit',required=True)
    args=parser.parse_args();print(json.dumps({'prepare':fresh,'submit':submit}[args.mode](args.commit)),flush=True)
