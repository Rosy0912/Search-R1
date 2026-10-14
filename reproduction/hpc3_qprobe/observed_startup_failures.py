"""Owner re-verification of the two actual failures preserved by an observer.

An observer receipt is never promoted by renaming/chowning it. The real owner
must freshly verify accounting and immutable submission/log bytes, then publish
a new, explicitly accounting-based owner receipt. No synthetic scontrol text.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from reproduction.hpc3_qprobe.control import publish

ROOT=Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-20261014')
SOURCE='594d43b73c5a1e5159a24fb0d97880ff22fb0056'
CASES={
 '717938':dict(parent='717827',node='ACD1-12',partition='emergency_acd',
     receipt='training-submission.json',claim='training-submit-claim.json',
     entry='train.sh'),
 '717934':dict(parent='717765',node='ACD1-67',partition='acd_ue',
     receipt='cleanup717827-submission.json',claim='cleanup717827-submit-claim.json',
     entry='cleanup_failed_717827.sh')}


def base_for(job):
    if job not in CASES:raise ValueError('Only actual observed own startup failures are supported')
    return ROOT.parent/('qprobe-teacher-warmup200-4gpu-jxu957-recovery'+CASES[job]['parent']+'-20261014')


def actual_terminal(job):
    case=CASES[job]
    cmd=['sacct','-nP','-j',job,'-X','--format=JobID,State,ExitCode,User,UID,Account,QOS,Partition,Start,End,NodeList']
    raw=subprocess.check_output(cmd,text=True,timeout=30)
    rows=[line.split('|') for line in raw.splitlines() if line.strip()]
    if len(rows)!=1 or rows[0][:8]!=[job,'FAILED','1:0','jxu957','14090','jxu957','formal-user',case['partition']]:
        raise ValueError('Actual owned terminal failure is missing or differs')
    row=rows[0]
    if len(row)!=11 or row[-1]!=case['node'] or any(x in ('Unknown','None','') for x in row[8:10]):
        raise ValueError('Actual terminal birth/end/node is missing')
    return dict(command=cmd,stdout=raw,row=row,checked_unix=time.time())


def verify(job):
    """Read-only cross-check; caller identity is recorded, never inferred."""
    base=base_for(job);case=CASES[job]
    marker=base/('failed-job-'+job+'-observer/terminal-and-files.json')
    if marker.is_symlink() or not marker.is_file():raise ValueError('Original observer archive required')
    archived=json.loads(marker.read_text())
    if archived['kind']!='SHARED_ACCOUNT_OBSERVER_ONLY' or archived['job']!=job or archived['source_commit']!=SOURCE or archived['owner_recovery_authorized_by_this_record'] is not False:
        raise ValueError('Observer identity or source differs')
    now=actual_terminal(job)
    if archived['accounting_before']['rows'][job]!=now['row'] or archived['accounting_after']['rows'][job]!=now['row']:
        raise ValueError('Fresh terminal differs from preserved observer accounting')
    required={'preparation/'+case['receipt'],'preparation/'+case['claim']}
    if job=='717938':required|={'logs/run-manifest.json','logs/process-ledger.json','logs/cleanup-status.json','logs/pipeline-status.json','logs/model-initial.log','logs/training-717938.log'}
    else:required|={'logs/cleanup717827-717934.log','failure-cleanup717827/logs/run-manifest.json',
                   'failure-cleanup717827/logs/process-ledger.json','failure-cleanup717827/logs/corpus-cache-claim.json','failure-cleanup717827/logs/retrieval-staging-claim.json'}
    if not required.issubset(archived['files']):raise ValueError('Required preserved failure evidence missing')
    for name,record in archived['files'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('Archive path escape')
        original=base/name;copy=marker.parent/name
        if original.is_symlink() or copy.is_symlink() or original.stat().st_uid!=14090 or record['source_uid']!=14090:
            raise ValueError('Original owned failure bytes were aliased or replaced')
        if original.stat().st_size!=record['bytes'] or hashlib.sha256(original.read_bytes()).hexdigest()!=record['sha256'] or hashlib.sha256(copy.read_bytes()).hexdigest()!=record['sha256']:
            raise ValueError('Original/archive full SHA differs: '+name)
    prior=json.loads((base/'preparation'/case['receipt']).read_text())
    claim=json.loads((base/'preparation'/case['claim']).read_text())
    command=claim['command'];entry=str(ROOT/'source'/SOURCE/'reproduction/hpc3_qprobe'/case['entry'])
    if prior.get('job')!=job or prior.get('commit')!=SOURCE or prior.get('returncode')!=0 or claim.get('commit')!=SOURCE:
        raise ValueError('Original immutable own submission differs')
    if command[0]!='sbatch' or '--account=jxu957' not in command or '--qos=formal-user' not in command or entry not in command:
        raise ValueError('Original entry/account/QOS binding differs')
    source=json.loads((ROOT/'preparation'/('source-'+SOURCE+'-verification.json')).read_text())
    if source['status']!='PUBLISHED_SOURCE_FULL_SHA_PASS' or source['commit']!=SOURCE:
        raise ValueError('Original source was not fully published')
    for name,sha in source['files'].items():
        p=ROOT/'source'/SOURCE/name
        if p.is_symlink() or hashlib.sha256(p.read_bytes()).hexdigest()!=sha:raise ValueError('Original immutable source differs')
    run=json.loads((base/'logs/run-manifest.json').read_text())
    if run['completed_updates']!=0 or run['training_world_size']!=4:raise ValueError('Not an actual zero-training native4 startup failure')
    if job=='717938':
        if run['source_commit']!=SOURCE or command[-3:]!=[entry,str(base),SOURCE] or '--chdir='+str(base) not in command or '--gres=gpu:4' not in command or '--cpus-per-task=48' not in command or '--mem=512G' not in command:
            raise ValueError('Original full four-GPU command/workdir differs')
        for name in ('training-launch-claim.json','search-g1-common8-training.log','actual-updates.jsonl','actual-attempts.jsonl'):
            if (base/'logs'/name).exists():raise ValueError('Actual trainer work requires full-state recovery')
        for name in ('teacher-warmup-cache','checkpoints','artifacts/question-probe'):
            p=base/name
            if p.exists() and any(p.iterdir()):raise ValueError('Teacher/checkpoint state requires full-state recovery')
        stage=json.loads((base/'logs/pipeline-status.json').read_text());cleanup=json.loads((base/'logs/cleanup-status.json').read_text());ledger=json.loads((base/'logs/process-ledger.json').read_text())
        if stage['stage']!='FAILED' or stage['error']!="AssertionError('model-initial failed; see project log')" or cleanup['cleaned'] is not True:
            raise ValueError('Observed closed model-only failure differs')
        if any(d['run_id']!=run['run_id'] for d in (stage,cleanup,ledger)) or set(ledger['jobs'])!={'gpu-preflight','model-initial'}:
            raise ValueError('Other pipeline work exists')
        if any(ledger['jobs'][n]['state']!='STOPPED' or ledger['jobs'][n].get('exit_code')!=code for n,code in [('gpu-preflight',0),('model-initial',1)]):
            raise ValueError('Startup ledger is live or differs')
        if 'Internal and actual separate actor logprob entrypoints disagree' not in (base/'logs/model-initial.log').read_text():
            raise ValueError('Actual internal LP failure absent')
    else:
        if run['source_commit']!='a75e0c944e2bbae284e52ce9b9f8bf909ab89564':
            raise ValueError('Original failed827 run source differs')
        if command[-2:]!=[entry,str(ROOT/'source'/SOURCE)] or '--nodelist=ACD1-67' not in command:
            raise ValueError('Original cleanup node/source differs')
        if "KeyError: 'pid'" not in (base/'logs/cleanup717827-717934.log').read_text() or (base/'failure-cleanup717827/complete.json').exists():
            raise ValueError('Observed failed cleanup absent or already completed')
    return dict(kind='FRESH_ACCOUNTING_AND_IMMUTABLE_SUBMISSION_BINDING',job=job,actual=now,
        base=str(base),source_commit=SOURCE,source_command=entry,original_submission=prior,
        original_claim=claim,run_id=run['run_id'],observer_manifest_sha256=hashlib.sha256(marker.read_bytes()).hexdigest(),
        original_files=archived['files'],verified_by_uid=os.getuid(),checked_unix=time.time())


def adopt(job):
    if os.getuid()!=14090:raise ValueError('Actual jxu957 owner session required; observer cannot adopt')
    value=verify(job)
    directory=base_for(job)/('failed-job-'+job);path=directory/'owner-accounting-verification.json'
    if path.exists() or directory.exists():raise ValueError('Owner adoption already attempted; inspect instead of replaying')
    directory.mkdir(mode=0o770)
    value.update(kind='ACTUAL_OWNER_ACCOUNTING_REVERIFICATION',owner_uid=14090,
        observer_receipt_not_promoted=True,synthetic_scontrol_stdout=False)
    publish(path,value)
    return value


def owner_verified(job):
    if os.getuid()!=14090:raise ValueError('Actual owner required before recovery')
    path=base_for(job)/('failed-job-'+job+'/owner-accounting-verification.json')
    if path.is_symlink() or not path.is_file() or path.stat().st_uid!=14090:
        raise ValueError('New actual owner verification required')
    saved=json.loads(path.read_text());now=verify(job)
    immutable=('job','base','source_commit','source_command','original_submission','original_claim','run_id','observer_manifest_sha256','original_files')
    if saved['kind']!='ACTUAL_OWNER_ACCOUNTING_REVERIFICATION' or saved['owner_uid']!=14090 or saved.get('synthetic_scontrol_stdout') is not False or any(saved[k]!=now[k] for k in immutable) or saved['actual']['row']!=now['actual']['row']:
        raise ValueError('Owner verification changed or differs from fresh terminal')
    return now


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('verify','adopt'));p.add_argument('job',choices=tuple(CASES));a=p.parse_args()
    print(json.dumps({'verify':verify,'adopt':adopt}[a.mode](a.job)),flush=True)
