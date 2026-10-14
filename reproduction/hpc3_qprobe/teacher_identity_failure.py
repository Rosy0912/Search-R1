"""Preserve/reverify the actual native718322 first-teacher generation failure."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from reproduction.hpc3_qprobe.control import publish
from search_g1.qprobe_features import sha_file

ROOT=Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-20261014')
BASE=ROOT.parent/'qprobe-teacher-warmup200-4gpu-jxu957-recovery718288-20261014'
SOURCE='c5961481b68d483e578f20d6786d314fcdabaa07'
JOB='718322'
RUN='a85d7644f9eb4d7b9d2a95cc51a4879f'
TERMINAL=[JOB,'FAILED','1:0','jxu957','14090','jxu957','formal-user','emergency_acd',
          '2026-10-14T18:30:49','2026-10-14T19:05:06','ACD1-1']


def validate_terminal(raw):
    if [line.split('|') for line in raw.splitlines() if line.strip()] != [TERMINAL]:
        raise ValueError('Exact genuinely failed own teacher allocation required')


def verify():
    if os.getuid()!=14090:raise ValueError('Actual jxu957 owner required')
    raw=subprocess.check_output(['sacct','-nP','-j',JOB,'-X',
        '--format=JobID,State,ExitCode,User,UID,Account,QOS,Partition,Start,End,NodeList'],text=True,timeout=30)
    validate_terminal(raw)
    run=json.loads((BASE/'logs/run-manifest.json').read_text())
    prior=json.loads((BASE/'preparation/training-submission.json').read_text())
    claim=json.loads((BASE/'preparation/training-submit-claim.json').read_text())
    entry=str(ROOT/'source'/SOURCE/'reproduction/hpc3_qprobe/train.sh')
    if (run['run_id']!=RUN or run['source_commit']!=SOURCE or run['training_world_size']!=4 or
        run['completed_updates']!=0 or prior.get('job')!=JOB or prior.get('returncode')!=0 or
        prior.get('commit')!=SOURCE or claim['commit']!=SOURCE or claim['command'][-3:]!=[entry,str(BASE),SOURCE]):
        raise ValueError('Original native4 source/owner submission differs')
    for key in ['--account=jxu957','--qos=formal-user','--partition=emergency_acd','--gres=gpu:4',
                '--cpus-per-task=48','--mem=512G','--chdir='+str(BASE)]:
        if key not in claim['command']:raise ValueError('Original resource binding differs')
    allocation=json.loads((BASE/'logs/native-allocation-admission.json').read_text())
    if not allocation['accepted'] or allocation['run_id']!=RUN or allocation['slurm_job_id']!=JOB:
        raise ValueError('Original actual native allocation differs')
    for name in ['cleanup-status.json','retrieval-staging-cleanup.json','corpus-cache-cleanup.json']:
        d=json.loads((BASE/'logs'/name).read_text())
        if not d['cleaned'] or d['run_id']!=RUN:raise ValueError('Actual private resource cleanup required')
    ledger=json.loads((BASE/'logs/process-ledger.json').read_text())
    stage=json.loads((BASE/'logs/pipeline-status.json').read_text())
    if ledger['run_id']!=RUN or any(x['state']!='STOPPED' for x in ledger['jobs'].values()) or stage['stage']!='FAILED':
        raise ValueError('Owned processes still active or unexpected pipeline failure')
    text=(BASE/'logs/search-g1-common8-training.log').read_text()
    if ('QPROBE_POLICY_SNAPSHOT_CHANGED_DURING_GENERATION' not in text or
        'qprobe_warmup.py' not in text or 'generation_id' not in text):
        raise ValueError('New actual error requires diagnosis')
    for name in ['actual-updates.jsonl','actual-attempts.jsonl']:
        if (BASE/'logs'/name).exists():raise ValueError('Actual formal work requires a different recovery')
    cache=BASE/'teacher-warmup-cache'
    if {p.name for p in cache.iterdir()}!={'identity.json','writer.lock'}:
        raise ValueError('Teacher data exists; do not discard or silently change its namespace')
    identity=json.loads((cache/'identity.json').read_text())
    if identity['training_world_size']!=4 or len(identity['questions'])!=5120 or len({q['qid'] for q in identity['questions']})!=5120:
        raise ValueError('Original actual teacher pool differs')
    for name in ['artifacts/question-probe/completed','artifacts/question-probe/completed-pairs',
                 'artifacts/question-probe/teacher-warmup-ready.json','checkpoints']:
        p=BASE/name
        if p.is_file() or (p.exists() and any(x.is_file() or x.is_symlink() for x in p.rglob('*'))):
            raise ValueError('Unexpected completed head/actor/teacher work')
    import torch
    states=[]
    for rank in range(4):
        p=BASE/'artifacts/question-probe/teacher-warmup-boundary/before'/('rank-'+str(rank)+'.pt')
        d=torch.load(p,map_location='cpu',weights_only=False)
        if (d['run_id']!=RUN or d['rank']!=rank or d['world_size']!=4 or d['actual_optimizer_writes'] or
            d['completed_outer'] or d['last_attempt_index'] or d['used_attempt_ids']):
            raise ValueError('Actual actor0 four-rank boundary required')
        states.append(dict(rank=rank,path=str(p),bytes=p.stat().st_size,sha256=sha_file(p),
                           actual_optimizer_writes=0,completed_outer=0))
    proof=json.loads((ROOT/'preparation'/('source-'+SOURCE+'-verification.json')).read_text())
    if proof['status']!='PUBLISHED_SOURCE_FULL_SHA_PASS' or proof['commit']!=SOURCE:raise ValueError('Original source publication missing')
    for name,sha in proof['files'].items():
        if sha_file(ROOT/'source'/SOURCE/name)!=sha:raise ValueError('Original source changed')
    return dict(kind='ACTUAL_NATIVE4_PRE_FORMAL_TEACHER_IDENTITY_TRANSPORT_FAILURE',job=JOB,
        base=str(BASE),source_commit=SOURCE,run_id=RUN,actual_accounting=raw,prior=prior,
        original_claim=claim,actual_fresh_worker_states=states,source_files=len(proof['files']),
        actual_optimizer_writes=0,completed_training_outers=0,committed_teacher_trajectories=0,
        original_teacher_identity_sha256=sha_file(cache/'identity.json'),verified_by_uid=os.getuid(),checked_unix=time.time())


def archive():
    value=verify();directory=BASE/('failed-job-'+JOB)
    if directory.exists():raise ValueError('Archive already attempted; inspect rather than replay')
    files={}
    for folder in ['logs','preparation','artifacts','teacher-warmup-cache']:
        for p in sorted((BASE/folder).rglob('*')):
            if p.is_file() and not p.is_symlink():
                st=p.stat();sha=sha_file(p);after=p.stat()
                if st.st_uid!=14090 or (st.st_ino,st.st_size,st.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns):
                    raise ValueError('Terminal evidence not owned or still changing')
                files[str(p.relative_to(BASE))]=dict(path=str(p),bytes=st.st_size,sha256=sha,
                                                   inode=st.st_ino,mtime_ns=st.st_mtime_ns)
    directory.mkdir(mode=0o770)
    for name,row in files.items():
        if row['bytes']<=16*2**20:
            target=directory/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(BASE/name,target)
            if sha_file(target)!=row['sha256']:raise ValueError('Failure copy SHA differs')
    publish(directory/'terminal-and-files.json',dict(value,files=files,large_files_retained_in_place=True))
    return value


def preserved():
    p=BASE/('failed-job-'+JOB)/'terminal-and-files.json'
    if p.is_symlink() or p.stat().st_uid!=14090:raise ValueError('Actual owner preservation required')
    saved=json.loads(p.read_text());now=verify()
    for key in ['kind','job','base','source_commit','run_id','actual_accounting','original_claim','prior',
                'actual_fresh_worker_states','original_teacher_identity_sha256']:
        if saved[key]!=now[key]:raise ValueError('Exact preserved failure changed')
    for name,row in saved['files'].items():
        path=Path(row['path']);st=path.stat()
        if (st.st_ino,st.st_size,st.st_mtime_ns)!=(row['inode'],row['bytes'],row['mtime_ns']) or sha_file(path)!=row['sha256']:
            raise ValueError('Original retained failure SHA changed')
        copy=p.parent/name
        if copy.exists() and sha_file(copy)!=row['sha256']:raise ValueError('Preserved copy changed')
    return now


if __name__=='__main__':print(json.dumps(archive()),flush=True)
