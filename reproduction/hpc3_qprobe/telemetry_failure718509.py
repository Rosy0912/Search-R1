"""Preserve the actual teacher1024 telemetry failure; reuse only verified chunks.

GPU context-query cleanup was unavailable. Never replace it with a fabricated
cleaned receipt: quarantine the old node and require a separate native process
birth readback there before final goal acceptance.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time

from reproduction.hpc3_qprobe.control import publish
from search_g1.qprobe_features import sha_file

ROOT=Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-20261014')
BASE=ROOT.parent/'qprobe-teacher-warmup200-4gpu-jxu957-recovery718322-20261014'
SOURCE='6537fea1d9abf0a97a1923554a9298a37af1c69a'
JOB='718509'
RUN='0742d8e346284d6da9656a826ce14395'
NODE='ACD1-31'
TERMINAL=[JOB,'FAILED','1:0','jxu957','14090','jxu957','formal-user','emergency_acd',
          '2026-10-14T19:17:51','2026-10-14T19:58:34',NODE]
ERROR="TimeoutExpired(['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'], 30)"


def need(ok,reason):
    if not ok:raise ValueError(reason)


def validate_terminal(raw):
    need([x.split('|') for x in raw.splitlines() if x.strip()]==[TERMINAL],
         'Exact terminal native718509 owner/source allocation required')


def cache_readback(root):
    from search_g1.qprobe_warmup import TeacherCache
    root=Path(root).resolve();identity=json.loads((root/'identity.json').read_text())
    need(identity['training_world_size']==4 and len(identity['questions'])==5120 and
         len({x['qid'] for x in identity['questions']})==5120,'Canonical teacher5120/native4 differs')
    need({p.name for p in root.iterdir()}=={'identity.json','writer.lock','chunk-00.json','chunk-01.json',
         'rollout-00.pt','rollout-01.pt'},'Actual committed two-chunk cache differs; diagnose new progress')
    cache=TeacherCache.__new__(TeacherCache);cache.root=root;cache.identity=identity
    chunks=[cache.read_chunk(i) for i in range(2)]
    import torch
    ids=[];states=[]
    for chunk in chunks:
        ids.extend(t for g in chunk['groups'] for t in g['trajectory_ids'])
        for phase in ('before','after'):
            for row in chunk['frozen_workers'][phase]:
                proof=row['rng_receipt'];p=Path(proof['path'])
                need(p.is_file() and not p.is_symlink() and p.stat().st_uid==14090 and
                     p.stat().st_size==proof['bytes'] and sha_file(p)==proof['sha256'],
                     'Actual original worker RNG receipt changed')
                d=torch.load(p,map_location='cpu',weights_only=False)
                need(d['run_id']==RUN and d['world_size']==4 and d['rank']==row['rank'] and
                     d['actual_optimizer_writes']==d['completed_outer']==d['last_attempt_index']==0 and
                     d['used_attempt_ids']==[],'Formal actor work exists; require full-state recovery')
                states.append(dict(rank=d['rank'],path=str(p),sha256=proof['sha256']))
    need(len(ids)==5120 and len(set(ids))==5120,'Actual trajectory identities missing/duplicated')
    return dict(questions=1024,trajectories=5120,identity_sha256=sha_file(root/'identity.json'),
                raw_bytes=sum(c['raw_rollout']['bytes'] for c in chunks),original_worker_states=states,
                files={p.name:dict(bytes=p.stat().st_size,sha256=sha_file(p))
                       for p in root.iterdir() if p.is_file() and p.name!='writer.lock'})


def verify():
    need(os.getuid()==14090,'Actual jxu957 owner required')
    raw=subprocess.check_output(['sacct','-nP','-j',JOB,'-X',
        '--format=JobID,State,ExitCode,User,UID,Account,QOS,Partition,Start,End,NodeList'],text=True,timeout=30)
    validate_terminal(raw)
    run=json.loads((BASE/'logs/run-manifest.json').read_text())
    prior=json.loads((BASE/'preparation/training-submission.json').read_text())
    claim=json.loads((BASE/'preparation/training-submit-claim.json').read_text())
    need(run['run_id']==RUN and run['source_commit']==SOURCE and run['training_world_size']==4 and
         run['completed_updates']==0 and prior.get('job')==JOB and prior.get('returncode')==0 and
         prior['commit']==claim['commit']==SOURCE and claim['command'][-3:]==[
             str(ROOT/'source'/SOURCE/'reproduction/hpc3_qprobe/train.sh'),str(BASE),SOURCE],
         'Original native4 source/claim differs')
    for arg in ('--account=jxu957','--qos=formal-user','--partition=emergency_acd','--gres=gpu:4',
                '--cpus-per-task=48','--mem=512G','--chdir='+str(BASE)):
        need(arg in claim['command'],'Original actual resource binding differs')
    stage=json.loads((BASE/'logs/pipeline-status.json').read_text())
    need(stage['run_id']==RUN and stage['stage']=='FAILED' and
         stage['error']=='GPU sampling did not finish durably: '+ERROR,'Unknown actual failure')
    log=(BASE/'logs/training-718509.log').read_text()
    need('AssertionError: GPU sampling failed: '+ERROR in log and
         '--query-compute-apps=gpu_uuid,pid' in log and 'timed out after 30 seconds' in log,
         'Actual telemetry and final context-query failure absent')
    train=(BASE/'logs/search-g1-common8-training.log').read_text()
    need('"completed_questions": 1024' in train and not any(x in train for x in (
        'Traceback (most recent call last)','torch.OutOfMemoryError','CUDA error:','QPROBE_POLICY_SNAPSHOT_CHANGED')),
         'New actual trainer failure requires separate diagnosis')
    ledger=json.loads((BASE/'logs/process-ledger.json').read_text())
    need(ledger['run_id']==RUN and all(x['state']=='STOPPED' for x in ledger['jobs'].values()),
         'Original owned components were not stopped')
    for name in ('retrieval-staging-cleanup.json','corpus-cache-cleanup.json'):
        d=json.loads((BASE/'logs'/name).read_text())
        need(d['cleaned'] and d['run_id']==RUN and d['slurm_job_id']==JOB,'Private tmpfs cleanup missing')
    need(not (BASE/'logs/cleanup-status.json').exists(),'Unexpected GPU context cleanup outcome; inspect')
    for name in ('actual-updates.jsonl','actual-attempts.jsonl'):
        need(not (BASE/'logs'/name).exists(),'Formal work already exists')
    for name in ('checkpoints','artifacts/question-probe/completed','artifacts/question-probe/completed-pairs',
                 'artifacts/question-probe/teacher-warmup-ready.json'):
        p=BASE/name
        need(not p.is_file() and not (p.exists() and any(q.is_file() for q in p.rglob('*'))),
             'Do not reset actual completed actor/head work')
    allocation=json.loads((BASE/'logs/native-allocation-admission.json').read_text())
    need(allocation['accepted'] and allocation['run_id']==RUN and allocation['slurm_job_id']==JOB and
         len(set(allocation['actual_cuda_runtime_uuids']))==4,'Original actual CUDA UUID admission differs')
    proof=json.loads((ROOT/'preparation'/('source-'+SOURCE+'-verification.json')).read_text())
    need(proof['status']=='PUBLISHED_SOURCE_FULL_SHA_PASS' and proof['commit']==SOURCE,'Original source missing')
    for name,sha in proof['files'].items():need(sha_file(ROOT/'source'/SOURCE/name)==sha,'Original source changed')
    return dict(kind='ACTUAL_NATIVE4_FROZEN_TEACHER1024_NVML_TIMEOUT_FAILURE',job=JOB,base=str(BASE),
        source_commit=SOURCE,run_id=RUN,actual_accounting=raw,prior=prior,original_claim=claim,
        teacher_cache=cache_readback(BASE/'teacher-warmup-cache'),actual_optimizer_writes=0,
        completed_training_outers=0,private_tmpfs_cleaned=True,gpu_context_query_cleanup_accepted=False,
        original_node_process_birth_readback_required=True,excluded_nodes=[NODE,'ACD1-67'],
        source_files=len(proof['files']),verified_by_uid=os.getuid(),checked_unix=time.time())


def archive():
    value=verify();directory=BASE/('failed-job-'+JOB)
    need(not directory.exists(),'Archive already attempted; inspect instead of replay')
    files={}
    for folder in ('logs','preparation','artifacts','teacher-warmup-cache'):
        for p in sorted((BASE/folder).rglob('*')):
            if p.is_file() and not p.is_symlink():
                st=p.stat();sha=sha_file(p);after=p.stat()
                need(st.st_uid==14090 and (st.st_ino,st.st_size,st.st_mtime_ns)==
                     (after.st_ino,after.st_size,after.st_mtime_ns),'Owned terminal evidence still changing')
                files[str(p.relative_to(BASE))]=dict(path=str(p),bytes=st.st_size,sha256=sha,
                                                   inode=st.st_ino,mtime_ns=st.st_mtime_ns)
    directory.mkdir(mode=0o770)
    for name,row in files.items():
        if row['bytes']<=16*2**20:
            target=directory/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(BASE/name,target)
            need(sha_file(target)==row['sha256'],'Failure copy SHA differs')
    publish(directory/'terminal-and-files.json',dict(value,files=files,large_files_retained_in_place=True))
    return value


def preserved():
    p=BASE/('failed-job-'+JOB)/'terminal-and-files.json'
    need(p.is_file() and not p.is_symlink() and p.stat().st_uid==14090,'Owner failure archive missing')
    saved=json.loads(p.read_text());now=verify()
    for key in ('kind','job','base','source_commit','run_id','actual_accounting','prior','original_claim','teacher_cache'):
        need(saved[key]==now[key],'Preserved failure binding changed')
    for row in saved['files'].values():
        q=Path(row['path']);st=q.stat()
        need((st.st_ino,st.st_size,st.st_mtime_ns)==(row['inode'],row['bytes'],row['mtime_ns']) and
             sha_file(q)==row['sha256'],'Original retained failure changed')
    return now


def reuse_cache(target,commit):
    old=preserved();source=ROOT/'source'/commit
    identity=json.loads((BASE/'teacher-warmup-cache/identity.json').read_text())
    for name,sha in dict(identity['scorer_sources'],**identity['rollout_sources']).items():
        need(sha_file(source/name)==sha,'Teacher sampling/scoring changed; existing chunks cannot be reused')
    dest=Path(target)/'teacher-warmup-cache';dest.mkdir(mode=0o770)
    (dest/'writer.lock').touch()
    for name,row in old['teacher_cache']['files'].items():
        p=BASE/'teacher-warmup-cache'/name;q=dest/name
        with p.open('rb') as r,q.open('xb') as w:
            shutil.copyfileobj(r,w,8*2**20);w.flush();os.fsync(w.fileno())
        need(q.stat().st_size==row['bytes'] and sha_file(q)==row['sha256'],'Reused actual chunk differs')
    copied=cache_readback(dest)
    need(copied==old['teacher_cache'],'Actual independently copied teacher cache differs')
    value=dict(source=str(BASE/'teacher-warmup-cache'),target=str(dest),source_commit=SOURCE,
        target_commit=commit,questions=1024,trajectories=5120,raw_bytes=copied['raw_bytes'],
        files=copied['files'],remaining_questions=4096,actual_formal_completed=0,checked_unix=time.time())
    publish(Path(target)/'preparation/teacher-cache-reuse.json',value)
    return value


def cleanup_binding(require_complete=False):
    claim=json.loads((BASE/'failure-cleanup718509/submit-claim.json').read_text())
    receipt=json.loads((BASE/'failure-cleanup718509/submission.json').read_text())
    need(receipt['returncode']==0 and receipt['commit']==claim['commit'],'Own cleanup submission absent')
    job=receipt['job'];source=ROOT/'source'/claim['commit'];entry=source/'reproduction/hpc3_qprobe/cleanup_telemetry718509.sh'
    need(claim['command'][-2:]==[str(entry),str(source)] and '--nodelist='+NODE in claim['command'],
         'Cleanup native source/node differs')
    state=subprocess.check_output(['sacct','-nP','-j',job,'-X','--format=JobID,State,ExitCode,UID,NodeList'],text=True,timeout=30).strip()
    complete=state==job+'|COMPLETED|0:0|14090|'+NODE
    if complete:
        d=json.loads((BASE/'failure-cleanup718509/complete.json').read_text())
        need(d['accepted'] and d['node']==NODE and d['run_id']==RUN and d['failed_job']==JOB and
             d['cleanup_job']==job and d['recorded_old_births_absent'] and d['original_private_tmpfs_absent'],
             'Actual native old process/private cache readback missing')
    else:
        from reproduction.common8.native_allocation import fields
        d=fields(subprocess.check_output(['scontrol','show','job','-d',job],text=True,timeout=30))
        need(not require_complete and d['JobState'] in ('PENDING','RUNNING') and
             d['UserId']=='jxu957(14090)' and d['Account']=='jxu957' and d['QOS']=='formal-user' and
             d['ReqNodeList']==NODE and d['JobName']=='qprobe-cleanup718509' and d['Command']==str(entry),
             'Native old-node cleanup is neither accepted nor the own pending allocation')
    return dict(private_cleanup_job=job,private_cleanup_job_name='qprobe-cleanup718509',
                private_cleanup_complete=complete,cleanup_required_before_goal_complete=True)


def submit_cleanup(commit):
    preserved();source=ROOT/'source'/commit
    proof=json.loads((ROOT/'preparation'/('source-'+commit+'-verification.json')).read_text())
    need(proof['commit']==commit and proof['status']=='PUBLISHED_SOURCE_FULL_SHA_PASS','Cleanup repair not published')
    for name,sha in proof['files'].items():need(sha_file(source/name)==sha,'Cleanup source changed')
    root=BASE/'failure-cleanup718509';need(not root.exists(),'Cleanup already attempted; inspect actual claim')
    root.mkdir(mode=0o770)
    command=['sbatch','--parsable','--account=jxu957','--qos=formal-user','--partition=emergency_acd',
        '--nodelist='+NODE,'--nodes=1','--ntasks=1','--gres=gpu:1','--cpus-per-task=1','--mem=1G','--time=00:10:00',
        '--job-name=qprobe-cleanup718509','--export=NONE','--chdir='+str(BASE),
        '--output='+str(root/'native-%j.log'),'--error='+str(root/'native-%j.log'),
        str(source/'reproduction/hpc3_qprobe/cleanup_telemetry718509.sh'),str(source)]
    publish(root/'submit-claim.json',dict(commit=commit,command=command,failed_job=JOB,epoch=time.time()))
    r=subprocess.run(command,capture_output=True,text=True,timeout=30)
    value=dict(commit=commit,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,epoch=time.time())
    if not r.returncode:
        value['job']=r.stdout.strip().split(';')[0];need(value['job'].isdecimal(),'Invalid native cleanup job')
    publish(root/'submission.json',value);need(r.returncode==0,'Cleanup submission failed; inspect saved evidence')
    return value


def native_cleanup():
    need(os.getuid()==14090 and socket.gethostname().split('.')[0]==NODE,'Actual original node/owner required')
    job=os.environ['SLURM_JOB_ID'];claim=json.loads((BASE/'failure-cleanup718509/submit-claim.json').read_text())
    from reproduction.common8.native_allocation import fields
    d=fields(subprocess.check_output(['scontrol','show','job','-d',job],text=True,timeout=30))
    source=ROOT/'source'/claim['commit']
    need(d['JobState']=='RUNNING' and d['UserId']=='jxu957(14090)' and d['NodeList']==NODE and
         d['JobName']=='qprobe-cleanup718509' and d['Command']==str(source/'reproduction/hpc3_qprobe/cleanup_telemetry718509.sh'),
         'Actual native cleanup birth differs')
    preserved()
    ledger=json.loads((BASE/'logs/process-ledger.json').read_text());sessions=set()
    from reproduction.a800.owned_processes import identity
    births=0
    for entry in ledger['jobs'].values():
        sessions.add(entry['leader']['sid'])
        for old in entry['ledger'].values():
            births+=1
            try:now=identity(old['pid'])
            except (FileNotFoundError,ProcessLookupError):continue
            need(now['state']=='Z' or any(now[k]!=old[k] for k in ('uid','start_ticks','sid')),
                 'Old owned process birth still active; refuse new signals')
    for p in Path('/proc').iterdir():
        if not p.name.isdecimal():continue
        try:now=identity(int(p.name))
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
        need(now['state']=='Z' or now['uid']!=14090 or now['sid'] not in sessions,'Old owned session remains active')
    stage=json.loads((BASE/'logs/retrieval-staging-cleanup.json').read_text())
    need(not Path(stage['root']).exists(),'Original private retrieval tmpfs remains')
    value=dict(accepted=True,node=NODE,run_id=RUN,failed_job=JOB,cleanup_job=job,verified_uid=os.getuid(),
        recorded_old_births_absent=True,verified_birth_count=births,original_private_tmpfs_absent=True,
        original_gpu_context_query_accepted=False,scope='Native original node /proc PID+birth+session readback; '
        'no CUDA health claim, no model load, no signals, original logs unchanged',checked_unix=time.time())
    publish(BASE/'failure-cleanup718509/complete.json',value);return value


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('archive','submit-cleanup','native-cleanup'))
    p.add_argument('--commit');a=p.parse_args()
    result={'archive':archive,'native-cleanup':native_cleanup}.get(a.mode)
    print(json.dumps(result() if result else submit_cleanup(a.commit)),flush=True)
