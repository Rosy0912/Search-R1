"""Once-only capped CPU preservation/copy/submission for actual failed718509.

The independent process outlives the SSH observation window. Durable identity,
stage and exit state make an unknown transport outcome inspectable, never a
reason to repeat a submission. This controller neither loads a model nor asks
for training GPUs itself; the existing strict recovery entry submits native4.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from reproduction.hpc3_qprobe.control import publish
from reproduction.hpc3_qprobe.telemetry_failure718509 import ROOT,BASE,JOB,need
from search_g1.qprobe_features import sha_file

DIRECTORY=ROOT/'preparation/recovery718509-controller'


def source(commit):
    need(os.getuid()==14090,'Actual jxu957 controller owner required')
    root=ROOT/'source'/commit
    proof=json.loads((ROOT/'preparation'/('source-'+commit+'-verification.json')).read_text())
    need(proof['commit']==commit and proof['published_remote_sha']==commit and
         proof['status']=='PUBLISHED_SOURCE_FULL_SHA_PASS','Exact published repair required')
    for name,sha in proof['files'].items():need(sha_file(root/name)==sha,'Actual deployed repair changed')
    return root


def launch(commit):
    root=source(commit)
    from reproduction.hpc3_qprobe.telemetry_failure718509 import validate_terminal
    validate_terminal(subprocess.check_output(['sacct','-nP','-j',JOB,'-X',
        '--format=JobID,State,ExitCode,User,UID,Account,QOS,Partition,Start,End,NodeList'],text=True,timeout=30))
    need(not DIRECTORY.exists(),'Controller already attempted; inspect actual PID/stage/claim instead of repeating')
    DIRECTORY.mkdir(mode=0o770)
    command=[str(BASE/'env/bin/python'),'-m','reproduction.hpc3_qprobe.recovery718509_controller',
             'run','--commit',commit]
    publish(DIRECTORY/'launch-claim.json',dict(commit=commit,command=command,failed_job=JOB,epoch=time.time()))
    env=dict(os.environ,QPROBE_SOURCE=str(root),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',
             MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    with (DIRECTORY/'output.log').open('xb') as f:
        p=subprocess.Popen(command,env=env,cwd=root,stdin=subprocess.DEVNULL,stdout=f,
                           stderr=subprocess.STDOUT,start_new_session=True)
    from reproduction.a800.owned_processes import identity
    born=identity(p.pid);need(born['uid']==14090 and born['sid']==p.pid,'Controller birth differs')
    value=dict(commit=commit,failed_job=JOB,controller_identity=born,epoch=time.time(),
               model_loading=False,training=False,thread_limit=1,output=str(DIRECTORY/'output.log'))
    publish(DIRECTORY/'launch-receipt.json',value);return value


def run(commit):
    root=source(commit)
    claim=json.loads((DIRECTORY/'launch-claim.json').read_text())
    need(claim['commit']==commit and claim['failed_job']==JOB,'Native CPU intent changed')
    need(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and all(os.environ.get(k)=='1' for k in (
        'OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')),'CPU worker cap differs')
    from reproduction.a800.owned_processes import identity
    from reproduction.a800.owned_processes import publish as atomic
    born=identity(os.getpid());need(born['sid']==os.getpid() and born['uid']==14090,'CPU worker session differs')
    def state(stage,**values):
        atomic(DIRECTORY/'state.json',dict(commit=commit,failed_job=JOB,controller_identity=born,
            stage=stage,epoch=time.time(),model_loading=False,training=False,**values))
    try:
        from reproduction.hpc3_qprobe.telemetry_failure718509 import archive,submit_cleanup
        state('PRESERVING_ACTUAL_FAILED_TEACHER1024_FULL_SHA')
        failed=archive();state('SUBMITTING_ONCE_OLD_NODE_PROCESS_READBACK',teacher_questions=1024)
        cleanup=submit_cleanup(commit)
        from reproduction.hpc3_qprobe import recover_pretraining as recovery
        recovery.select_failed_job(JOB)
        state('PREPARING_INDEPENDENT_VERIFIED_TEACHER_CACHE_COPY',cleanup_job=cleanup['job'],
              target=str(recovery.TARGET),planned_raw_bytes=failed['teacher_cache']['raw_bytes'])
        prepared=recovery.fresh(commit)
        state('SUBMITTING_ONCE_NATIVE4_RECOVERY',target=prepared['target'],cleanup_job=cleanup['job'])
        submitted=recovery.submit(commit)
        value=dict(commit=commit,failed_job=JOB,target=str(recovery.TARGET),cleanup_job=cleanup['job'],
            training_job=submitted['job'],teacher_questions_reused=1024,teacher_trajectories_reused=5120,
            formal_completed_outers=0,training_progress_accepted=False,finished_unix=time.time())
        publish(DIRECTORY/'complete.json',value);state('COMPLETED_ONCE_NATIVE4_SUBMISSION',exit_code=0,**value)
        print(json.dumps(value),flush=True);return value
    except Exception as error:
        state('FAILED',exit_code=1,error=repr(error));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('launch','run'));p.add_argument('--commit',required=True)
    a=p.parse_args();result=launch(a.commit) if a.mode=='launch' else run(a.commit)
    if a.mode=='launch':print(json.dumps(result),flush=True)
