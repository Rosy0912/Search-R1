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
RETRY_SHADOW=FAILED/'failure-cleanup717827-recovery717934'
RETRY_STEM='cleanup717827-recovery717934'


def check_failed_readers(base):
    """The stager and corpus creator have distinct, retained claim schemas."""
    from reproduction.a800.owned_processes import identity
    stage=json.loads((base/'logs/retrieval-staging-claim.json').read_text())
    corpus=json.loads((base/'logs/corpus-cache-claim.json').read_text())
    assert (corpus['run_id'],corpus['slurm_job_id'],corpus['uid']) == (stage['run_id'],stage['slurm_job_id'],stage['uid'])
    assert_quiescent(base,stage)
    creator=corpus['creator']
    assert creator['uid']==corpus['uid']
    try: actual=identity(creator['pid'])
    except (FileNotFoundError,ProcessLookupError):actual=None
    assert actual is None or actual['state']=='Z' or actual['start_ticks']!=creator['start_ticks'], 'Cache creator remains live'


def bind_native_retry_runner(job_id):
    if os.getuid()!=14090:raise ValueError('Actual owner required for native retry binding')
    claim=json.loads((FAILED/'preparation'/(RETRY_STEM+'-submit-claim.json')).read_text())
    assert claim['retry_of']=='717934'
    detail=subprocess.check_output(['scontrol','show','job','-o',job_id],text=True)
    actual_job=fields(detail)
    root=FAILED.parent/'qprobe-teacher-warmup200-20261014'
    assert actual_job['JobId']==job_id and actual_job['JobState']=='RUNNING' and actual_job['UserId']=='jxu957(14090)' and actual_job['Account']=='jxu957' and actual_job['QOS']=='formal-user' and actual_job['NodeList']=='ACD1-67'
    assert actual_job['JobName']=='qprobe-cleanup717934' and actual_job['WorkDir']==str(FAILED)
    assert actual_job['Command']==str(root/'source'/claim['commit']/'reproduction/hpc3_qprobe/cleanup_failed_717827.sh')
    receipt_path=FAILED/'preparation'/(RETRY_STEM+'-submission.json')
    # emergency_acd may start the job before sbatch returns or before its
    # receipt becomes visible. The durable pre-submit claim and actual
    # owner-bound Slurm birth are authoritative for this native runner.
    if receipt_path.exists():
        receipt=json.loads(receipt_path.read_text())
        assert receipt['job']==job_id and receipt['commit']==claim['commit']
        assert receipt.get('returncode')==0 or receipt.get('status')=='ACTUAL_SLURM_RECONCILED'
    publish(RETRY_SHADOW/'runner-job-binding.json',dict(kind='NATIVE_RETRY_RUNNER_JOB_BINDING',
        job=job_id,commit=claim['commit'],uid=os.getuid(),actual_job_detail=detail,
        claim_sha256=hashlib.sha256((FAILED/'preparation'/(RETRY_STEM+'-submit-claim.json')).read_bytes()).hexdigest(),
        submit_receipt_observed=receipt_path.exists(),checked_unix=time.time()))


def main(retry_of=None):
    assert os.getuid()==14090 and socket.gethostname().split('.')[0]=='ACD1-67'
    assert os.environ.get('SLURM_JOB_ID') not in (None,'717827','717934')
    shadow=SHADOW
    if retry_of is not None:
        assert retry_of=='717934'
        from reproduction.hpc3_qprobe.observed_startup_failures import owner_verified
        owner_verified('717934')
        bind_native_retry_runner(os.environ['SLURM_JOB_ID'])
        shadow=RETRY_SHADOW
    proof=json.loads((FAILED/'failed-job-717827/terminal-and-files.json').read_text())
    old=fields(proof['terminal']['stdout'])
    assert old['JobId']=='717827' and old['JobState']=='FAILED' and old['UserId']=='jxu957(14090)' and old['NodeList']=='ACD1-67'
    actual=subprocess.check_output(['sacct','-nP','-j','717827','-X','--format=JobID,State,ExitCode,UID,NodeList'],text=True).strip()
    assert actual=='717827|FAILED|1:0|14090|ACD1-67'
    assert shadow.is_dir() and not shadow.is_symlink() and not (shadow/'complete.json').exists()
    run=json.loads((FAILED/'logs/run-manifest.json').read_text())
    assert run['completed_updates']==0 and run['training_world_size']==4 and run['source_commit']=='a75e0c944e2bbae284e52ce9b9f8bf909ab89564'
    from reproduction.common8.temporary_cache import cleanup as clean_corpus
    from reproduction.common8.stage_retrieval import cleanup as clean_stage
    from reproduction.a800.owned_processes import OwnedProcesses
    check_failed_readers(shadow)
    owned=OwnedProcesses(shadow/'logs/process-ledger.json',run['run_id'])
    # No live process is eligible after terminal Slurm teardown. Fail rather
    # than signal any recovered PID when the kernel has no pidfd support.
    assert all(not owned.observe(n) for n in owned.jobs)
    owned.stop_all()
    corpus=clean_corpus(shadow);stage=clean_stage(shadow)
    value=dict(cleaned=True,run_id=run['run_id'],failed_job='717827',cleanup_job=os.environ['SLURM_JOB_ID'],node=socket.gethostname(),actual_terminal=actual,corpus=corpus,stage=stage,original_logs_unchanged=True,checked_unix=time.time())
    value.update(retry_of=retry_of,shadow=str(shadow))
    publish(shadow/'complete.json',value);print(json.dumps(value),flush=True)


def submit_retry(commit):
    if os.getuid()!=14090:raise ValueError('Actual owner required for failed cleanup retry')
    from reproduction.hpc3_qprobe.observed_startup_failures import owner_verified
    failure=owner_verified('717934')
    root=FAILED.parent/'qprobe-teacher-warmup200-20261014';source=root/'source'/commit
    verification=json.loads((root/'preparation'/('source-'+commit+'-verification.json')).read_text())
    assert verification['status']=='PUBLISHED_SOURCE_FULL_SHA_PASS' and verification['commit']==commit
    for n,h in verification['files'].items():assert hashlib.sha256((source/n).read_bytes()).hexdigest()==h
    proof=json.loads((FAILED/'failed-job-717827/terminal-and-files.json').read_text())
    assert subprocess.check_output(['sacct','-nP','-j','717827','-X','--format=JobID,State,ExitCode,UID,NodeList'],text=True,timeout=30).strip()=='717827|FAILED|1:0|14090|ACD1-67'
    for n,h in proof['files'].items():assert hashlib.sha256((FAILED/n).read_bytes()).hexdigest()==h['sha256']
    for name in ('corpus-cache-claim.json','retrieval-staging-claim.json'):
        old=json.loads((SHADOW/'logs'/name).read_text())
        assert old['phase'] not in ('CLEANING','CLEANED'),'Prior cleanup changed cache state; inspect instead'
    intent=FAILED/'preparation'/(RETRY_STEM+'-submit-claim.json');receipt=FAILED/'preparation'/(RETRY_STEM+'-submission.json')
    assert not RETRY_SHADOW.exists() and not intent.exists() and not receipt.exists(),'Inspect existing retry instead of replaying'
    queue=subprocess.check_output(['squeue','-h','-u','jxu957','-o','%i|%j|%T'],text=True,timeout=30)
    assert not any('qprobe-cleanup' in line for line in queue.splitlines()),'Another cleanup is live or pending'
    command=['sbatch','--parsable','--account=jxu957','--qos=formal-user','--partition=emergency_acd',
        '--nodelist=ACD1-67','--nodes=1','--ntasks=1','--cpus-per-task=1','--mem=1G','--gres=gpu:1',
        '--time=00:10:00','--job-name=qprobe-cleanup717934','--export=NONE','--chdir='+str(FAILED),
        '--output='+str(FAILED/'logs/cleanup717934-retry-%j.log'),str(source/'reproduction/hpc3_qprobe/cleanup_failed_717827.sh'),
        str(source),'--retry-of','717934']
    test=subprocess.run(command[:1]+['--test-only']+command[1:],capture_output=True,text=True,timeout=30)
    publish(FAILED/'preparation'/(RETRY_STEM+'-test-only.json'),dict(command=command,exit=test.returncode,stdout=test.stdout,stderr=test.stderr,epoch=time.time()))
    assert test.returncode==0,test.stderr
    publish(intent,dict(commit=commit,command=command,retry_of='717934',failure=failure,epoch=time.time()))
    (RETRY_SHADOW/'logs').mkdir(parents=True)
    for n in ('run-manifest.json','process-ledger.json','corpus-cache-claim.json','retrieval-staging-claim.json'):
        original=SHADOW/'logs'/n;target=RETRY_SHADOW/'logs'/n
        assert not original.is_symlink();shutil.copyfile(original,target)
        assert hashlib.sha256(original.read_bytes()).digest()==hashlib.sha256(target.read_bytes()).digest()
    result=subprocess.run(command,capture_output=True,text=True,timeout=30)
    value=dict(commit=commit,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,retry_of='717934',epoch=time.time())
    if result.returncode==0:
        value['job']=result.stdout.strip().split(';')[0];assert value['job'].isdecimal()
    publish(receipt,value);assert result.returncode==0,result.stderr
    return value


def reconcile_retry(job):
    """Recover an unknown sbatch result from this native runner, never resubmit."""
    if os.getuid()!=14090:raise ValueError('Actual owner required for submission reconciliation')
    from reproduction.hpc3_qprobe.observed_startup_failures import owner_verified
    owner_verified('717934')
    receipt=FAILED/'preparation'/(RETRY_STEM+'-submission.json')
    assert not receipt.exists(),'Existing submission result must be inspected, not replaced'
    path=FAILED/'preparation'/(RETRY_STEM+'-submit-claim.json');claim=json.loads(path.read_text())
    runner=RETRY_SHADOW/'runner-job-binding.json';assert not runner.is_symlink() and runner.stat().st_uid==14090
    value=json.loads(runner.read_text());actual=fields(value['actual_job_detail'])
    root=FAILED.parent/'qprobe-teacher-warmup200-20261014'
    assert value['kind']=='NATIVE_RETRY_RUNNER_JOB_BINDING' and value['job']==job and value['commit']==claim['commit'] and value['uid']==14090
    assert value['claim_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual['JobId']==job and actual['UserId']=='jxu957(14090)' and actual['Account']=='jxu957' and actual['QOS']=='formal-user'
    assert actual['JobName']=='qprobe-cleanup717934' and actual['NodeList']=='ACD1-67' and actual['WorkDir']==str(FAILED)
    assert actual['Command']==str(root/'source'/claim['commit']/'reproduction/hpc3_qprobe/cleanup_failed_717827.sh')
    fresh=subprocess.check_output(['sacct','-nP','-j',job,'-X','--format=JobID,State,User,UID,Account,QOS,Start,NodeList'],text=True,timeout=30).strip().split('|')
    assert len(fresh)==8 and fresh[0]==job and fresh[1] in ('RUNNING','COMPLETING','COMPLETED','FAILED','TIMEOUT','CANCELLED')
    assert fresh[2:]==['jxu957','14090','jxu957','formal-user',actual['StartTime'],'ACD1-67']
    result=dict(status='ACTUAL_SLURM_RECONCILED',returncode=None,commit=claim['commit'],job=job,
        retry_of='717934',submission_repeated=False,native_runner_sha256=hashlib.sha256(runner.read_bytes()).hexdigest(),
        actual_accounting=fresh,epoch=time.time())
    publish(receipt,result);return result


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
        '--nodelist=ACD1-67','--nodes=1','--ntasks=1','--cpus-per-task=1','--mem=1G','--gres=gpu:1',
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
    p=argparse.ArgumentParser();p.add_argument('--submit-commit');p.add_argument('--retry-of',choices=('717934',));p.add_argument('--reconcile-retry-job');a=p.parse_args()
    if a.reconcile_retry_job:print(json.dumps(reconcile_retry(a.reconcile_retry_job)),flush=True)
    elif a.submit_commit:print(json.dumps((submit_retry if a.retry_of else submit)(a.submit_commit)),flush=True)
    else:main(a.retry_of)
