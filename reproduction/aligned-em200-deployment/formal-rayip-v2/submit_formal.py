"""One durable new-source formal submission. Candidate is locked until actual acceptance. Unknown outcomes are never retried."""
from pathlib import Path
import datetime, fcntl, hashlib, json, os, re, subprocess, sys, uuid
ROOT=Path('/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-baseline-aligned-em200-20261009')
SOURCE='080ebbc46b89d61567cd33e1a5e2f1b40afd8b84'
from formal_admission import validate_admission, FORMAL_FILES
def require(value,message):
    if not value:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    require(path.parent.resolve()==path.parent and not path.is_symlink(),'canonical own scheduler destination required')
    with path.open('x') as f:json.dump(value,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    fd=os.open(str(path.parent),os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
def main():
    validate_admission()  # unbound candidate exits before any scheduler query/write
    directory=ROOT/'tools/deployment/formal-rayip-v2'
    head=ROOT/'scheduler-head/formal-rayip-v2';require(head.resolve()==head,'canonical scheduler directory required')
    lock=ROOT/'scheduler-head/submission-global.lock';require(lock.resolve()==lock and not lock.is_symlink(),'canonical submission lock required')
    with lock.open('a+') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        intent=head/'formal-submit-intent.json';receipt=head/'formal-submit-receipt.json'
        response=head/'formal-sbatch-response.json'
        require(all(not p.exists() and not p.is_symlink() for p in (intent,receipt,response)),'existing formal submission or unknown outcome: inspect original job; never retry')
        queue=subprocess.check_output(['squeue','-h','-o','%i|%j|%k|%T'],text=True)
        require(not any('BAPO_ALIGNED_EM200:' in line for line in queue.splitlines()),'new-source BAPO allocation already exists')
        token=uuid.uuid4().hex;comment='BAPO_ALIGNED_EM200:formal:'+SOURCE+':'+token
        value=dict(deployment_directory=str(ROOT/'tools/deployment/formal-rayip-v2'),submission_namespace='formal-rayip-v2',previous_failed_job='3348',source=SOURCE,root=str(ROOT),token=token,comment=comment,hardware_review_sha256=__import__('formal_admission').HARDWARE_REVIEW_SHA256,formal_authorization_sha256=__import__('formal_admission').FORMAL_AUTHORIZATION_SHA256,files=FORMAL_FILES,time=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='FORMAL_FRESH200_SINGLE_SUBMISSION_NOT_TRAINING_COMPLETION')
        write(intent,value)  # Must be durable before sbatch can create an allocation.
        command=['sbatch','--parsable','--no-requeue','--comment='+comment,str(directory/'formal.sbatch'),token]
        result=subprocess.run(command,capture_output=True,text=True,timeout=45)
        write(response,dict(command=command,exit=result.returncode,stdout=result.stdout,stderr=result.stderr,scope='RAW_SUBMISSION_RESPONSE_NOT_HARDWARE_ACCEPTANCE'))
        require(result.returncode==0 and re.fullmatch(r'[1-9][0-9]*(;[A-Za-z0-9_.-]+)?\n?',result.stdout) is not None,'unknown or failed submit outcome retained: do not retry')
        job=result.stdout.strip().split(';')[0]
        write(receipt,dict(value,job=job,run=str(ROOT/'runs'/('formal-aligned-em200-'+token+'-'+job)),scope='ACTUAL_SINGLE_FORMAL_JOB_SUBMITTED_NOT_TRAINING_COMPLETION'))
        print(json.dumps(dict(job=job,token=token,run=str(ROOT/'runs'/('formal-aligned-em200-'+token+'-'+job)),scope='ACTUAL_FORMAL_SUBMISSION_NOT_TRAINING_COMPLETION')))
if __name__=='__main__':main()
