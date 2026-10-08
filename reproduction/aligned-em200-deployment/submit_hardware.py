"""One durable new-source hardware submission. Unknown outcomes are never retried."""
from pathlib import Path
import datetime, fcntl, hashlib, json, os, re, subprocess, sys, uuid
ROOT=Path('/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-baseline-aligned-em200-20261009')
SOURCE='080ebbc46b89d61567cd33e1a5e2f1b40afd8b84'
# Replaced with reviewed fixed bytes before this launcher is published.
HARDWARE_FILES={'hardware.sbatch': 'afc53d15545dea79d3c32fd5ec58216ff104474b634173c8d5346af7029f0101', 'hardware_worker.sh': '15e170b11c7e36a16600d501388d326bad1f93c01a112a8828583e8a6a1e0886', 'hardware_inner.sh': 'd89e90ba172fb3ade4ad670de39abe656fb55addb2274c77fc454fc25cfb6ac1', 'verify_live_retrieval.py': '66dbe3d7cf77fc60a86207ec6d17210848a5f0e4c7e769109b1a895a80e06e2c', 'bind_retrieval_proof.py': '3ae4f0095b775716b0f2d3180b513906a7be51837d1f2786b3ec72447733704d', 'capture_retrieval_golden.py': '67c888845d5c7708a239d508fcd7c697f27b8e4c4a495ab89027cd5b17b8b9fa', 'numerics_gpu_fixture.py': 'adb451471ccf7d75b74ab78a30a01d074e460c69dd1cbcd4ebad7eced86c4028', 'reference/retrieval_server.py': 'e85ab42b9032d66a35239a36e8cd73c6f3e3efde929541150fe7c5bfdb4936d9', 'reference/retrieval-provenance.json': '20e8c73053014e729c92fe242e47eaf36f5e13f77658034a1d785bec434c8c7c'}
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
    require(sys.prefix==str(ROOT/'env') and os.uname().nodename=='h100-5' and ROOT.resolve()==ROOT,'actual new-source environment required')
    require(subprocess.check_output(['git','-C',str(ROOT/'repo'),'rev-parse','HEAD'],text=True).strip()==SOURCE,'fixed source differs')
    require(not subprocess.check_output(['git','-C',str(ROOT/'repo'),'status','--porcelain'],text=True).strip(),'dirty production source')
    require(len(HARDWARE_FILES)>=7,'unbound hardware launcher forbidden')
    for name in ('runs','logs','tools','tools/deployment','repo','env'):
        path=ROOT/name
        require(path.is_dir() and path.resolve()==path,'canonical own admission directories required')
    directory=ROOT/'tools/deployment'
    for n,h in HARDWARE_FILES.items():
        path=directory/n
        require(path.resolve()==path and path.is_file() and sha(path)==h,'reviewed hardware bytes differ: '+n)
    sys.path.insert(0,str(ROOT/'tools'))
    import resource_watch
    require(Path(resource_watch.__file__).resolve()==ROOT/'tools/resource_watch.py','foreign watcher module')
    anchor=resource_watch.load_runtime_anchor(ROOT/'runs'/'hardware-submit-anchor',verify_current=True)
    require(json.loads((ROOT/'logs/launch-approval.json').read_text())['status']=='HARDWARE_ONLY_RUNTIME_ANCHOR_NOT_TRAINING_AUTHORIZATION','hardware-only runtime scope required')
    head=ROOT/'scheduler-head';require(head.resolve()==head,'canonical scheduler directory required')
    lock=head/'submission-global.lock';require(lock.resolve()==lock and not lock.is_symlink(),'canonical submission lock required')
    with lock.open('a+') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        intent=head/'hardware-submit-intent.json';receipt=head/'hardware-submit-receipt.json'
        response=head/'hardware-sbatch-response.json'
        require(all(not p.exists() and not p.is_symlink() for p in (intent,receipt,response)),'existing hardware submission or unknown outcome: inspect original job; never retry')
        queue=subprocess.check_output(['squeue','-h','-o','%i|%j|%k|%T'],text=True)
        require(not any('BAPO_ALIGNED_EM200:' in line for line in queue.splitlines()),'new-source BAPO allocation already exists')
        token=uuid.uuid4().hex;comment='BAPO_ALIGNED_EM200:hardware:'+SOURCE+':'+token
        value=dict(source=SOURCE,root=str(ROOT),token=token,comment=comment,runtime_anchor_sha256=anchor['sha256'],files=HARDWARE_FILES,time=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='HARDWARE_ONLY_NO_TRAINING_SUBMISSION')
        write(intent,value)  # Must be durable before sbatch can create an allocation.
        command=['sbatch','--parsable','--no-requeue','--comment='+comment,str(directory/'hardware.sbatch'),token]
        result=subprocess.run(command,capture_output=True,text=True,timeout=45)
        write(response,dict(command=command,exit=result.returncode,stdout=result.stdout,stderr=result.stderr,scope='RAW_SUBMISSION_RESPONSE_NOT_HARDWARE_ACCEPTANCE'))
        require(result.returncode==0 and re.fullmatch(r'[1-9][0-9]*(;[A-Za-z0-9_.-]+)?\n?',result.stdout) is not None,'unknown or failed submit outcome retained: do not retry')
        job=result.stdout.strip().split(';')[0]
        write(receipt,dict(value,job=job,run=str(ROOT/'runs'/('hardware-aligned-em200-'+token+'-'+job)),scope='ACTUAL_SINGLE_HARDWARE_JOB_SUBMITTED_NOT_GPU_ACCEPTANCE_OR_TRAINING'))
        print(json.dumps(dict(job=job,token=token,run=str(ROOT/'runs'/('hardware-aligned-em200-'+token+'-'+job)),scope='ACTUAL_HARDWARE_SUBMISSION_NOT_TRAINING')))
if __name__=='__main__':main()
