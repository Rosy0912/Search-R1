#!/usr/bin/env python3
"""Owned Linux tinyfile/proc + immutable v6 CPU44 proof; no live relay or GPU."""
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT=Path('/dataset/shaoshitong/search-r1-rl/topd')
HERE=Path(__file__).resolve().parent
CANDIDATE=HERE.parent/'topd-official-resume-transfer'
ARCHIVE_SHA='f6b25cf3095917001abf468b1fa0a0869f50c3420b624e1d36840e11571e216a'
TRANSFER_SHA='11f703ec21139b437eb5044703041077af310e196acc5b2b9bdcd634f0f95d0a'
METADATA_SHA='e546ba207f643dbe1a098323e28d45ff07942a6390bc73b8391c0acfc20d84d1'


def require(ok,message):
    if not ok:raise RuntimeError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bytes_sha(raw):return hashlib.sha256(raw).hexdigest()


def write_record(directory,name,value):
    path=directory/name;fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'w') as stream:json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    fd=os.open(directory,os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return dict(name=name,sha256=sha(path))


def identity(pid):
    p=Path('/proc')/str(pid);raw=(p/'stat').read_text();values=raw[raw.rfind(')')+2:].split()
    return dict(pid=pid,ppid=int(values[1]),starttime=int(values[19]),state=values[0],cgroups=[line.split(':',2)[-1] for line in (p/'cgroup').read_text().splitlines()])


def preflight():
    require(HERE==ROOT/'tools/topd-official-resume-transfer-native' and Path(os.environ.get('AGENT_ROOT','/invalid')).resolve()==ROOT,'Foreign ownROOT/tool directory')
    require(socket.gethostname().split('.')[0]=='h100-11' and os.environ.get('SLURM_JOB_NODELIST')=='h100-11','Foreign node')
    job=os.environ.get('SLURM_JOB_ID');require(job and job.isdigit(),'Actual own Slurm job required')
    require(int(os.environ.get('SLURM_CPUS_PER_TASK','0'))>=8 and int(os.environ.get('SLURM_MEM_PER_NODE','0'))>=16384,'Actual8CPU/16GiB allocation required')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not os.environ.get('SLURM_JOB_GPUS') and not os.environ.get('SLURM_STEP_GPUS'),'No GPU allocation/visibility permitted')
    require(os.environ.get('PYTHONDONTWRITEBYTECODE')=='1','No runtime tool/cache writes permitted')
    require(Path(sys.executable).absolute()==ROOT/'env/bin/python','Wrong fixed native interpreter')
    require(sys.version_info[:2]==(3,10),'Fixed Python3.10 required')
    parent=identity(os.getpid());require(any('job_'+job in p.split('/') for p in parent['cgroups']),'Outside own job scope')
    package=json.loads((HERE/'files.json').read_text())
    require(package['scope']=='OWNED_CPU_TRANSPORT_NATIVE_SUPPLEMENT_NO_ALGORITHM' and set(package['files'])=={'probe.py','test_probe.py','native-cpu.sbatch','README.md','frozen-candidate-v6.tar.gz'},'Wrong native supplement scope/files')
    for name,digest in package['files'].items():require(sha(HERE/name)==digest,'Supplement bytes changed: '+name)
    require(sha(HERE/'frozen-candidate-v6.tar.gz')==ARCHIVE_SHA,'Reviewed v6 archive changed')
    manifest=json.loads((CANDIDATE/'files.json').read_text());require(sha(CANDIDATE/'files.json')=='f88ca6578e135bb684ced13c5c44b2dea3e265f34041693d9d8a88e3fa0047af','Frozen v6 manifest differs')
    require(manifest['cpu_fixtures']==44 and manifest['official_metadata_sha256']==METADATA_SHA,'Frozen CPU44/metadata differs')
    for name,digest in manifest['files'].items():require(sha(CANDIDATE/name)==digest,'Frozen v6 extracted bytes differ: '+name)
    require(sha(CANDIDATE/'transfer.py')==TRANSFER_SHA and sha(CANDIDATE/'official-metadata.json')==METADATA_SHA,'Frozen relay/metadata bytes differ')
    import torch
    require(torch.__version__=='2.4.0+cu121' and not torch.cuda.is_initialized(),'FixedTorch2.4/noCUDA required')
    return job,parent,package


CHILD_CODE="""import json,os,sys
from pathlib import Path
legacy,ready=map(Path,sys.argv[1:3])
with legacy.open('rb') as owned:
 raw=(Path('/proc')/str(os.getpid())/'stat').read_text();v=raw[raw.rfind(')')+2:].split()
 record=dict(pid=os.getpid(),ppid=int(v[1]),starttime=int(v[19]),legacy_fd=owned.fileno(),legacy_path=str(legacy))
 fd=os.open(ready,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,'w') as f:json.dump(record,f);f.flush();os.fsync(f.fileno())
 assert sys.stdin.buffer.read(1)==b'X'
"""


def tiny_probe(directory,execute,hostname='h100-11',job=None):
    require(directory.is_dir() and directory.resolve().is_relative_to(ROOT/'logs'),'Tiny fixture must stay inside ownROOT/logs')
    require(not directory.is_symlink(),'Fixture directory symlink refused')
    name='model-00001-of-00017.safetensors';data=b'TINY_TRANSPORT_FIXTURE_NO_MODEL_WEIGHTS\n'*257;size=len(data);prefix_size=137
    legacy=directory/(name+'.transfer');legacy.write_bytes(data[:prefix_size]);legacy.chmod(0o600)
    before=dict(size=legacy.stat().st_size,inode=legacy.stat().st_ino,sha256=sha(legacy))
    expected=bytes_sha(data);prefix_sha=bytes_sha(data[:prefix_size]);ready=directory/'child-open.json';log=directory/'child.log'
    result=dict(scope='OWNED_TINYFILE_PROC_NOT_MODEL_TRANSFER_OR_TRAINING',official_model_weights=False,fixture_size=size,initial_prefix=before,
        candidate_archive_sha256=ARCHIVE_SHA,transfer_sha256=TRANSFER_SHA,child_source_sha256=bytes_sha(CHILD_CODE.encode()),receipts=[])
    child=None;actual=None;cleanup_error=None
    with log.open('x') as output:
        log.chmod(0o600)
        try:
            child=subprocess.Popen([sys.executable,'-c',CHILD_CODE,str(legacy),str(ready)],stdin=subprocess.PIPE,stdout=output,stderr=output,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
            actual=identity(child.pid)
            deadline=time.monotonic()+20
            while not ready.exists() and child.poll() is None and time.monotonic()<deadline:time.sleep(.05)
            require(ready.exists(),'Owned child did not open fixture')
            raw=json.loads(ready.read_text());actual=identity(child.pid)
            require(raw['pid']==child.pid and raw['ppid']==actual['ppid']==os.getpid() and raw['starttime']==actual['starttime'],'Actual owned child PID/PPID/starttime differs')
            if job:require(any('job_'+job in p.split('/') for p in actual['cgroups']),'Child outside own job scope')
            fd_target=(Path('/proc')/str(child.pid)/'fd'/str(raw['legacy_fd'])).resolve()
            require(fd_target==legacy.resolve(),'Actual child FD did not hold legacy fixture')
            result['actual_child_open']=dict(actual,legacy_fd=raw['legacy_fd'],fd_target=str(fd_target))
            result['receipts'].append(write_record(directory,'child-open-identity.json',result['actual_child_open']))
            try:
                execute(directory,name,size,expected,'adopt',prefix_size,0,bytes_sha(b''),io.BytesIO(b''),hostname=hostname,adopt_legacy=True,prefix_sha=prefix_sha,exclusive=True)
            except RuntimeError as error:
                require('Legacy partial still open' in str(error),'Rejection was not owned-open-process gate')
                rejection=dict(status='EXPECTED_REJECTION_OWNED_CHILD_OPEN',error=str(error),actual_child=actual,legacy_after=dict(size=legacy.stat().st_size,inode=legacy.stat().st_ino,sha256=sha(legacy)),resume_created=(directory/(name+'.resume-v1.transfer')).exists())
                require(rejection['legacy_after']==before and not rejection['resume_created'],'Rejected adoption changed file')
                result['receipts'].append(write_record(directory,'adopt-open-rejection.json',rejection))
            else:raise RuntimeError('Open-child adoption unexpectedly accepted')
            child.stdin.write(b'X');child.stdin.flush();child.stdin.close();child.wait(timeout=20)
            require(child.returncode==0 and not (Path('/proc')/str(child.pid)).exists(),'Child failed exit/reap or actualPID still remains')
            result['child_exit']=dict(pid=child.pid,ppid=actual['ppid'],starttime=actual['starttime'],returncode=child.returncode,actual_proc_remaining=False,wait_reaped=True)
            result['receipts'].append(write_record(directory,'child-exit.json',result['child_exit']))
            adopted=execute(directory,name,size,expected,'adopt',prefix_size,0,bytes_sha(b''),io.BytesIO(b''),hostname=hostname,adopt_legacy=True,prefix_sha=prefix_sha,exclusive=True)
            require(adopted['status']=='ADOPTED_EXCLUSIVE_PREFIX' and adopted['size']==prefix_size and adopted['sha256']==prefix_sha and adopted['open_legacy_pids']==[],'Released-child prefix adoption failed')
            result['receipts'].append(write_record(directory,'adopt-after-exit.json',adopted))
            offset=prefix_size;middle=(size+prefix_size)//2
            for index,stop in enumerate([middle,size]):
                block=data[offset:stop];receipt=execute(directory,name,size,expected,'append',offset,len(block),bytes_sha(block),io.BytesIO(block),hostname=hostname)
                require(receipt['status']=='APPENDED' and receipt['size']==stop and receipt['block_sha256']==bytes_sha(block),'Tiny append ACK differs')
                result['receipts'].append(write_record(directory,f'chunk-{index}.json',dict(receipt,input_offset=offset,length=len(block))))
                offset=stop
            final=execute(directory,name,size,expected,'finalize',size,0,bytes_sha(b''),io.BytesIO(b''),hostname=hostname)
            require(final['status']=='VERIFIED_FINAL' and final['sha256']==expected and (directory/name).read_bytes()==data,'Final fixture fullSHA differs')
            require(not legacy.exists() and not (directory/(name+'.resume-v1.transfer')).exists(),'Owned partials remain after verified rename')
            result['receipts'].append(write_record(directory,'final-sha.json',dict(final,actual_size=(directory/name).stat().st_size,actual_sha256=sha(directory/name))))
            result['status']='OWNED_OPEN_REJECTION_THEN_RESUME_FULL_SHA_PASS'
        except BaseException as error:
            result['status']='FAILED_CPU_TRANSPORT_PROBE';result['error']=repr(error)
            raise
        finally:
            if child is not None and child.poll() is None:
                try:
                    now=identity(child.pid)
                    require(actual is not None and now['starttime']==actual['starttime'] and now['ppid']==os.getpid(),'Owned child cleanup identity drift')
                    child.stdin.close();child.wait(timeout=20)
                except Exception as error:cleanup_error=repr(error)
            result['cleanup']=dict(owned_child_remaining=child is not None and child.poll() is None,error=cleanup_error)
            result['actual_remaining_pids']=[child.pid] if child is not None and (Path('/proc')/str(child.pid)).exists() else []
            result['cleanup_passed']=not result['actual_remaining_pids'] and cleanup_error is None
            write_record(directory,'tiny-result.json',result)
    require(result['cleanup_passed'],'Owned tinyfile process cleanup failed')
    return result


class Collection:
    collected=0;passed=0;failed=0;skipped=0
    def pytest_collection_finish(self,session):self.collected=len(session.items)
    def pytest_runtest_logreport(self,report):
        if report.passed and report.when=='call':self.passed+=1
        if report.failed:self.failed+=1
        if report.skipped:self.skipped+=1


def main():
    job,parent,package=preflight();started=time.time();directory=ROOT/'logs'/('topd-v6-transport-native-'+job);directory.mkdir(mode=0o700)
    fixture=directory/'tiny-fixture';fixture.mkdir(mode=0o700)
    sys.path.insert(0,str(CANDIDATE));import transfer
    require(sha(CANDIDATE/'transfer.py')==TRANSFER_SHA,'Relay changed before execution')
    namespace={};exec(transfer.RECEIVER,namespace);tiny=tiny_probe(fixture,namespace['execute'],job=job)
    # Only a logs-scoped metadata mirror is supplied to the frozen tests. The
    # immutable CLI and official workspace/asset directories are never changed.
    cpu_base=directory/'cpu-test-root';(cpu_base/'outputs').mkdir(parents=True,mode=0o700)
    (cpu_base/'outputs/qwen2.5-32b-official-metadata.json').write_bytes((CANDIDATE/'official-metadata.json').read_bytes())
    transfer.BASE=cpu_base
    import pytest,torch
    collection=Collection();exitcode=pytest.main([str(CANDIDATE/'test_transfer.py'),str(CANDIDATE/'test_downloader_v6.py'),'--basetemp',str(directory/'pytest-temporary'),'-o','cache_dir='+str(directory/'pytest-cache'),'-q','-ra'],plugins=[collection])
    require(exitcode==0 and collection.collected==collection.passed==44 and collection.failed==collection.skipped==0,'Native candidate CPU44 failed/skipped or collection differs')
    require(not torch.cuda.is_initialized(),'Parent fixture initialized CUDA')
    inventory={str(p.relative_to(directory)):sha(p) for p in directory.rglob('*') if p.is_file()}
    record=dict(scope='OWNED_CPU_TRANSPORT_CONTRACT_ONLY_NOT_ALGORITHM_ACCEPTANCE',status='NATIVE_CPU44_AND_OWNED_PROC_TINY_PASS',
        job=job,host=socket.gethostname(),actual_parent=parent,cuda_initialized=False,torch_version=torch.__version__,
        candidate_archive_sha256=ARCHIVE_SHA,transfer_sha256=TRANSFER_SHA,official_metadata_sha256=METADATA_SHA,
        supplement_files=package['files'],supplement_manifest_sha256=sha(HERE/'files.json'),
        native_collected=collection.collected,native_passed=collection.passed,native_failed=collection.failed,native_skipped=collection.skipped,
        tiny_status=tiny['status'],owned_child_identity=tiny['actual_child_open'],actual_remaining_pids=tiny['actual_remaining_pids'],cleanup_passed=tiny['cleanup_passed'],
        receipts=inventory,started_epoch=started,completed_epoch=time.time())
    write_record(directory,'result.json',record);print(json.dumps(record,sort_keys=True,allow_nan=False),flush=True)


if __name__=='__main__':main()
