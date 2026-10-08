"""Independent source/CPU counterexamples; no GPU, real /proc or target assets."""
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch

SOURCE=Path(__file__).with_name('resource_adapter.py')
digest=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('independent_resource_adapter_snapshot',SOURCE)
A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)


def lost_lease_still_accepted():
    """Real foreign flock on eight synthetic CPU-only files, actual check body.

    Mock host/PID/cgroup/guard preconditions only. Real fd inode checks and flock
    calls show a supervisor descriptor may identify the right file without owning
    its lock after a verified startup ownership has been lost.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp).resolve();base=root/'tmp'/('search-r1-gpu-leases-'+str(os.getuid()));base.mkdir(parents=True)
        uuids=['synthetic-cpu-review-'+str(i) for i in range(8)];fds=[]
        for uid in uuids:
            fd=os.open(str(base/(uid+'.lock')),os.O_CREAT|os.O_RDWR,0o600)
            fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);fds.append(fd)
        startup_verified=True
        for fd in fds:fcntl.flock(fd,fcntl.LOCK_UN)
        ready_read,ready_write=os.pipe();end_read,end_write=os.pipe()
        child=os.fork()
        if child==0:
            os.close(ready_read);os.close(end_write);held=[]
            try:
                for uid in uuids:
                    fd=os.open(str(base/(uid+'.lock')),os.O_RDWR);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);held.append(fd)
                os.write(ready_write,b'1');os.close(ready_write);os.read(end_read,1)
            finally:
                for fd in held:os.close(fd)
                os.close(end_read);os._exit(0)
        os.close(ready_write);os.close(end_read)
        require_ready=os.read(ready_read,1);os.close(ready_read)
        assert require_ready==b'1'
        guard=root/'guard.jsonl';guard.write_text(json.dumps(dict(job_id='99999',epoch=time.time(),
            gpus=[dict(uuid=u,foreign_memory_fraction=.01,foreign_sm_upper=0.) for u in uuids]))+'\n')
        state=dict(job_id='99999',supervisor=dict(pid=777777,start_ticks=111),guard=dict(pid=777778,start_ticks=222),
            guard_path=str(guard),gpu_uuids=uuids,leases=[dict(uuid=uid,fd=fd) for uid,fd in zip(uuids,fds)])
        def fake_path(*parts):
            if len(parts)>=3 and parts[0]=='/proc':
                if parts[-1]=='cgroup':return SimpleNamespace(read_text=lambda:'0::/slice/job_99999/cpu_fixture\n')
                if parts[-2]=='fd':return SimpleNamespace(stat=lambda:os.fstat(int(parts[-1])))
                raise AssertionError('Unexpected /proc access '+repr(parts))
            if parts==('/tmp',):return root/'tmp'
            return Path(*parts)
        def fake_process(pid):
            if pid in A.PROTECTED:return 'S',A.PROTECTED[pid]
            return 'S',111 if pid==777777 else 222
        actual_supervisor_owned=0
        try:
            for fd in fds:
                try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:pass
                else:actual_supervisor_owned+=1;fcntl.flock(fd,fcntl.LOCK_UN)
            with (patch.object(A,'ROOT',root),patch.object(A,'ACTIVE',state),patch.object(A,'Path',fake_path),
                patch.object(A,'process',fake_process),patch.object(A.socket,'gethostname',lambda:'h100-9'),
                patch.dict(os.environ,{'SLURM_JOB_ID':'99999'})):
                receipt=A.check('independent_real_cpu_flock_counterexample')
        finally:
            os.close(end_write);os.waitpid(child,0)
            for fd in fds:os.close(fd)
        assert actual_supervisor_owned==0
        return dict(status='COUNTEREXAMPLE_REPRODUCED',startup_owner_verified=startup_verified,
            current_supervisor_owned_locks=actual_supervisor_owned,actual_foreign_process_held_locks=8,
            adapter_check_returned_success=receipt,scope='Actual check body, real synthetic inode/FD/flock; host/PID/cgroup/guard preconditions simulated')


if __name__=='__main__':
    # Continued-with clauses are written with parentheses for source portability.
    result=lost_lease_still_accepted()
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==digest,'Candidate adapter changed during this read-only review'
    result.update(adapter_path=str(SOURCE),adapter_sha256=digest,actual_gpu_calls=0,actual_asset_reads=0,
        real_background_processes_operated=0,actual_slurm_actions=0)
    path=SOURCE.with_name('ADAPTER_LEASE_COUNTEREXAMPLE.json')
    with path.open('x') as stream:json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False)
    print(json.dumps(result,sort_keys=True))
