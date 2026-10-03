"""CPU-only real fcntl locks and /proc identity fixtures; no GPU operations."""
import fcntl
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import uuid

import pytest
from search_r1.baselines import preflight


@pytest.fixture
def leased(monkeypatch):
    uid='GPU-supervised-test-'+uuid.uuid4().hex
    directory=Path('/tmp')/f'search-r1-gpu-leases-{os.getuid()}'
    directory.mkdir(mode=0o700,exist_ok=True)
    path=directory/(uid+'.lock')
    fd=os.open(path,os.O_CREAT|os.O_RDWR,0o600)
    fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    inventory={'0':dict(uuid=uid,memory_used=90.,memory_total=100.,utilization=90.)}
    monkeypatch.setattr(preflight,'gpu_snapshot',lambda:(inventory,set()))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    monkeypatch.setenv('SEARCH_R1_GPU_PREFLIGHT','1')
    monkeypatch.setenv('SEARCH_R1_GPU_LEASES',json.dumps([dict(uuid=uid,fd=fd)]))
    monkeypatch.delenv('SEARCH_R1_GPU_RESOURCE_GUARD',raising=False)
    try:yield uid,path,fd,inventory
    finally:os.close(fd);path.unlink()


@pytest.fixture
def guard(tmp_path,monkeypatch):
    run=tmp_path/'own/runs/test-run';run.mkdir(parents=True)
    script=run.parent.parent/'tools/resource_watch.py';script.parent.mkdir();script.write_text('# identity fixture')
    proc=tmp_path/'proc';proc.mkdir()
    def identity(pid,parent,ticks):
        p=proc/str(pid);p.mkdir(exist_ok=True)
        fields=['S',str(parent)]+['0']*18;fields[19]=str(ticks)
        (p/'stat').write_text(f'{pid} (fixture with spaces) '+' '.join(fields))
        return p
    identity(123,400,88);p=identity(400,1,56)
    argv=[sys.executable,str(script),'--run',str(run),'--','fixture-command']
    (p/'cmdline').write_bytes(b'\0'.join(v.encode()for v in argv)+b'\0')
    (p/'environ').write_bytes(('BAPO_RUN_ID='+str(run)).encode()+b'\0')
    (p/'exe').symlink_to(sys.executable)
    real_path=Path
    monkeypatch.setattr(preflight,'Path',lambda *a,**kw:real_path(proc if a and a[0]=='/proc' else a[0],*a[1:],**kw))
    monkeypatch.setattr(preflight.os,'getpid',lambda:123)
    monkeypatch.setenv('BAPO_RUN_ID',str(run))
    monkeypatch.setenv('SEARCH_R1_GPU_RESOURCE_GUARD',json.dumps(dict(pid=400,start_ticks=56,run=str(run))))
    return p,run,script


def test_busy_gpu_requires_real_guard_and_exclusive_fd(leased,guard):
    preflight.validate_gpu_lease(SimpleNamespace(trainer=SimpleNamespace(nnodes=1,n_gpus_per_node=1)))
    assert os.get_inheritable(leased[2])


def test_busy_legacy_without_guard_is_rejected(leased):
    with pytest.raises(RuntimeError,match='15%'):preflight.validate_inherited_gpu_leases(['0'])
    with pytest.raises(ValueError,match='guard'):preflight.validate_inherited_gpu_leases(['0'],require_guard=True)


@pytest.mark.parametrize('kind',['reused','unrelated','self','wrong_marker','wrong_argument','argument_spoof','zombie'])
def test_guard_identity_counterexamples(leased,guard,monkeypatch,kind):
    p,run,script=guard
    obj=json.loads(os.environ['SEARCH_R1_GPU_RESOURCE_GUARD'])
    if kind=='reused':obj['start_ticks']+=1
    elif kind=='unrelated':obj['pid']=401
    elif kind=='self':obj['pid']=123;obj['start_ticks']=88
    elif kind=='wrong_marker':(p/'environ').write_bytes(b'BAPO_RUN_ID=/other\0')
    elif kind=='wrong_argument':(p/'cmdline').write_bytes(b'\0'.join(v.encode()for v in [sys.executable,str(script),'--run','/other']))
    elif kind=='argument_spoof':(p/'cmdline').write_bytes(b'\0'.join(v.encode()for v in [sys.executable,'/other.py',str(script),'--run',str(run)]))
    elif kind=='zombie':(p/'stat').write_text((p/'stat').read_text().replace(') S',') Z'))
    monkeypatch.setenv('SEARCH_R1_GPU_RESOURCE_GUARD',json.dumps(obj))
    with pytest.raises(ValueError):preflight.validate_inherited_gpu_leases(['0'],require_guard=True)


def test_same_inode_different_unlocked_descriptor_is_not_inheritance(leased,guard,monkeypatch):
    uid,path,fd,_=leased
    other=os.open(path,os.O_RDWR)
    try:
        monkeypatch.setenv('SEARCH_R1_GPU_LEASES',json.dumps([dict(uuid=uid,fd=other)]))
        with pytest.raises(ValueError,match='descriptor does not hold'):preflight.validate_inherited_gpu_leases(['0'],True)
    finally:os.close(other)


def test_shared_lock_is_rejected_without_upgrade(leased,guard):
    fcntl.flock(leased[2],fcntl.LOCK_SH)
    with pytest.raises(ValueError,match='exclusively'):preflight.validate_inherited_gpu_leases(['0'],True)
    probe=os.open(leased[1],os.O_RDWR)
    try:fcntl.flock(probe,fcntl.LOCK_SH|fcntl.LOCK_NB)
    finally:os.close(probe)


def test_wrong_file_uuid_and_closed_fd_rejected(leased,guard,monkeypatch,tmp_path):
    uid,path,fd,_=leased
    other=os.open(tmp_path/'other',os.O_CREAT|os.O_RDWR,0o600)
    try:
        monkeypatch.setenv('SEARCH_R1_GPU_LEASES',json.dumps([dict(uuid=uid,fd=other)]))
        with pytest.raises(ValueError,match='identify'):preflight.validate_inherited_gpu_leases(['0'],True)
    finally:os.close(other)
    monkeypatch.setenv('SEARCH_R1_GPU_LEASES',json.dumps([dict(uuid='GPU-wrong',fd=fd)]))
    with pytest.raises(ValueError,match='hardware'):preflight.validate_inherited_gpu_leases(['0'],True)
    monkeypatch.setenv('SEARCH_R1_GPU_LEASES',json.dumps([dict(uuid=uid,fd=other)]))
    with pytest.raises(OSError):preflight.validate_inherited_gpu_leases(['0'],True)


def test_main_reuses_without_reopening_or_idle_checks(leased,guard,monkeypatch):
    monkeypatch.setattr(sys,'argv',['preflight','--','fixture-command'])
    calls=[];monkeypatch.setattr(preflight.os,'execvp',lambda cmd,args:calls.append((cmd,args)))
    preflight.main()
    assert calls==[('fixture-command',['fixture-command'])]


def test_first_launcher_still_double_checks_idle_and_locks(monkeypatch):
    uid='GPU-supervised-first-'+uuid.uuid4().hex
    inventory={'0':dict(uuid=uid,memory_used=0.,memory_total=100.,utilization=0.)}
    for n in ['SEARCH_R1_GPU_PREFLIGHT','SEARCH_R1_GPU_LEASES','SEARCH_R1_GPU_RESOURCE_GUARD']:monkeypatch.delenv(n,raising=False)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0');monkeypatch.setattr(sys,'argv',['preflight','--','fixture-command'])
    snapshots=[];monkeypatch.setattr(preflight,'gpu_snapshot',lambda:(snapshots.append(1)or inventory,set()))
    monkeypatch.setattr(preflight.time,'sleep',lambda _:None)
    monkeypatch.setattr(preflight.os,'execvp',lambda *_:None)
    preflight.main();records=json.loads(os.environ['SEARCH_R1_GPU_LEASES']);fd=records[0]['fd']
    try:
        assert len(snapshots)==2
        preflight.validate_inherited_gpu_leases(['0'])
    finally:os.close(fd);(Path('/tmp')/f'search-r1-gpu-leases-{os.getuid()}'/(uid+'.lock')).unlink()


def test_guard_reparent_or_pid_change_during_read_rejected(leased,guard,monkeypatch):
    original=preflight._process_identity;count=0
    def changing(pid):
        nonlocal count
        value=original(pid)
        if pid==400:
            count+=1
            if count>=3:return (value[0]+1,value[1])
        return value
    monkeypatch.setattr(preflight,'_process_identity',changing)
    with pytest.raises(ValueError,match='ancestry changed'):preflight.validate_inherited_gpu_leases(['0'],True)


@pytest.mark.skipif(not Path('/proc/self/stat').exists(),reason='Real Linux proc identity required')
def test_actual_linux_guard_ancestor_and_inherited_fds_cpu_only(leased,tmp_path):
    import subprocess
    uid,path,fd,inventory=leased
    run=tmp_path/'actual/runs/test-run';run.mkdir(parents=True)
    script=run.parent.parent/'tools/resource_watch.py';script.parent.mkdir()
    child="from types import SimpleNamespace; from search_r1.baselines import preflight as p; p.gpu_snapshot=lambda: ("+repr(inventory)+",set()); p.validate_gpu_lease(SimpleNamespace(trainer=SimpleNamespace(nnodes=1,n_gpus_per_node=1))); print('ACTUAL_LINUX_ANCESTOR_AND_LOCK_FD_OK')"
    script.write_text('import os,json,subprocess,sys\nfrom search_r1.baselines import preflight as p\nenv=dict(os.environ)\nenv["SEARCH_R1_GPU_RESOURCE_GUARD"]=json.dumps(dict(pid=os.getpid(),start_ticks=p._process_identity(os.getpid())[0],run=env["BAPO_RUN_ID"]))\nsubprocess.check_call([sys.executable,"-c",'+repr(child)+'],env=env,pass_fds=('+str(fd)+',))\n')
    env=dict(os.environ);env['BAPO_RUN_ID']=str(run)
    repo=Path(__file__).resolve().parents[2];env['PYTHONPATH']=str(repo)
    result=subprocess.run([sys.executable,str(script),'--run',str(run)],env=env,pass_fds=(fd,),capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    assert 'ACTUAL_LINUX_ANCESTOR_AND_LOCK_FD_OK' in result.stdout
