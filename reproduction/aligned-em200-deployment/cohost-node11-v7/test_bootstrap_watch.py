import importlib.util,os,sys,fcntl,json
from pathlib import Path
import pytest
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import resource_watch as w
import bootstrap as b

def raw(free=39236,sm=98):
 return '\n'.join(f'{i}, GPU-{i}, 81559, {free}, {81559-free}, {sm}'for i in range(8))
def test_cohost_high_foreign_memory_sm_allowed():
 rows=w.inventory(raw(),[str(i)for i in range(8)])
 assert w.admission(rows,30720) and all(r['sm_percent']==98 for r in rows)
def test_insufficient_or_nonfinite_memory_rejected():
 assert not w.admission(w.inventory(raw(30000),list(map(str,range(8)))),30720)
 with pytest.raises(RuntimeError):w.inventory(raw().replace('39236','nan'),list(map(str,range(8))))
 with pytest.raises(RuntimeError):w.admission([],float('nan'))
def test_duplicate_gpu_refused():
 with pytest.raises(RuntimeError):w.inventory(raw(),['0']*8)
def test_real_ofd_lease_blocked_and_inherited(tmp_path):
 rows=w.inventory(raw(),list(map(str,range(8))));leases=b.acquire(rows,tmp_path/'leases')
 try:
  for item in leases:
   assert os.get_inheritable(item['fd'])
   probe=os.open(tmp_path/'leases'/(item['uuid']+'.lock'),os.O_RDWR)
   try:
    with pytest.raises(BlockingIOError):fcntl.flock(probe,fcntl.LOCK_SH|fcntl.LOCK_NB)
   finally:os.close(probe)
 finally:
  for item in leases:os.close(item['fd'])
def test_lease_symlink_refused(tmp_path):
 d=tmp_path/'leases';d.mkdir();foreign=tmp_path/'foreign';foreign.write_text('keep');(d/'GPU-0.lock').symlink_to(foreign)
 with pytest.raises(OSError):b.acquire(w.inventory(raw(),list(map(str,range(8)))),d)
 assert foreign.read_text()=='keep'
def test_cleanup_pidfd_birth_recheck_prevents_reused_pid(monkeypatch,tmp_path):
 monkeypatch.setenv('SLURM_JOB_ID','9000');calls=[];snapshots=iter([{99:1},{99:2},{}])
 monkeypatch.setattr(w,'own_processes',lambda run:next(snapshots,{}));monkeypatch.setattr(w.os,'pidfd_open',lambda pid:100,raising=False);monkeypatch.setattr(w.os,'close',lambda fd:calls.append(('close',fd)));monkeypatch.setattr(w.signal,'pidfd_send_signal',lambda fd,sig:calls.append(('signal',sig)),raising=False)
 assert w.cleanup(tmp_path)['remaining']==[]
 assert not any(c[0]=='signal'for c in calls)
def test_source_and_old_monitor_unchanged_contract():
 s=(HERE/'resource_watch.py').read_text();boot=(HERE/'bootstrap.py').read_text()
 assert "phase_pid=child.pid" in s and 'signal.pidfd_send_signal' in s
 assert 'COHOST_FREE_RESERVE_NO_FOREIGN_SM_OR_MEMORY_PERCENT_GATE' in s
 assert "[python,str(watch),'--run'" in boot and "'SEARCH_R1_GPU_LEASES'" in boot
 assert "'COHOST_WATCHER_ADMITTED_RUN'" in s
