import ast,importlib.util,json,os,signal,subprocess,sys,time
from pathlib import Path
from unittest.mock import patch
import pytest
from types import SimpleNamespace
SOURCE=Path(__file__).resolve().parents[1]/'tree_shared_owned_runtime_fixed.py'
spec=importlib.util.spec_from_file_location('owner_cleanup_candidate',SOURCE);runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)

@pytest.mark.parametrize('state,accepted',[('Z',True),('R',False),('S',False),('D',False),('T',False),('X',False),('?',False),('',False)])
def test_residual_only_explicit_z_accepted(tmp_path,state,accepted):
 (tmp_path/'cgroup.procs').write_text('100\n');own=runtime.OwnedProcesses(tmp_path,'/slurm/job_123/step_0');calls=[];old=Path.read_text
 def read(path,*a,**k):
  if str(path)=='/proc/200/cgroup':return '0::/slurm/job_123/step_0\n'
  if str(path)=='/proc/200/stat':return f'200 (name with ) embedded) {state} 1 2 3'
  return old(path,*a,**k)
 with patch.object(runtime,'cgroup_pids',lambda scope:{100,200}),patch.object(Path,'read_text',read),patch.object(runtime.os,'pidfd_open',lambda pid:pid,create=True),patch.object(runtime.os,'close',lambda fd:None),patch.object(runtime.signal,'pidfd_send_signal',lambda fd,sig:calls.append((fd,sig)),create=True),patch.object(runtime.time,'sleep',lambda x:None):
  if accepted:own.cleanup();assert own.cleanup_residual=={'200':'Z'}
  else:
   with pytest.raises(RuntimeError):own.cleanup()
 assert all(pid==200 for pid,sig in calls) and len(calls)>=12 and calls[0][1]==signal.SIGTERM and calls[1][1]==signal.SIGKILL

@pytest.mark.parametrize('error',[PermissionError('unreadable'),OSError('unknown proc failure')])
def test_unreadable_residual_rejects(tmp_path,error):
 (tmp_path/'cgroup.procs').write_text('');own=runtime.OwnedProcesses(tmp_path,'/slurm/job_123/step_0');old=Path.read_text
 def read(path,*a,**k):
  if str(path)=='/proc/200/cgroup':return '0::/slurm/job_123/step_0'
  if str(path)=='/proc/200/stat':raise error
  return old(path,*a,**k)
 with patch.object(runtime,'cgroup_pids',lambda scope:{200}),patch.object(Path,'read_text',read),patch.object(runtime.os,'pidfd_open',lambda pid:pid,create=True),patch.object(runtime.os,'close',lambda fd:None),patch.object(runtime.signal,'pidfd_send_signal',lambda *a:None,create=True),patch.object(runtime.time,'sleep',lambda x:None),pytest.raises(type(error)):own.cleanup()

def test_real_local_cpu_child_cleanup_and_exception_reaping(tmp_path):
 # Real CPU subprocesses; pidfd/cgroup interfaces mocked because Mac has no /proc/pidfd.
 (tmp_path/'repo').mkdir();(tmp_path/'cgroup.procs').write_text(str(os.getpid())+'\n');own=runtime.OwnedProcesses(tmp_path,'/slurm/job_123/step_0');children={};calls=[];old=Path.read_text
 def active(scope):return {os.getpid()}|{pid for pid,p in children.items() if p.poll() is None}
 def openfd(pid):
  if children[pid].poll() is not None:raise ProcessLookupError(pid)
  return pid
 def send(fd,sig):calls.append((fd,sig));children[fd].send_signal(sig)
 def read(path,*a,**k):
  if str(path).startswith('/proc/') and str(path).endswith('/cgroup'):return '0::/slurm/job_123/step_0\n'
  return old(path,*a,**k)
 def popped(*a,**k):
  child=original(*a,**k);children[child.pid]=child;return child
 original=runtime.subprocess.Popen
 fake_os=SimpleNamespace(**vars(os));fake_os.pidfd_open=openfd;fake_os.close=lambda fd:None
 with patch.object(runtime,'ROOT',tmp_path),patch.object(runtime,'cgroup_pids',active),patch.object(runtime.subprocess,'Popen',popped),patch.object(runtime,'os',fake_os),patch.object(runtime.signal,'pidfd_send_signal',send,create=True),patch.object(Path,'read_text',read):
  live=own.spawn([sys.executable,'-c','import time;time.sleep(60)'],tmp_path/'live.log')
  bad=own.spawn([sys.executable,'-c','raise RuntimeError("intentional own CPU failure")'],tmp_path/'failure.log')
  deadline=time.monotonic()+5
  while bad.poll() is None:
   assert time.monotonic()<deadline;time.sleep(.01)
  assert bad.returncode!=0 and 'intentional own CPU failure' in (tmp_path/'failure.log').read_text()
  try:own.cleanup()
  finally:
   for c in children.values():
    if c.poll() is None:c.terminate()
    c.wait(timeout=5)
 assert live.poll() is not None and bad.poll() is not None and all(pid in children for pid,sig in calls) and own.cleanup_residual=={}
 assert all(c.returncode is not None for c in children.values())

def test_foreign_scope_and_baseline_no_signal(tmp_path):
 (tmp_path/'cgroup.procs').write_text('100\n');own=runtime.OwnedProcesses(tmp_path,'/slurm/job_123/step_0');calls=[];old=Path.read_text
 def read(path,*a,**k):
  if str(path)=='/proc/200/cgroup':return '0::/slurm/job_123/step_01\n'
  if str(path)=='/proc/200/stat':return '200 (neighbor) S 1 2'
  return old(path,*a,**k)
 with patch.object(runtime,'cgroup_pids',lambda scope:{100,200}),patch.object(Path,'read_text',read),patch.object(runtime.os,'pidfd_open',lambda pid:pid,create=True),patch.object(runtime.os,'close',lambda fd:None),patch.object(runtime.signal,'pidfd_send_signal',lambda *a:calls.append(a),create=True),patch.object(runtime.time,'sleep',lambda x:None),pytest.raises(RuntimeError):own.cleanup()
 assert calls==[]
