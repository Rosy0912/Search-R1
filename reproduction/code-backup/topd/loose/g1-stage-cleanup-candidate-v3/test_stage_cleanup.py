import importlib.util,pathlib,types,os,hashlib,pytest
s=importlib.util.spec_from_file_location('stage',pathlib.Path(__file__).with_name('stage_cleanup.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
SOURCE=pathlib.Path(__file__).with_name('owner_support.py')
if not SOURCE.is_file():SOURCE=pathlib.Path(__file__).resolve().parents[1]/'g1-staged-calibration-producer-v3/owner_support.py'
def row(pid,parent,start=None,uid=None):return {'pid':pid,'ppid':parent,'uid':os.getuid() if uid is None else uid,'starttime_ticks':pid if start is None else start,'cgroup':['1:freezer:/job']}
def setup(rows):
 t=types.SimpleNamespace(cgroup_pids=lambda scope:set(rows));return m.StageScope(t,'scope','/job',proc_read=lambda p:dict(rows[p]),fd_open=lambda p:p,fd_read=lambda p:p,fd_close=lambda f:None)
def test_parent_diagnostics_and_sibling_model_not_stage_owned():
 rows={10:row(10,1),20:row(20,10),30:row(30,20),40:row(40,10)};s=setup(rows);s.register(20)
 assert s.pids('scope')=={20,30}
 assert s.events[-1]['same_job_siblings_not_stage_owned']==[10,40]
def test_registered_orphan_survives_parent_exit_and_sibling_appear():
 rows={20:row(20,10),30:row(30,20)};s=setup(rows);s.register(20);assert s.pids('scope')=={20,30}
 del rows[20];rows[30]['ppid']=1;rows[40]=row(40,10)
 assert s.pids('scope')=={30}
@pytest.mark.parametrize('field,value',[('starttime_ticks',99),('uid',-1)])
def test_registered_PID_reuse_or_UID_drift_is_fatal(field,value):
 rows={20:row(20,10)};s=setup(rows);s.register(20);rows[20][field]=value
 with pytest.raises(RuntimeError,match='reused/UID'):s.pids('scope')
def test_foreign_child_UID_is_not_owned_by_parent_label():
 rows={20:row(20,10),30:row(30,20,uid=-1)};s=setup(rows);s.register(20)
 with pytest.raises(RuntimeError,match='UID/job'):s.pids('scope')
def test_actual_original_cleanup_AST_does_not_signal_parent_query(monkeypatch):
 rows={10:row(10,1),20:row(20,10),30:row(30,20)};signals=[];clock=[0.]
 def signal(fd,kind):signals.append(fd);rows.pop(fd,None)
 fakeos=types.SimpleNamespace(pidfd_open=lambda p:p,close=lambda f:None)
 faketime=types.SimpleNamespace(monotonic=lambda:clock[0],sleep=lambda n:clock.__setitem__(0,clock[0]+n))
 tools=types.SimpleNamespace(__file__=str(SOURCE),cgroup_pids=lambda scope:set(rows),process_identity=lambda p:dict(rows[p]),os=fakeos,time=faketime,signal=types.SimpleNamespace(SIGTERM=15,SIGKILL=9,pidfd_send_signal=signal),require=m.require,Path=pathlib.Path)
 obj=m.owned_stage(tools,'scope','/job',expected_tools_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),proc_read=lambda p:dict(rows[p]),fd_open=lambda p:p,fd_read=lambda p:p,fd_close=lambda f:None);obj.stage_scope.register(20)
 # This same-job parent diagnostic starts AFTER stage baseline, matching3124 risk.
 rows[40]=row(40,10)
 class Membership:
  def __init__(self,*a):pass
  def read_text(self):return '1:freezer:/job\n'
 # Replace only cloned class dependency; original source file remains untouched.
 original=obj.cleanup.__closure__
 cloned=next(c.cell_contents for c in original if callable(c.cell_contents))
 cloned.__func__.__globals__['Path']=Membership
 obj.cleanup()
 assert set(signals)=={20,30} and 10 in rows and 40 in rows
 assert obj.cleanup_record['actual_cgroup_empty'] and obj.cleanup_record['ownership_scope'].endswith('NOT_WHOLE_JOB')
def test_source_drift_refused_before_stage_initialization():
 tools=types.SimpleNamespace(__file__=str(SOURCE))
 with pytest.raises(RuntimeError,match='tools drift'):m.owned_stage(tools,'scope','/job',expected_tools_sha256='0'*64)


def test_sealed_stage_cannot_adopt_new_child_during_cleanup():
 rows={20:row(20,10)};s=setup(rows);s.register(20);s.seal();rows[30]=row(30,20)
 with pytest.raises(RuntimeError,match='New stage descendant'):s.pids('scope')

def test_registration_actual_pidfd_must_match_PID():
 rows={20:row(20,10)};s=setup(rows);s.fd_read=lambda f:-1
 with pytest.raises(RuntimeError,match='pidfd registration'):s.register(20)


@pytest.mark.skipif(not hasattr(os,'pidfd_open'),reason='actual Linux proc/pidfd native contract')
def test_Linux_real_nested_cleanup_preserves_parent_concurrent_sibling(tmp_path):
 import subprocess,sys,time,signal
 relative=next(v.split(':',2)[2] for v in pathlib.Path('/proc/self/cgroup').read_text().splitlines() if v.split(':',2)[1]=='freezer')
 tracked={os.getpid()};query=None;owned=None
 def census(scope):
  result=set()
  for pid in tracked:
   try:
    state=pathlib.Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()[0]
    if state not in ('Z','X'):result.add(pid)
   except FileNotFoundError:pass
  return result
 def identity(pid):return m.StageScope.proc(None,pid)
 (tmp_path/'repo').mkdir();tools=types.SimpleNamespace(__file__=str(SOURCE),cgroup_pids=census,process_identity=identity,os=os,time=time,signal=signal,require=m.require,Path=pathlib.Path,subprocess=subprocess,ROOT=tmp_path)
 try:
  owned=m.owned_stage(tools,'scope',relative,expected_tools_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest())
  script="import subprocess,sys,time,pathlib; c=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)']);pathlib.Path(sys.argv[1]).write_text(str(c.pid));time.sleep(120)"
  pidfile=tmp_path/'grandchild';child=owned.spawn([sys.executable,'-c',script,str(pidfile)],tmp_path/'child.log');tracked.add(child.pid)
  deadline=time.monotonic()+5
  while not pidfile.exists() and time.monotonic()<deadline:time.sleep(.01)
  assert pidfile.exists();grandchild=int(pidfile.read_text());tracked.add(grandchild);assert owned.stage_scope.pids('scope')=={child.pid,grandchild}
  # Same parent and actual job, but created AFTER stage's registration/baseline.
  query=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)']);tracked.add(query.pid)
  owned.cleanup();assert query.poll() is None and os.getpid() in census('scope')
  signalled={v['pid'] for v in owned.cleanup_record['signals']};assert signalled=={child.pid,grandchild} and query.pid not in signalled
  assert owned.cleanup_record['ownership_scope'].endswith('NOT_WHOLE_JOB')
 finally:
  if query is not None:query.terminate();query.wait(timeout=5)
  if owned is not None:
   for fd in list(owned.stage_scope.fds.values()):
    try:signal.pidfd_send_signal(fd,signal.SIGKILL)
    except ProcessLookupError:pass
   for child,fd in owned.children:
    if child.poll() is None:child.kill();child.wait(timeout=5)
   owned.stage_scope.close()

@pytest.mark.parametrize('point',['pidfd_open','second_proc'])
def test_unregistered_short_lived_descendant_disappears_without_adoption(point):
 rows={20:row(20,10),30:row(30,20)};s=setup(rows);s.register(20);closed=[];s.fd_close=closed.append
 reads={}
 def read(pid):
  reads[pid]=reads.get(pid,0)+1
  if pid==30 and point=='second_proc' and reads[pid]>=2:rows.pop(30,None)
  if pid not in rows:raise FileNotFoundError(pid)
  return dict(rows[pid])
 s.proc_read=read
 def openfd(pid):
  if pid==30 and point=='pidfd_open':rows.pop(30);raise ProcessLookupError(pid)
  return pid
 s.fd_open=openfd
 assert s.pids('scope')=={20}
 assert 30 not in s.ledger and 30 not in s.fds
 assert s.events[-2]['scope'].endswith('NOT_OWNED')
 assert closed==([30] if point=='second_proc' else [])

def test_registration_PPID_reparent_preserves_exact_process_identity():
 rows={20:row(20,10),30:row(30,20)};s=setup(rows);s.register(20)
 def openfd(pid):
  if pid==30:rows[30]['ppid']=1
  return pid
 s.fd_open=openfd
 assert s.pids('scope')=={20,30} and s.ledger[30]['ppid']==1

@pytest.mark.parametrize('field,value',[('starttime_ticks',999),('uid',-1),('cgroup',['1:freezer:/foreign'])])
def test_second_read_live_identity_change_not_natural_exit(field,value):
 rows={20:row(20,10),30:row(30,20)};s=setup(rows);s.register(20)
 def openfd(pid):
  if pid==30:rows[30][field]=value
  return pid
 s.fd_open=openfd
 with pytest.raises(RuntimeError,match='identity changed'):s.pids('scope')
 assert 30 not in s.ledger

def test_lookup_error_with_live_replacement_is_not_silently_skipped():
 rows={20:row(20,10),30:row(30,20)};s=setup(rows);s.register(20)
 def openfd(pid):
  if pid==30:rows[30]['starttime_ticks']=999;raise ProcessLookupError(pid)
  return pid
 s.fd_open=openfd
 with pytest.raises(ProcessLookupError):s.pids('scope')
 assert 30 not in s.ledger
