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
 def identity(pid):return m.StageScope(types.SimpleNamespace(),'read-only',relative).proc(pid)
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


def test_registered_exit_between_census_and_pidfd_keeps_history_not_active():
 rows={20:row(20,10),30:row(30,20)};s=setup(rows);s.register(20);assert s.pids('scope')=={20,30}
 def fdread(fd):
  if fd==30:rows.pop(30,None);return -1
  return fd
 s.fd_read=fdread
 def read(pid):
  if pid not in rows:raise FileNotFoundError(pid)
  return dict(rows[pid])
 s.proc_read=read
 assert s.pids('scope')=={20}
 assert 30 in s.ledger and 30 in s.fds
 assert any(e.get('scope')=='REGISTERED_STAGE_PIDFD_DEAD_AND_PROC_ABSENT_NOT_ACTIVE' for e in s.events)

@pytest.mark.parametrize('replacement',[False,True])
def test_dead_registered_pidfd_with_live_proc_not_skipped(replacement):
 rows={20:row(20,10),30:row(30,20)};s=setup(rows);s.register(20);s.pids('scope')
 def fdread(fd):
  if fd==30:
   if replacement:rows[30]['starttime_ticks']=999
   return -1
  return fd
 s.fd_read=fdread
 with pytest.raises(RuntimeError,match='live/reused'):s.pids('scope')

@pytest.mark.parametrize('fd_value',[0,999])
def test_unknown_registered_pidfd_value_not_treated_as_dead(fd_value):
 rows={20:row(20,10),30:row(30,20)};s=setup(rows);s.register(20);s.pids('scope');s.fd_read=lambda fd:fd_value if fd==30 else fd
 with pytest.raises(RuntimeError,match='pidfd drift'):s.pids('scope')

def test_registered_exit_evidence_does_not_hide_original_UID_drift():
 rows={20:row(20,10),30:row(30,20)};s=setup(rows);s.register(20);s.pids('scope');rows[30]['uid']=-1;s.fd_read=lambda fd:-1 if fd==30 else fd
 with pytest.raises(RuntimeError,match='reused/UID'):s.pids('scope')


def test_registered_dead_qualification_rejects_same_job_subcgroup_migration():
 rows={20:row(20,10),30:row(30,20)};s=setup(rows);s.register(20);s.pids('scope');rows[30]['cgroup']=['1:freezer:/job/subgroup']
 def fdread(fd):
  if fd==30:rows.pop(30,None);return -1
  return fd
 s.fd_read=fdread
 with pytest.raises(RuntimeError,match='job/pidfd drift'):s.pids('scope')

@pytest.mark.parametrize('field,value',[('uid',-1),('cgroup',[])])
def test_unregistered_rejected_census_only_fresh_absence_skips_without_authority(tmp_path,field,value):
 rows={20:row(20,10),30:row(30,20)};rows[30][field]=value;calls={};path=tmp_path/'events.jsonl'
 def proc(pid):
  calls[pid]=calls.get(pid,0)+1
  if pid==30 and calls[pid]>1:raise FileNotFoundError('actual proc absent')
  return dict(rows[pid])
 s=m.StageScope(types.SimpleNamespace(cgroup_pids=lambda scope:set(rows)),'scope','/job',proc_read=proc,fd_open=lambda p:p,fd_read=lambda p:p,fd_close=lambda f:None,event_path=path);s.register(20)
 assert s.pids('scope')=={20} and 30 not in s.ledger and 30 not in s.fds
 import json
 events=[json.loads(v) for v in path.read_text().splitlines()];e=next(e for e in events if e['scope']=='ACTUAL_CENSUS_REJECTION_BEFORE_AUTHORITY')
 assert e['census_identity'][field]==value and e['fresh_proc_absent'] and e['registered'] is False

@pytest.mark.parametrize('fresh',[row(30,20,uid=-1),dict(row(30,20),cgroup=[]),row(30,20,start=999)])
def test_live_UID_job_or_reused_unknown_candidate_is_always_fatal_with_raw(fresh):
 rows={20:row(20,10),30:dict(row(30,20),cgroup=[])};calls={}
 def proc(pid):
  calls[pid]=calls.get(pid,0)+1
  return dict(fresh if pid==30 and calls[pid]>1 else rows[pid])
 s=m.StageScope(types.SimpleNamespace(cgroup_pids=lambda scope:set(rows)),'scope','/job',proc_read=proc,fd_open=lambda p:p,fd_read=lambda p:p,fd_close=lambda f:None);s.register(20)
 with pytest.raises(RuntimeError,match='UID/job'):s.pids('scope')
 assert 30 not in s.ledger and s.events[-1]['fresh_identity']==fresh

def test_unknown_recheck_exception_does_not_mean_absent():
 rows={20:row(20,10),30:dict(row(30,20),cgroup=[])};calls={}
 def proc(pid):
  calls[pid]=calls.get(pid,0)+1
  if pid==30 and calls[pid]>1:raise PermissionError('unknown actual proc')
  return dict(rows[pid])
 s=m.StageScope(types.SimpleNamespace(cgroup_pids=lambda scope:set(rows)),'scope','/job',proc_read=proc,fd_open=lambda p:p,fd_read=lambda p:p,fd_close=lambda f:None);s.register(20)
 with pytest.raises(RuntimeError,match='UID/job'):s.pids('scope')
 assert s.events[-1]['fresh_error']['type']=='PermissionError' and not s.events[-1]['fresh_proc_absent']

def test_unregistered_pidfd_dies_then_fresh_proc_absent_is_not_owned():
 rows={20:row(20,10),30:row(30,20)};calls={};closed=[]
 def proc(pid):
  calls[pid]=calls.get(pid,0)+1
  if pid==30 and calls[pid]>=3:raise ProcessLookupError('gone')
  return dict(rows[pid])
 s=m.StageScope(types.SimpleNamespace(cgroup_pids=lambda scope:set(rows)),'scope','/job',proc_read=proc,fd_open=lambda p:p,fd_read=lambda p:-1 if p==30 else p,fd_close=closed.append);s.register(20)
 assert s.pids('scope')=={20} and 30 not in s.ledger and closed==[30]

def test_rejection_recheck_keeps_live_bad_row_then_actual_absence_raw():
 rows={20:row(20,10),30:dict(row(30,20),cgroup=[])};calls={}
 def proc(pid):
  calls[pid]=calls.get(pid,0)+1
  if pid==30 and calls[pid]>=3:raise FileNotFoundError('fresh absence after exiting row')
  return dict(rows[pid])
 s=m.StageScope(types.SimpleNamespace(cgroup_pids=lambda scope:set(rows)),'scope','/job',proc_read=proc,fd_open=lambda p:p,fd_read=lambda p:p,fd_close=lambda f:None);s.register(20)
 assert s.pids('scope')=={20};e=next(e for e in s.events if e.get('scope')=='ACTUAL_CENSUS_REJECTION_BEFORE_AUTHORITY')
 assert len(e['fresh_rechecks'])==2 and e['fresh_rechecks'][0]['identity']['cgroup']==[] and e['fresh_rechecks'][1]['proc_absent']
 assert 30 not in s.ledger


def test_live_registered_same_job_subcgroup_migration_is_fatal_and_recorded():
 rows={20:row(20,10)};s=setup(rows);s.register(20);rows[20]['cgroup']=['1:freezer:/job/subgroup']
 with pytest.raises(RuntimeError,match='job/pidfd drift'):s.pids('scope')
 e=next(e for e in s.events if e.get('scope')=='ACTUAL_CENSUS_BAD_ROW_BEFORE_RECHECK')
 assert e['census_identity']['cgroup']==['1:freezer:/job/subgroup'] and e['registered']

def test_sealed_new_row_is_recorded_before_rejection_without_registration():
 s=setup({});s.sealed=True;r=row(30,20)
 with pytest.raises(RuntimeError,match='sealed cleanup'):s.remember(30,r,allow_gone=True)
 assert s.events[0]['census_identity']==r and 30 not in s.ledger

def zombie_scope(rows,ready=True,fdpid=None,mask=1):
 s=setup(rows);s.fd_poll=lambda fd:{'ready':ready,'mask':mask,'events':[[fd,mask]]};s.register(20)
 if fdpid is not None:s.fd_read=lambda fd:fdpid
 return s

def test_registered_kernel_freezer_detach_zombie_excluded_keeps_historical_ledger():
 rows={20:dict(row(20,10),state='S',cgroup=['1:freezer:/job','0::/system.slice/slurmd.service'])};s=zombie_scope(rows)
 rows[20].update(state='Z',cgroup=['1:freezer:/','0::/system.slice/slurmd.service'])
 assert s.pids('scope')==set() and 20 in s.ledger and 20 in s.fds
 event=next(e for e in s.events if e.get('scope')=='REGISTERED_EXIT_PIDFD_READY_AND_Z_OR_PROC_ABSENT_NOT_ACTIVE_NOT_SIGNALLED')
 assert event['pidfd_poll']['mask']==1 and event['actual_pidfd_pid']==20

@pytest.mark.parametrize('kind',['live','uid','start','unified','foreigncg','nopoll','pollerror','wrongfd'])
def test_zombie_exit_never_accepts_live_identity_drift_unknown_or_bad_poll(kind):
 rows={20:dict(row(20,10),state='S',cgroup=['1:freezer:/job','0::/system.slice/slurmd.service'])};s=zombie_scope(rows)
 rows[20].update(state='Z',cgroup=['1:freezer:/','0::/system.slice/slurmd.service'])
 if kind=='live':rows[20]['state']='S'
 elif kind=='uid':rows[20]['uid']=-1
 elif kind=='start':rows[20]['starttime_ticks']=999
 elif kind=='unified':rows[20]['cgroup'][1]='0::/other'
 elif kind=='foreigncg':rows[20]['cgroup'][0]='1:freezer:/otherjob'
 elif kind=='nopoll':s.fd_poll=lambda fd:{'ready':False,'mask':0,'events':[]}
 elif kind=='pollerror':s.fd_poll=lambda fd:{'ready':True,'mask':9,'events':[[fd,9]]}
 elif kind=='wrongfd':s.fd_read=lambda fd:99
 with pytest.raises(RuntimeError):s.pids('scope')
 assert 20 in s.ledger

@pytest.mark.skipif(not hasattr(os,'pidfd_open'),reason='actual Linux unreaped zombie pidfd POLLIN contract')
def test_Linux_real_unreaped_zombie_excluded_with_actual_held_poll_and_stat(tmp_path):
 import subprocess,sys,time,json
 read,write=os.pipe();child=None;s=None
 try:
  child=subprocess.Popen([sys.executable,'-c','import os,sys;os.read(int(sys.argv[1]),1)',str(read)],pass_fds=(read,));os.close(read);read=None
  cg=pathlib.Path('/proc',str(child.pid),'cgroup').read_text().splitlines();relative=next((v.split(':',2)[-1] for v in cg if v.startswith('1:freezer:')),cg[0].split(':',2)[-1])
  s=m.StageScope(types.SimpleNamespace(cgroup_pids=lambda scope:{child.pid}),'scope',relative,event_path=tmp_path/'events.jsonl');s.register(child.pid)
  initial=s.ledger[child.pid];os.write(write,b'x');os.close(write);write=None
  end=time.monotonic()+5
  while time.monotonic()<end:
   actual=s.proc(child.pid)
   if actual['state']=='Z' and s.fd_poll(s.fds[child.pid])['ready']:break
   time.sleep(.01)
  assert actual['state']=='Z' # No child.poll/wait has reaped it.
  assert s.pids('scope')==set() and child.pid in s.ledger
  raw=next(e for e in s.events if e.get('scope')=='REGISTERED_EXIT_PIDFD_READY_AND_Z_OR_PROC_ABSENT_NOT_ACTIVE_NOT_SIGNALLED')
  assert raw['actual_pidfd_pid']==child.pid and raw['fresh_identity']['state']=='Z' and raw['pidfd_poll']['mask'] & 1
  assert raw['fresh_identity']['starttime_ticks']==initial['starttime_ticks'] and raw['fresh_identity']['uid']==initial['uid']
  child.wait(timeout=3);assert s.fd_read(s.fds[child.pid])==-1
 finally:
  for fd in (read,write):
   if fd is not None:os.close(fd)
  if child is not None:
   if child.poll() is None:child.terminate()
   child.wait(timeout=3)
  if s:s.close()

def test_actual3140_recorded_Z_detach_replay_requires_new_poll_proof(monkeypatch):
 import json
 fixture=json.loads(pathlib.Path(__file__).with_name('actual3140-exit.fixture.json').read_text());old=fixture['registered_identity'];event=fixture['recorded_event'];pid=old['pid']
 assert event['actual_pidfd_pid'] is None and event['census_identity']['state']=='Z'
 monkeypatch.setattr(m.os,'getuid',lambda:old['uid']);rows={pid:old}
 relative=next(v.split(':',2)[-1] for v in old['cgroup'] if v.startswith('1:freezer:'))
 s=m.StageScope(types.SimpleNamespace(cgroup_pids=lambda scope:set(rows)),'scope',relative,proc_read=lambda p:dict(rows[p]),fd_open=lambda p:p,fd_read=lambda p:p,fd_close=lambda f:None);s.register(pid);rows[pid]=event['census_identity']
 s.fd_poll=lambda fd:{'ready':True,'mask':1,'events':[[fd,1]],'scope':'CPU_FIXTURE_POLL_NOT_HISTORICAL3140_PROOF'}
 assert s.pids('scope')==set() and pid in s.ledger
 assert s.events[-2]['pidfd_poll']['scope']=='CPU_FIXTURE_POLL_NOT_HISTORICAL3140_PROOF'


def test_unregistered_zombie_never_receives_held_ownership():
 rows={20:row(20,10),30:dict(row(30,20),state='Z')};s=setup(rows);s.register(20)
 with pytest.raises(RuntimeError,match='Unregistered zombie'):s.pids('scope')
 assert 30 not in s.ledger and 30 not in s.fds
