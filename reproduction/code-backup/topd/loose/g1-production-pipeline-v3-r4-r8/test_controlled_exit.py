import copy,importlib.util,json,os,pathlib,sys,types,time
import pytest
sys.path.insert(0,str(pathlib.Path(__file__).parent))
import controlled_exit as x

def require(value,message):
 if not value:raise RuntimeError(message)
def actual_runtime_functions():
 import ast,math,hashlib
 path=pathlib.Path(__file__).parents[1]/'qwen25-32b-window-safety-candidate-v4/runtime_safety.py'
 assert hashlib.sha256(path.read_bytes()).hexdigest()=='9e41464d49054a1765fed29f5839f9fd8c568f0040aedbf6cf36b3c096adb3c6'
 tree=ast.parse(path.read_text());namespace={'require':require,'math':math,'FRACTION':.4,'RESERVE_MIB':2048.}
 exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('sm_aggregate_bounds','accounting_gate')],type_ignores=[]),str(path),'exec'),namespace)
 return namespace

def fixture(monkeypatch):
 root=pathlib.Path(__file__).parent
 fixture_path=root/'actual3120-exit.fixture.json'
 assert __import__('hashlib').sha256(fixture_path.read_bytes()).hexdigest()=='64dcc6d1ffe350cba4292c22a4f86ba7df5bdb58b421ee96f2f0330595056743'
 historical=json.loads(fixture_path.read_text());cap=historical['failed_guard_row']['captures'][0];bad=copy.deepcopy(cap['raw_snapshot'])
 # The historical exact snapshots prove provenance; Z below is explicitly synthetic.
 ready=copy.deepcopy(historical['ready_guard_row']['captures'][-1]['raw_snapshot'])
 stage=2886829;service=2884165;uuids={g['uuid'] for g in ready['gpus']};protected={a['pid'] for a in ready['applications'] if a['pid']<2000000}
 base=copy.deepcopy(ready);base['applications']=[a for a in ready['applications'] if a['pid'] in protected];base['identities']={str(p):ready['identities'][str(p)] for p in protected}
 runtime=dict(baseline=base,retriever_identity=ready['identities'][str(service)],cgroup='/slurm_h100-11/uid_0/job_3120/step_batch')
 current=[ready];states={pid:dict(pid=pid,starttime_ticks=i['starttime_ticks'],uid=0,state='S',ppid=stage if pid not in (stage,service) and pid not in protected else 1) for pid,i in [(int(p),v) for p,v in ready['identities'].items()]}
 fd_map={}
 def openfd(pid,flags):
  fd=os.open(os.devnull,os.O_RDONLY);fd_map[fd]=pid;return fd
 safety=types.SimpleNamespace(pid_identity=lambda pid:current[0]['identities'][str(pid)],accounting_gate=actual_runtime_functions()['accounting_gate'])
 monkeypatch.setattr(x.os,'getuid',lambda:0)
 ledger=x.ReadyLedger(safety,runtime,uuids,ready,proc_read=lambda pid:dict(states[pid]),fd_open=openfd,fd_read=lambda fd:fd_map[fd],stage_pid=stage);ledger.active=True
 return ledger,safety,current,states,bad,ready,cap

def test_3120_exact_raw_membership_replayed_but_nonZ_is_not_qualified(monkeypatch):
 ledger,safety,current,states,bad,ready,cap=fixture(monkeypatch)
 try:
  current[0]=bad
  with pytest.raises(RuntimeError,match='not instantaneous Z/X'):ledger.validate(bad)
  assert len(ledger.exiting_pids)==8 and all(cap['uids'][str(p)]==ledger.rows[p]['uid'] for p in ledger.rows)
 finally:ledger.close()

def test_same_historical_identities_with_new_instantaneous_Z_fixture_qualify_only(monkeypatch):
 ledger,safety,current,states,bad,ready,_=fixture(monkeypatch)
 try:
  current[0]=bad
  for p in ledger.exiting_pids:states[p]['state']='Z';states[p]['ppid']=1
  foreign,own,retired,metadata=ledger.validate(bad)
  assert len(retired)==7 and len(own)==16 and len(foreign)==8
 finally:ledger.close()

@pytest.mark.parametrize('drift',['reuse','uid','newpid','newuuid','pidfd','foreign_identity','foreign_memory','nonZ_adopted'])
def test_exit_cannot_qualify_identity_or_foreign_breach(monkeypatch,drift):
 ledger,safety,current,states,bad,ready,_=fixture(monkeypatch)
 try:
  current[0]=bad
  for p in ledger.exiting_pids:states[p]['state']='Z';states[p]['ppid']=1
  p=next(p for p in ledger.exiting_pids if p!=2886829)
  if drift=='reuse':states[p]['starttime_ticks']+=1
  if drift=='uid':states[p]['uid']=123
  if drift=='newpid':bad['applications'][2]['pid']=999999;bad['identities']['999999']=dict(bad['identities'][str(p)],pid=999999);states[999999]=dict(states[p],pid=999999)
  if drift=='newuuid':next(a for a in bad['applications'] if a['pid']==p)['uuid']='GPU-UNLEASED'
  if drift=='pidfd':ledger.fd_read=lambda fd:-1
  if drift=='foreign_identity':bad['identities']['1042865']['cgroup']=['0::/different']
  if drift=='foreign_memory':bad['applications'][0]['used_mib']=16000
  if drift=='nonZ_adopted':states[p]['state']='S'
  with pytest.raises((RuntimeError,KeyError)):ledger.validate(bad)
 finally:ledger.close()

def guard_fixture(monkeypatch):
 ledger,safety,current,states,bad,ready,cap=fixture(monkeypatch)
 for p in ledger.exiting_pids:states[p]['state']='Z';states[p]['ppid']=1
 gone=copy.deepcopy(bad);gone['applications']=[a for a in bad['applications'] if a['pid'] not in ledger.exiting_pids];gone['identities']={p:i for p,i in bad['identities'].items() if int(p) not in ledger.exiting_pids}
 rows=[];calls=[]
 guard=types.SimpleNamespace(_write=lambda r:rows.append(copy.deepcopy(r)),lease_reader=lambda:ledger.uuids,max_attempts=6,retry_call_seconds=30)
 def full(runtime,pre_model):
  calls.append(1);current[0]=bad if len(calls)==1 else gone
  guard._write(dict(attempt=1,accepted=len(calls)>1,captures=[dict(raw_snapshot=copy.deepcopy(current[0]),uids={p:states[int(p)]['uid'] for p in current[0]['identities']},uid_errors={})]))
  if len(calls)==1:raise RuntimeError(x.ERROR)
  return {'original_full_fresh':True}
 guard.active_gate=full;safety.nvml_snapshot=lambda:copy.deepcopy(current[0]);safety.collect_process_windows=lambda *a:({},int(time.time()*1e6));safety.sm_aggregate_bounds=actual_runtime_functions()['sm_aggregate_bounds']
 return ledger,safety,guard,calls,rows,current,bad

def test_only_new_original_full_gate_after_actual_records_gone_can_accept(monkeypatch):
 ledger,safety,guard,calls,rows,current,bad=guard_fixture(monkeypatch);evidence=[];original=guard._write
 try:
  assert x.exit_gate(guard,safety,ledger.runtime,ledger,evidence.append,pause=0)=={'original_full_fresh':True}
  assert len(calls)==2 and rows[0]['accepted'] is False and evidence[-1]['scope'].endswith('RECORDS_GONE') and guard._write is original
 finally:ledger.close()

def test_held_FDs_change_and_other_errors_never_exit_retry(monkeypatch):
 ledger,safety,guard,calls,rows,current,bad=guard_fixture(monkeypatch)
 try:
  guard.lease_reader=lambda:set()
  with pytest.raises(RuntimeError,match='leases changed'):x.exit_gate(guard,safety,ledger.runtime,ledger,lambda r:None,pause=0)
  assert not calls
 finally:ledger.close()

def test_unknown_stale_or_foreign_sample_not_retryable(monkeypatch):
 ledger,safety,guard,calls,rows,current,bad=guard_fixture(monkeypatch)
 try:
  safety.sm_aggregate_bounds=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('Stale/future process sample'))
  with pytest.raises(RuntimeError,match='Stale'):x.exit_gate(guard,safety,ledger.runtime,ledger,lambda r:None,pause=0)
  assert len(calls)==1
 finally:ledger.close()


def test_actual_foreign_SM20_during_exit_fails_not_nextfresh(monkeypatch):
 ledger,safety,guard,calls,rows,current,bad=guard_fixture(monkeypatch)
 try:
  uid=bad['gpus'][0]['uuid'];bad['gpus'][0]['sm']=90
  def windows(snapshot,foreign):
   now=int(time.time()*1e6);foreign_pid=next(p for u,p in foreign if u==uid);own_pid=next(a['pid'] for a in snapshot['applications'] if a['uuid']==uid and (uid,a['pid']) not in foreign)
   return {uid:dict(status='SUCCESS',start_us=now-1100000,end_us=now,samples=[dict(pid=foreign_pid,sm=20,timestamp_us=now-1000),dict(pid=own_pid,sm=70,timestamp_us=now-1000)])},now
  safety.collect_process_windows=windows
  with pytest.raises(RuntimeError,match='foreign SM exceeds15'):x.exit_gate(guard,safety,ledger.runtime,ledger,lambda r:None,pause=0)
  assert len(calls)==1
 finally:ledger.close()

def test_original6_attempt_budget_never_expanded(monkeypatch):
 ledger,safety,guard,calls,rows,current,bad=guard_fixture(monkeypatch)
 try:
  def always(runtime,pre_model):
   calls.append(1);current[0]=bad
   guard._write(dict(attempt=1,accepted=False,captures=[dict(raw_snapshot=copy.deepcopy(bad),uids={p:ledger.proc_read(int(p))['uid'] for p in bad['identities']},uid_errors={})]));raise RuntimeError(x.ERROR)
  guard.active_gate=always
  with pytest.raises(RuntimeError,match='original6'):x.exit_gate(guard,safety,ledger.runtime,ledger,lambda r:None,pause=0)
  assert len(calls)==6 and guard.max_attempts==6 and guard.retry_call_seconds==30
 finally:ledger.close()

@pytest.mark.skipif(sys.platform!='linux',reason='actual Linux pidfd/proc zombie native contract')
def test_real_pidfd_and_instantaneous_zombie_identity():
 readfd,writefd=os.pipe();pid=os.fork()
 if pid==0:
  os.close(writefd);os.read(readfd,1);os._exit(0)
 os.close(readfd);fd=os.pidfd_open(pid,0)
 try:
  initial=x.proc(pid);assert x.fd_pid(fd)==pid and initial['uid']==os.getuid()
  os.write(writefd,b'x');os.close(writefd);writefd=-1
  deadline=time.monotonic()+2
  while time.monotonic()<deadline:
   terminal=x.proc(pid)
   if terminal['state']=='Z':break
   time.sleep(.01)
  assert terminal['state']=='Z' and terminal['starttime_ticks']==initial['starttime_ticks'] and x.fd_pid(fd)==pid
 finally:
  if writefd>=0:os.close(writefd)
  os.close(fd);os.waitpid(pid,0)


def test_captured_old_job_identity_new_actual_Z_is_not_retroactive_error(monkeypatch):
 ledger,safety,current,states,bad,ready,cap=fixture(monkeypatch)
 try:
  current[0]=bad
  for p in ledger.exiting_pids:states[p]['state']='Z';states[p]['ppid']=1
  _,_,retired,_=ledger.validate(ready,captured_uids=cap['uids'])
  assert len(retired)==7 and all(r['captured_identity']!=r['actual_identity'] for r in retired)
 finally:ledger.close()

def test_precise_identity_changed_exit_error_requires_next_original_full_gate(monkeypatch):
 ledger,safety,guard,calls,rows,current,bad=guard_fixture(monkeypatch);full=guard.active_gate
 def changed(runtime,pre_model):
  try:return full(runtime,pre_model)
  except RuntimeError as e:
   if str(e)==x.ERROR:raise RuntimeError(x.IDENTITY_ERROR)
   raise
 guard.active_gate=changed
 try:
  assert x.exit_gate(guard,safety,ledger.runtime,ledger,lambda r:None,pause=0)['original_full_fresh'] and len(calls)==2
 finally:ledger.close()

def test_actual_dead_pidfd_proc_absent_NVML_tail_only_qualifies_nextfresh(monkeypatch):
 ledger,safety,guard,calls,rows,current,bad=guard_fixture(monkeypatch);pid=next(p for p in ledger.exiting_pids if p!=2886829);original_proc=ledger.proc_read;originalfd=ledger.fd_read
 try:
  def missing(p):
   if p==pid:raise FileNotFoundError(2,'gone',f'/proc/{p}/stat')
   return original_proc(p)
  ledger.proc_read=missing;ledger.fd_read=lambda fd:-1 if fd==ledger.fds[pid] else originalfd(fd)
  raw=dict(gpus=bad['gpus'],applications=bad['applications'],**query_times());monkeypatch.setattr(x,'missing_proc_diagnostic',lambda ledger:copy.deepcopy(raw))
  full=guard.active_gate
  def vanished(runtime,pre_model):
   if not calls:
    calls.append(1);current[0]=bad;guard._write(dict(attempt=1,accepted=False,captures=[]));raise FileNotFoundError(2,'gone',f'/proc/{pid}/stat')
   return full(runtime,pre_model)
  guard.active_gate=vanished;evidence=[]
  assert x.exit_gate(guard,safety,ledger.runtime,ledger,evidence.append,pause=0)['original_full_fresh'] and len(calls)==2
  first=evidence[0];assert first['scope'].endswith('NOT_ACCEPTED') and first['first_qualification']['unresolved'][0]['scope'].endswith('NOT_CURRENT_MEASUREMENT')
 finally:ledger.close()

@pytest.mark.parametrize('violation',['highSM','newPID','aliveReuse'])
def test_dead_pidfd_UNKNOWN_never_hides_unknown_or_high_SM(monkeypatch,violation):
 ledger,safety,guard,calls,rows,current,bad=guard_fixture(monkeypatch);pid=next(p for p in ledger.exiting_pids if p!=2886829);original=ledger.proc_read
 try:
  ledger.proc_read=lambda p:(_ for _ in ()).throw(FileNotFoundError()) if p==pid else original(p)
  ledger.fd_read=lambda fd:-1 if fd==ledger.fds[pid] else next(p for p,f in ledger.fds.items() if f==fd)
  current[0]=bad;raw=dict(gpus=copy.deepcopy(bad['gpus']),applications=copy.deepcopy(bad['applications']),**query_times())
  if violation=='highSM':raw['gpus'][0]['sm']=50
  elif violation=='newPID':raw['applications'][2]['pid']=9999
  else:ledger.proc_read=lambda p:dict(original(p),starttime_ticks=0) if p==pid else original(p)
  with pytest.raises((RuntimeError,KeyError)):x.qualify_missing_proc(ledger,raw)
 finally:ledger.close()


def query_times():
 now=int(time.time()*1e6)
 return {k:now for k in ('gpu_query_started_us','gpu_query_completed_us','app_query_started_us','app_query_completed_us')}

@pytest.mark.parametrize('failure',['slow_apps','future','unordered'])
def test_missing_proc_raw_diagnostic_time_is_not_fresh_by_assertion(monkeypatch,failure):
 raw=query_times();now=int(time.time()*1e6)
 monkeypatch.setattr(x.time,'time',lambda:now/1e6)
 if failure=='slow_apps':raw['gpu_query_started_us']=raw['gpu_query_completed_us']=now-10_000_000
 elif failure=='future':raw['app_query_completed_us']=now+1
 else:raw['gpu_query_completed_us']=now+1;raw['app_query_started_us']=now-1
 with pytest.raises(RuntimeError,match='stale/future or unordered'):x.diagnostic_fresh(raw)

def test_raw_queries_record_independent_real_start_completion(monkeypatch):
 clock=iter([10.,10.1,10.2,10.3]);monkeypatch.setattr(x.time,'time',lambda:next(clock))
 calls=[]
 def read(args,**kw):
  calls.append(args)
  return '0, GPU-x, 2, 100, 0\n' if len(calls)==1 else 'GPU-x, 42, 2\n'
 monkeypatch.setattr(x.subprocess,'check_output',read)
 raw=x.missing_proc_diagnostic(None)
 assert [raw[k] for k in ('gpu_query_started_us','gpu_query_completed_us','app_query_started_us','app_query_completed_us')]==[10000000,10100000,10200000,10300000]
 assert raw['applications'][0]['pid']==42 and raw['gpus'][0]['sm']==0
