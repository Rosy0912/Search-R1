"""READY-ledger exit qualification, never a numeric acceptance of departed PIDs."""
import copy,json,math,os,pathlib,time,subprocess
ERROR='Unknown/foreign GPU process or unleased UUID'
IDENTITY_ERROR='Process identities changed during SM window'

def need(value,message):
 if not value:raise RuntimeError(message)
def member(identity,relative):return any(r.split(':',2)[-1]==relative or r.split(':',2)[-1].startswith(relative+'/') for r in identity['cgroup'])
def proc(pid):
 root=pathlib.Path('/proc',str(pid));fields=(root/'stat').read_text().rsplit(')',1)[1].split()
 return dict(pid=pid,starttime_ticks=int(fields[19]),state=fields[0],ppid=int(fields[1]),uid=root.stat().st_uid)
def fd_pid(fd):
 return int(next(s.split(':',1)[1].strip() for s in pathlib.Path('/proc/self/fdinfo',str(fd)).read_text().splitlines() if s.startswith('Pid:')))

class ReadyLedger:
 def __init__(self,safety,runtime,uuids,snapshot,*,proc_read=proc,fd_open=None,fd_read=fd_pid,stage_pid=None):
  self.safety,self.runtime,self.uuids=safety,runtime,set(uuids);self.proc_read,self.fd_read=proc_read,fd_read;self.rows={};self.fds={};self.foreign_uids={};self.active=False
  fd_open=fd_open or os.pidfd_open
  try:
   protected={(p['uuid'],p['pid']) for p in runtime['baseline']['applications'] if p['uuid'] in self.uuids}
   safety.accounting_gate(snapshot,runtime['baseline'],self.uuids,runtime['retriever_identity'],runtime['cgroup'],{},False,validate_sm=False)
   for app in snapshot['applications']:
    pid=app['pid'];ident=snapshot['identities'][str(pid)];meta=proc_read(pid)
    need(meta['pid']==pid and meta['starttime_ticks']==ident['starttime_ticks'],'READY actual proc identity drift')
    if (app['uuid'],pid) in protected:self.foreign_uids[pid]=meta['uid'];continue
    need(meta['uid']==os.getuid() and meta['state'] not in ('Z','X') and member(ident,runtime['cgroup']) and safety.pid_identity(pid)==ident,'READY actual live owned PID/UID/job drift')
    if pid==runtime['retriever_identity']['pid']:continue
    if pid not in self.rows:
     fd=fd_open(pid,0);self.fds[pid]=fd;need(fd_read(fd)==pid,'READY actual pidfd drift')
     self.rows[pid]=dict(identity=copy.deepcopy(ident),uid=meta['uid'],uuids=[],ready_proc=meta)
    self.rows[pid]['uuids'].append(app['uuid'])
   def descendant(pid):
    seen=set()
    while pid>1 and pid not in seen:
     if pid==stage_pid:return True
     seen.add(pid);pid=proc_read(pid)['ppid']
    return False
   self.exiting_pids=set(self.rows) if stage_pid is None else {pid for pid in self.rows if descendant(pid)}
   need(self.exiting_pids and {u for pid,row in self.rows.items() if pid in self.exiting_pids for u in row['uuids']}==self.uuids,'READY actual stage model all8 PID/UUID ledger absent')
  except BaseException:self.close();raise
 def close(self):
  for fd in self.fds.values():
   try:os.close(fd)
   except OSError:pass
  self.fds.clear()
 def evidence(self):return dict(scope='ACTUAL_READY_PIDFD_IDENTITY_UUID_LEDGER_NOT_UPDATE_PROOF',rows=self.rows,exiting_pids=sorted(self.exiting_pids),foreign_uids=self.foreign_uids)
 def validate(self,snapshot,*,captured_uids=None):
  protected={(p['uuid'],p['pid']) for p in self.runtime['baseline']['applications'] if p['uuid'] in self.uuids}
  apps=snapshot['applications'];gpus={g['uuid']:g for g in snapshot['gpus'] if g['uuid'] in self.uuids}
  need(len(gpus)==8 and all(all(math.isfinite(g[k]) for k in ('sm','used_mib','total_mib')) and 0<=g['sm']<=100 and 0<=g['used_mib']<=g['total_mib'] and g['total_mib']>0 for g in gpus.values()),'Unknown allocated exit GPU measurements')
  need(all(math.isfinite(a['used_mib']) and a['used_mib']>=0 for a in apps),'Unknown exit process memory')
  need(protected<={(a['uuid'],a['pid']) for a in apps},'Protected baseline PID/UUID disappeared')
  own=set();retired=[];meta_rows={}
  service=self.runtime['retriever_identity']
  for app in apps:
   pid=app['pid'];uid=app['uuid'];ident=snapshot['identities'][str(pid)]
   try:meta=self.proc_read(pid);actual_identity=self.safety.pid_identity(pid)
   except FileNotFoundError:
    need(captured_uids is not None and pid in self.exiting_pids and self.fd_read(self.fds[pid])==-1,'EXIT missing proc has no actual READY pidfd exit proof')
    old=self.rows[pid];need(ident['starttime_ticks']==old['identity']['starttime_ticks'] and captured_uids[str(pid)]==old['uid'],'Captured exited PID/start/UID mismatch')
    current_gpu=self.safety.nvml_snapshot();need(not any(a['pid']==pid for a in current_gpu['applications']),'Exited proc still has unknown NVML record')
    meta=dict(pid=pid,starttime_ticks=old['identity']['starttime_ticks'],uid=old['uid'],state='PROC_ABSENT_PIDFD_EXITED_NVML_GONE',ppid=None);actual_identity=ident
   meta_rows[pid]=meta
   need(meta['pid']==pid and meta['starttime_ticks']==ident['starttime_ticks'],'EXIT instantaneous PID/start identity drift')
   if captured_uids is None:need(actual_identity==ident,'EXIT current snapshot identity drift')
   else:need(captured_uids[str(pid)]==meta['uid'],'Captured/current EXIT UID differs')
   if (uid,pid) in protected:
    need(ident==actual_identity==self.runtime['baseline']['identities'][str(pid)] and meta['uid']==self.foreign_uids[pid],'Protected foreign PID/UID changed');continue
   need(uid in self.uuids and meta['uid']==os.getuid(),'Unknown EXIT PID/UID or unleased GPU')
   if pid==service['pid']:need(ident==actual_identity==service,'Retriever actual identity changed')
   else:
    need(pid in self.rows and uid in self.rows[pid]['uuids'],'Unknown model EXIT PID/UUID')
    old=self.rows[pid];    # Captured membership is immutable provenance; current Z/absence is separate evidence.
    absent=meta['state']=='PROC_ABSENT_PIDFD_EXITED_NVML_GONE'
    need(ident['starttime_ticks']==old['identity']['starttime_ticks'] and actual_identity['starttime_ticks']==old['identity']['starttime_ticks'] and meta['uid']==old['uid'] and (absent or self.fd_read(self.fds[pid])==pid),'READY PID reused/UID/pidfd changed')
    if ident!=old['identity'] or actual_identity!=old['identity'] or absent:
     need(pid in self.exiting_pids and (absent or (meta['state'] in ('Z','X') and not member(actual_identity,self.runtime['cgroup']))),'Out-of-job EXIT PID is not instantaneous Z/X')
     need(ident==old['identity'] or not member(ident,self.runtime['cgroup']),'Captured unexplained job-scope change')
     retired.append(dict(pid=pid,uuid=uid,captured_identity=ident,actual_proc=meta,actual_identity=actual_identity))
   own.add((uid,pid))
  need({u for u,p in own if p==service['pid']}==self.uuids,'Retriever all8 actual residency changed')
  for uid,g in gpus.items():need(sum(a['used_mib'] for a in apps if (a['uuid'],a['pid']) in protected and a['uuid']==uid)/g['total_mib']<=.15,'Foreign memory exceeds15% during EXIT')
  return protected,own,retired,meta_rows

def missing_proc_diagnostic(ledger):
 """Independent raw read; missing PIDs stay unresolved, never fabricated identities."""
 q=['nvidia-smi','--format=csv,noheader,nounits']
 gpu_query_started_us=int(time.time()*1e6)
 gpu_text=subprocess.check_output(q+['--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu'],text=True,timeout=10)
 gpu_query_completed_us=int(time.time()*1e6);app_query_started_us=int(time.time()*1e6)
 app_text=subprocess.check_output(q+['--query-compute-apps=gpu_uuid,pid,used_gpu_memory'],text=True,timeout=10)
 app_query_completed_us=int(time.time()*1e6)
 gpus=[];apps=[]
 for line in gpu_text.splitlines():
  i,u,m,t,s=[v.strip() for v in line.split(',')];gpus.append(dict(index=i,uuid=u,used_mib=float(m),total_mib=float(t),sm=float(s)))
 for line in app_text.splitlines():
  u,p,m=[v.strip() for v in line.split(',')];apps.append(dict(uuid=u,pid=int(p),used_mib=float(m)))
 return dict(gpus=gpus,applications=apps,raw_gpu_stdout=gpu_text,raw_application_stdout=app_text,gpu_query_started_us=gpu_query_started_us,gpu_query_completed_us=gpu_query_completed_us,app_query_started_us=app_query_started_us,app_query_completed_us=app_query_completed_us)

def diagnostic_fresh(snapshot):
 now=int(time.time()*1e6)
 stamps=[snapshot[k] for k in ('gpu_query_started_us','gpu_query_completed_us','app_query_started_us','app_query_completed_us')]
 need(all(isinstance(t,int) and not isinstance(t,bool) for t in stamps) and stamps==sorted(stamps) and 0<=now-stamps[0]<=1_500_000 and stamps[-1]<=now,'Absent-proc raw GPU/application diagnostic stale/future or unordered')
 return dict(evaluation_now_us=now,oldest_query_age_us=now-stamps[0],scope='ACTUAL_QUERY_START_TO_CURRENT_TIME_MAX1_5_SECONDS')

def qualify_missing_proc(ledger,snapshot,*,require_unresolved=True):
 diagnostic_fresh(snapshot)
 foreign={(a['uuid'],a['pid']) for a in ledger.runtime['baseline']['applications'] if a['uuid'] in ledger.uuids}
 need(foreign<={(a['uuid'],a['pid']) for a in snapshot['applications']},'Protected baseline disappeared during absent-proc EXIT')
 gpus={g['uuid']:g for g in snapshot['gpus'] if g['uuid'] in ledger.uuids}
 need(len(gpus)==8 and all(all(math.isfinite(g[k]) for k in ('sm','used_mib','total_mib')) and g['total_mib']>0 and 0<=g['used_mib']<=g['total_mib'] and 0<=g['sm']<=15 for g in gpus.values()),'Absent-proc EXIT lacks actual low total SM qualification; remains UNKNOWN')
 unresolved=[];service=ledger.runtime['retriever_identity'];service_keys=set();alive={}
 for app in snapshot['applications']:
  pid,uid=app['pid'],app['uuid'];need(uid in ledger.uuids and math.isfinite(app['used_mib']) and app['used_mib']>=0,'Unknown absent-proc EXIT application')
  try:meta=ledger.proc_read(pid);ident=ledger.safety.pid_identity(pid)
  except FileNotFoundError:
   need(pid in ledger.exiting_pids and uid in ledger.rows[pid]['uuids'] and ledger.fd_read(ledger.fds[pid])==-1,'Missing GPU PID has no actual registered dead pidfd')
   unresolved.append(dict(pid=pid,uuid=uid,pidfd_pid=-1,proc_absent=True,registered_identity=ledger.rows[pid]['identity'],registered_uid=ledger.rows[pid]['uid'],scope='UID_IDENTITY_PROVENANCE_ONLY_NOT_CURRENT_MEASUREMENT'));continue
  need(meta['pid']==pid and meta['starttime_ticks']==ident['starttime_ticks'],'Actual missing-proc diagnostic identity inconsistent')
  alive[pid]=dict(actual_proc=meta,actual_identity=ident)
  if (uid,pid) in foreign:need(ident==ledger.runtime['baseline']['identities'][str(pid)] and meta['uid']==ledger.foreign_uids[pid],'Protected foreign identity changed')
  elif pid==service['pid']:need(ident==service and meta['uid']==os.getuid(),'Retriever changed');service_keys.add(uid)
  else:
   need(pid in ledger.rows and uid in ledger.rows[pid]['uuids'] and meta['uid']==ledger.rows[pid]['uid'] and ident['starttime_ticks']==ledger.rows[pid]['identity']['starttime_ticks'],'Unknown/reused PID in absent-proc EXIT')
   need(ledger.fd_read(ledger.fds[pid])==pid and (ident==ledger.rows[pid]['identity'] or (pid in ledger.exiting_pids and meta['state'] in ('Z','X') and not member(ident,ledger.runtime['cgroup']))),'Non-exit identity changed')
 need((unresolved or not require_unresolved) and service_keys==ledger.uuids,'Actual absent READY model PID and stable all8 retriever required')
 for uid,g in gpus.items():need(sum(a['used_mib'] for a in snapshot['applications'] if (a['uuid'],a['pid']) in foreign and a['uuid']==uid)/g['total_mib']<=.15,'Foreign GPU memory exceeds15% in absent-proc EXIT')
 freshness=diagnostic_fresh(snapshot)
 return dict(freshness=freshness,scope='REGISTERED_DEAD_PIDFD_NVML_TAIL_UNKNOWN_REQUIRES_NEXT_ORIGINAL_FULL_GATE',unresolved=unresolved,alive=alive,actual_low_device_sm=[dict(uuid=u,measured_total_sm=g['sm']) for u,g in gpus.items()])

def exit_gate(guard,safety,runtime,ledger,raw_writer,*,seconds=30,max_attempts=6,pause=.25):
 need(ledger is not None and ledger.active,'Explicit controlled EXIT ledger absent')
 need(0<seconds<=30 and 1<=max_attempts<=6,'Exit bound exceeds original budget')
 start=time.monotonic();deadline=start+seconds;history=[];used=0
 original_write=guard._write;old_attempts,old_seconds=guard.max_attempts,guard.retry_call_seconds
 try:
  while used<max_attempts:
   need(time.monotonic()<deadline,'Controlled EXIT original30s budget exhausted')
   need(set(guard.lease_reader())==ledger.uuids,'Actual held8 leases changed at EXIT')
   records=[]
   def observe(record):
    result=original_write(record);records.append(copy.deepcopy(record));return result
   guard._write=observe;guard.max_attempts=max_attempts-used;guard.retry_call_seconds=min(old_seconds,deadline-time.monotonic())
   try:result=guard.active_gate(runtime,pre_model=False);error=None
   except (RuntimeError,FileNotFoundError) as exc:result=None;error=exc
   finally:guard._write=original_write
   used+=max(1,max((r.get('attempt',1) for r in records),default=1))
   need(time.monotonic()<deadline,'Controlled EXIT complete measurement exceeded original30s budget')
   if isinstance(error,FileNotFoundError):
    filename=pathlib.Path(error.filename or '')
    need(filename.name in ('stat','cgroup') and filename.parent.parent==pathlib.Path('/proc') and filename.parent.name.isdecimal() and int(filename.parent.name) in ledger.exiting_pids,'Unrelated missing file is not a controlled EXIT')
    first=missing_proc_diagnostic(ledger);qualification=qualify_missing_proc(ledger,first)
    second=missing_proc_diagnostic(ledger);second_qualification=qualify_missing_proc(ledger,second,require_unresolved=False)
    need(all(second_qualification["alive"][p]["actual_identity"]==v["actual_identity"] and second_qualification["alive"][p]["actual_proc"]["uid"]==v["actual_proc"]["uid"] for p,v in qualification["alive"].items() if p in second_qualification["alive"]),"Absent-proc qualification live identity changed")
    need(set(guard.lease_reader())==ledger.uuids,'Absent-proc EXIT held8 leases changed')
    need(time.monotonic()<deadline,'Absent-proc EXIT exceeded original30s budget')
    raw_writer(dict(scope='ORIGINAL_FULL_GUARD_FILE_NOT_FOUND_UNKNOWN_NOT_ACCEPTED',original_guard_rows=records,original_error=str(error),first=first,first_qualification=qualification,second=second,second_qualification=second_qualification))
    time.sleep(pause);continue
   if error is not None:need(str(error) in (ERROR,IDENTITY_ERROR),'Non-exit full guardian failure: '+str(error))
   # Every original captured identity is checked without rewriting its cgroup/value.
   captures=[c for r in records for c in r.get('captures',[])]
   need(captures,'Original full guardian did not capture actual EXIT identities')
   any_retired=False
   for cap in captures:
    need(not cap.get('uid_errors'),'Original EXIT UID sidecar unknown')
    _,_,retired,metas=ledger.validate(cap['raw_snapshot'],captured_uids=cap['uids'])
    need(all(cap['uids'][str(pid)]==meta['uid'] for pid,meta in metas.items()),'Captured/current EXIT UID differs')
    any_retired=any_retired or bool(retired)
   if error is not None:
    need(any_retired,'Original ownership rejection is not known READY Z/X EXIT')
    # Keep protecting foreign15 during UNKNOWN wait using complete current raw bounds.
    current=safety.nvml_snapshot();foreign,own,retired,metas=ledger.validate(current)
    windows,query=safety.collect_process_windows(current,foreign)
    raw=dict(scope='REJECTED_EXIT_UNKNOWN_NOT_ACCEPTED',original_guard_rows=records,current_snapshot=current,actual_proc=metas,windows=windows,query_completed_us=query,retired=retired)
    raw_writer(raw)
    bounds,evidence=safety.sm_aggregate_bounds(current,foreign,own,windows,query)
    final=safety.nvml_snapshot();f,o,retired_final,metas_final=ledger.validate(final)
    need(f==foreign and o<=own and all(pid in ledger.exiting_pids for uid,pid in own-o),'EXIT qualification unknown identity changed; no acceptance')
    need(all(final['identities'][pid]==ident for pid,ident in current['identities'].items() if pid in final['identities']),'EXIT qualification existing PID changed; no acceptance')
    final_bounds,final_evidence=safety.sm_aggregate_bounds(final,foreign,own,windows,int(time.time()*1e6),actual_query_completed_us=query)
    need(all(max(bounds[u],final_bounds[u])<=15 for u,p in foreign),'Observed foreign SM exceeds15% at EXIT')
    need(set(guard.lease_reader())==ledger.uuids,'EXIT held8 leases changed after qualification')
    history.append(dict(scope='QUALIFIED_KNOWN_READY_Z_EXIT_REQUIRES_NEXT_FULL_FRESH_NOT_NUMERIC_ACCEPTANCE',used_attempts=used,initial_evidence=evidence,final_evidence=final_evidence));raw_writer(history[-1])
   else:
    latest=captures[-1]['raw_snapshot'];remaining=[a for a in latest['applications'] if a['pid'] in ledger.exiting_pids]
    if not remaining:
     raw_writer(dict(scope='ORIGINAL_FULL_FRESH_EXIT_GATE_PASS_ACTUAL_READY_GPU_RECORDS_GONE',actual=result,history=history,used_attempts=used));return result
    history.append(dict(scope='ORIGINAL_FULL_GUARD_PASS_BUT_MODEL_GPU_EXIT_PENDING_NOT_EXIT_ACCEPTANCE',remaining=remaining));raw_writer(history[-1])
   time.sleep(pause)
  raise RuntimeError('Controlled EXIT original6 attempts exhausted')
 finally:guard._write=original_write;guard.max_attempts=old_attempts;guard.retry_call_seconds=old_seconds
