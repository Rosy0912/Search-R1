"""Restrict original cleanup to the actual spawned stage tree, not same-job siblings."""
import ast,hashlib,os,pathlib,time,json

def require(ok,message):
 if not ok:raise RuntimeError(message)
class EvidenceEvents(list):
 def __init__(self,path=None):
  super().__init__();self.path=pathlib.Path(path) if path else None
  if self.path:
   fd=os.open(self.path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
   try:os.fsync(fd)
   finally:os.close(fd)
   d=os.open(self.path.parent,os.O_RDONLY)
   try:os.fsync(d)
   finally:os.close(d)
 def append(self,value):
  if self.path:
   fd=os.open(self.path,os.O_APPEND|os.O_WRONLY|os.O_NOFOLLOW)
   try:
    raw=(json.dumps(value,allow_nan=False,sort_keys=True)+'\n').encode();offset=0
    while offset<len(raw):offset+=os.write(fd,raw[offset:])
    os.fsync(fd)
   finally:os.close(fd)
  super().append(value)
class StageScope:
 def __init__(self,tools,scope,relative,*,proc_read=None,fd_open=None,fd_read=None,fd_close=None,event_path=None):
  self.tools,self.scope,self.relative=tools,scope,relative
  self.proc_read=proc_read or self.proc;self.roots={};self.ledger={};self.events=EvidenceEvents(event_path);self.fds={};self.sealed=False
  self.fd_open=fd_open or (lambda p:os.pidfd_open(p,0));self.fd_read=fd_read or self.readfd;self.fd_close=fd_close or os.close
 def proc(self,pid):
  p=pathlib.Path('/proc',str(pid));raw={'pid':pid,'epoch':time.time()}
  try:
   first=(p/'stat').read_text();f=first.rsplit(')',1)[1].split()
   uid=p.stat().st_uid;cg=(p/'cgroup').read_text().splitlines()
   raw.update(first_stat=first,first_uid=uid,cgroup=cg)
   second=(p/'stat').read_text();g=second.rsplit(')',1)[1].split();last_uid=p.stat().st_uid
   raw.update(second_stat=second,second_uid=last_uid)
   r={'pid':pid,'starttime_ticks':int(f[19]),'ppid':int(f[1]),'uid':uid,'cgroup':cg,'state':f[0]}
   # A changed live identity cannot become ownership through a mixed proc read.
   if int(g[19])!=r['starttime_ticks'] or last_uid!=uid:
    self.events.append(dict(raw,scope='NONATOMIC_PROC_LIVE_IDENTITY_DRIFT_REJECTED'))
    raise RuntimeError('Proc identity drift across stat/UID/cgroup reads')
   return r
  except BaseException as exc:
   raw.update(scope='ACTUAL_PROC_READ_FAILURE_RAW',error_type=type(exc).__name__,error=str(exc))
   self.events.append(raw);raise
 def readfd(self,fd):return int(next(v.split(':',1)[1].strip() for v in pathlib.Path('/proc/self/fdinfo',str(fd)).read_text().splitlines() if v.startswith('Pid:')))
 def rejection(self,message,pid,row,*,allow_unregistered_gone=False,pidfd_pid=None):
  self.events.append({'scope':'ACTUAL_CENSUS_BAD_ROW_BEFORE_RECHECK','epoch':time.time(),'reason':message,'pid':pid,'census_identity':row,'registered':pid in self.ledger,'actual_pidfd_pid':pidfd_pid})
  fresh=None;fresh_error=None;absent=False;attempts=[]
  for attempt in range(3 if allow_unregistered_gone and pid not in self.ledger else 1):
   fresh=None;fresh_error=None
   try:fresh=self.proc_read(pid)
   except (FileNotFoundError,ProcessLookupError):absent=True
   except BaseException as exc:fresh_error={'type':type(exc).__name__,'error':str(exc)}
   attempts.append({'epoch':time.time(),'identity':fresh,'error':fresh_error,'proc_absent':absent})
   self.events.append({'scope':'ACTUAL_REJECTION_PROC_RECHECK','pid':pid,'attempt':attempt,**attempts[-1]})
   if absent or fresh_error:break
   initial=row.get('initial',row)
   if any(fresh.get(k)!=initial.get(k) for k in ('pid','starttime_ticks','uid','cgroup')):break
   if attempt<2:time.sleep(.01)
  event={'scope':'ACTUAL_CENSUS_REJECTION_BEFORE_AUTHORITY','epoch':time.time(),'reason':message,'pid':pid,'census_identity':row,'fresh_identity':fresh,'fresh_error':fresh_error,'fresh_proc_absent':absent,'fresh_rechecks':attempts,'registered':pid in self.ledger,'actual_pidfd_pid':pidfd_pid}
  self.events.append(event)
  if allow_unregistered_gone and pid not in self.ledger and absent:
   self.events.append(dict(event,scope='UNREGISTERED_REJECTED_CENSUS_FRESH_PROC_ABSENT_NOT_REGISTERED_NOT_SIGNALLED'))
   return False
  raise RuntimeError(message)
 def checked(self,ok,message,pid,row,**kwargs):
  if not ok:return self.rejection(message,pid,row,**kwargs)
  return True
 def remember(self,pid,r,*,allow_gone=False):
  self.checked(not self.sealed,'New stage descendant during sealed cleanup',pid,r)
  fd=None
  try:
   fd=self.fd_open(pid);actual_fd_pid=self.fd_read(fd)
   again=self.proc_read(pid)
   if actual_fd_pid==-1 and allow_gone:
    if not self.checked(False,'Unregistered stage pidfd died during registration',pid,r,allow_unregistered_gone=True,pidfd_pid=actual_fd_pid):
     self.fd_close(fd);return False
   self.checked(actual_fd_pid==pid,'Actual stage pidfd registration drift',pid,r,pidfd_pid=actual_fd_pid)
   if not self.checked(all(again[k]==r[k] for k in ('pid','starttime_ticks','uid','cgroup')),'Actual stage identity changed during pidfd registration',pid,{'initial':r,'second':again},pidfd_pid=actual_fd_pid,allow_unregistered_gone=allow_gone):
    self.fd_close(fd);return False
  except (FileNotFoundError,ProcessLookupError):
   # A census is not ownership. A disappeared, unregistered descendant is
   # excluded without signalling it or inventing identity/SM evidence.
   try:live=self.proc_read(pid)
   except (FileNotFoundError,ProcessLookupError):live=None
   if fd is not None:self.fd_close(fd)
   if allow_gone and live is None:
    self.events.append({'scope':'UNREGISTERED_DESCENDANT_DISAPPEARED_DURING_REGISTRATION_NOT_OWNED','epoch':time.time(),'census_identity':r})
    return False
   raise
  except BaseException as exc:
   self.events.append({'scope':'ACTUAL_REGISTRATION_FAILURE_RAW','epoch':time.time(),'pid':pid,'census_identity':r,'error_type':type(exc).__name__,'error':str(exc),'pidfd_opened':fd is not None})
   if fd is not None:self.fd_close(fd)
   raise
  self.fds[pid]=fd;self.ledger[pid]=again
  self.events.append({'scope':'ACTUAL_LIVE_STAGE_PIDFD_REGISTRATION','epoch':time.time(),'identity':again,'pidfd_pid_at_registration':actual_fd_pid})
  return True
 def close(self):
  for fd in self.fds.values():self.fd_close(fd)
  self.fds.clear()
 def seal(self):
  self.pids(self.scope);self.sealed=True
  self.events.append({'scope':'ACTUAL_STAGE_LEDGER_SEALED_BEFORE_ORIGINAL_CLEANUP','registered':list(self.ledger.values())})
 def member(self,r):return any(v.split(':',2)[-1]==self.relative or v.split(':',2)[-1].startswith(self.relative+'/') for v in r['cgroup'])
 def register(self,pid):
  r=self.proc_read(pid);self.checked(r['pid']==pid and r['uid']==os.getuid() and self.member(r),'Actual stage root not owned same job',pid,r)
  self.roots[pid]=r;self.remember(pid,r)
 def pids(self,scope):
  require(scope==self.scope,'Stage scope changed');actual=self.tools.cgroup_pids(scope);rows={}
  for pid in actual:
   try:rows[pid]=self.proc_read(pid)
   except (FileNotFoundError,ProcessLookupError):continue
  owned=set();departed=set()
  for pid,r in rows.items():
   if pid in self.ledger:
    self.checked(r['starttime_ticks']==self.ledger[pid]['starttime_ticks'] and r['uid']==self.ledger[pid]['uid'],'Registered stage PID reused/UID changed',pid,r)
    self.checked(r['pid']==self.ledger[pid]['pid'] and r['cgroup']==self.ledger[pid]['cgroup'] and self.member(r),'Registered stage job/pidfd drift',pid,r)
    try:actual_fd_pid=self.fd_read(self.fds[pid])
    except BaseException as exc:self.rejection('Registered stage pidfd query failed: '+type(exc).__name__+': '+str(exc),pid,r)
    if actual_fd_pid==-1:
     self.checked(r['pid']==self.ledger[pid]['pid'] and r['cgroup']==self.ledger[pid]['cgroup'],'Registered stage exit census PID/cgroup drift',pid,r,pidfd_pid=actual_fd_pid)
     try:live=self.proc_read(pid)
     except (FileNotFoundError,ProcessLookupError):live=None
     self.checked(live is None,'Registered stage dead pidfd but live/reused PID',pid,r,pidfd_pid=actual_fd_pid)
     self.events.append({'scope':'REGISTERED_STAGE_PIDFD_DEAD_AND_PROC_ABSENT_NOT_ACTIVE','epoch':time.time(),'historical_identity':self.ledger[pid],'census_identity':r,'actual_pidfd_pid':-1})
     departed.add(pid);continue
    self.checked(actual_fd_pid==pid,'Registered stage job/pidfd drift',pid,r,pidfd_pid=actual_fd_pid);owned.add(pid)
  changed=True
  while changed:
   changed=False
   for pid,r in rows.items():
    if pid not in owned and pid not in departed and r['ppid'] in owned:
     if not self.checked(r['pid']==pid and r['uid']==os.getuid() and self.member(r),'Unknown stage descendant UID/job',pid,r,allow_unregistered_gone=True):
      departed.add(pid);continue
     if self.remember(pid,r,allow_gone=True):owned.add(pid);changed=True
     else:rows.pop(pid,None);changed=True;break
  self.events.append({'actual_job_pids':sorted(actual),'actual_stage_tree_pids':sorted(owned),'same_job_siblings_not_stage_owned':sorted(actual-owned)})
  return owned

def owned_stage(tools,scope,relative,*,expected_tools_sha256,proc_read=None,fd_open=None,fd_read=None,fd_close=None,event_path=None):
 """Clone the exact reviewed class with only its cgroup census dependency scoped."""
 path=pathlib.Path(tools.__file__).resolve();raw=path.read_bytes()
 require(hashlib.sha256(raw).hexdigest()==expected_tools_sha256,'Pinned original cleanup tools drift')
 node=next(n for n in ast.parse(raw).body if isinstance(n,ast.ClassDef) and n.name=='OwnedProcesses')
 tracker=StageScope(tools,scope,relative,proc_read=proc_read,fd_open=fd_open,fd_read=fd_read,fd_close=fd_close,event_path=event_path);namespace=dict(vars(tools));namespace['cgroup_pids']=tracker.pids
 exec(compile(ast.Module(body=[node],type_ignores=[]),str(path)+':stage-scoped-original-cleanup', 'exec'),namespace)
 obj=namespace['OwnedProcesses'](scope,relative);original_spawn=obj.spawn
 def spawn(*args,**kwargs):
  child=original_spawn(*args,**kwargs);tracker.register(child.pid)
  require(tracker.ledger[child.pid]['ppid']==os.getpid(),'Actual Popen stage root parent mismatch')
  return child
 obj.spawn=spawn;obj.stage_scope=tracker
 original_cleanup=obj.cleanup
 def cleanup():
  try:
   tracker.seal();return original_cleanup()
  finally:
   if hasattr(obj,'cleanup_record'):obj.cleanup_record.update(ownership_scope='ACTUAL_SPAWNED_STAGE_DESCENDANT_TREE_NOT_WHOLE_JOB',stage_identity_ledger=list(tracker.ledger.values()),stage_census_events=tracker.events)
   tracker.close()
 obj.cleanup=cleanup
 return obj
