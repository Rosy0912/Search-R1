"""Restrict original cleanup to the actual spawned stage tree, not same-job siblings."""
import ast,hashlib,os,pathlib,time

def require(ok,message):
 if not ok:raise RuntimeError(message)
class StageScope:
 def __init__(self,tools,scope,relative,*,proc_read=None,fd_open=None,fd_read=None,fd_close=None):
  self.tools,self.scope,self.relative=tools,scope,relative
  self.proc_read=proc_read or self.proc;self.roots={};self.ledger={};self.events=[];self.fds={};self.sealed=False
  self.fd_open=fd_open or (lambda p:os.pidfd_open(p,0));self.fd_read=fd_read or self.readfd;self.fd_close=fd_close or os.close
 def proc(self,pid):
  p=pathlib.Path('/proc',str(pid));f=(p/'stat').read_text().rsplit(')',1)[1].split()
  return {'pid':pid,'starttime_ticks':int(f[19]),'ppid':int(f[1]),'uid':p.stat().st_uid,'cgroup':(p/'cgroup').read_text().splitlines()}
 def readfd(self,fd):return int(next(v.split(':',1)[1].strip() for v in pathlib.Path('/proc/self/fdinfo',str(fd)).read_text().splitlines() if v.startswith('Pid:')))
 def remember(self,pid,r,*,allow_gone=False):
  require(not self.sealed,'New stage descendant during sealed cleanup')
  fd=None
  try:
   fd=self.fd_open(pid);actual_fd_pid=self.fd_read(fd)
   again=self.proc_read(pid)
   require(actual_fd_pid==pid,'Actual stage pidfd registration drift')
   require(all(again[k]==r[k] for k in ('pid','starttime_ticks','uid','cgroup')),'Actual stage identity changed during pidfd registration')
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
  except BaseException:
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
  r=self.proc_read(pid);require(r['pid']==pid and r['uid']==os.getuid() and self.member(r),'Actual stage root not owned same job')
  self.roots[pid]=r;self.remember(pid,r)
 def pids(self,scope):
  require(scope==self.scope,'Stage scope changed');actual=self.tools.cgroup_pids(scope);rows={}
  for pid in actual:
   try:rows[pid]=self.proc_read(pid)
   except FileNotFoundError:continue
  owned=set();departed=set()
  for pid,r in rows.items():
   if pid in self.ledger:
    require(r['starttime_ticks']==self.ledger[pid]['starttime_ticks'] and r['uid']==self.ledger[pid]['uid'],'Registered stage PID reused/UID changed')
    require(self.member(r),'Registered stage job/pidfd drift')
    actual_fd_pid=self.fd_read(self.fds[pid])
    if actual_fd_pid==-1:
     try:live=self.proc_read(pid)
     except (FileNotFoundError,ProcessLookupError):live=None
     require(live is None,'Registered stage dead pidfd but live/reused PID')
     self.events.append({'scope':'REGISTERED_STAGE_PIDFD_DEAD_AND_PROC_ABSENT_NOT_ACTIVE','epoch':time.time(),'historical_identity':self.ledger[pid],'census_identity':r,'actual_pidfd_pid':-1})
     departed.add(pid);continue
    require(actual_fd_pid==pid,'Registered stage job/pidfd drift');owned.add(pid)
  changed=True
  while changed:
   changed=False
   for pid,r in rows.items():
    if pid not in owned and pid not in departed and r['ppid'] in owned:
     require(r['uid']==os.getuid() and self.member(r),'Unknown stage descendant UID/job')
     if self.remember(pid,r,allow_gone=True):owned.add(pid);changed=True
     else:rows.pop(pid,None);changed=True;break
  self.events.append({'actual_job_pids':sorted(actual),'actual_stage_tree_pids':sorted(owned),'same_job_siblings_not_stage_owned':sorted(actual-owned)})
  return owned

def owned_stage(tools,scope,relative,*,expected_tools_sha256,proc_read=None,fd_open=None,fd_read=None,fd_close=None):
 """Clone the exact reviewed class with only its cgroup census dependency scoped."""
 path=pathlib.Path(tools.__file__).resolve();raw=path.read_bytes()
 require(hashlib.sha256(raw).hexdigest()==expected_tools_sha256,'Pinned original cleanup tools drift')
 node=next(n for n in ast.parse(raw).body if isinstance(n,ast.ClassDef) and n.name=='OwnedProcesses')
 tracker=StageScope(tools,scope,relative,proc_read=proc_read,fd_open=fd_open,fd_read=fd_read,fd_close=fd_close);namespace=dict(vars(tools));namespace['cgroup_pids']=tracker.pids
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
