"""Read-only continuous roots protection, distinct from stage cleanup ownership."""
import os,pathlib

def protected_stage_baseline(tools,scope,relative,gpu_records,*,read=None,start_pid=None):
 def actual(pid):
  r=dict(tools.process_identity(pid));r['uid']=(pathlib.Path('/proc')/str(pid)).stat().st_uid;return r
 read=read or actual
 def member(r):return any(v.split(':',2)[-1]==relative or v.split(':',2)[-1].startswith(relative+'/') for v in r.get('cgroup',[]))
 records={};pid=os.getpid() if start_pid is None else start_pid;visited=set()
 while pid and pid not in visited:
  visited.add(pid);r=read(pid)
  if not member(r):break
  if r.get('uid')!=os.getuid() or not isinstance(r.get('starttime_ticks'),int):raise RuntimeError('Continuous stage ancestor UID/start identity unavailable')
  records[pid]=r;pid=r['ppid']
 if not records:raise RuntimeError('Actual stage owner/root baseline absent')
 for entry in gpu_records:
  if member(entry['identity']):
   r=read(entry['pid'])
   if r.get('uid')!=os.getuid() or not member(r) or r.get('starttime_ticks')!=entry['identity']['starttime_ticks']:raise RuntimeError('Original resident GPU identity changed')
   records[entry['pid']]=r
 return records

def post_cleanup_remaining(owned,tools,scope):
 """Read-only evidence after cleanup closes pidfds; never grant signal authority."""
 evidence={'scope':'READ_ONLY_POST_CLEANUP_IDENTITY_CENSUS_NOT_PIDFD_AUTHORITY'}
 try:
  actual=set(tools.cgroup_pids(scope));ledger=owned.stage_scope.ledger
  evidence['actual_job_pids']=sorted(actual);remaining=[]
  for pid in sorted(actual.intersection(ledger)):
   r=tools.process_identity(pid);previous=ledger[pid]
   if r.get('starttime_ticks')!=previous['starttime_ticks']:raise RuntimeError('Post-cleanup ledger PID reused')
   remaining.append(pid)
  evidence['actual_registered_stage_pids_remaining']=remaining
  record=getattr(owned,'cleanup_record',{})
  if record.get('actual_cgroup_empty') is not True:raise RuntimeError('Original scoped cleanup did not prove empty')
  if remaining:raise RuntimeError('Registered stage identity still present after cleanup')
  return remaining,evidence
 except BaseException as exc:
  evidence.update(error_type=type(exc).__name__,error=str(exc))
  record=getattr(owned,'cleanup_record',{})
  recorded=record.get('remaining')
  remaining=[r['pid'] for r in recorded] if isinstance(recorded,list) and recorded else None
  if evidence.get('actual_registered_stage_pids_remaining'):remaining=evidence['actual_registered_stage_pids_remaining']
  return remaining,evidence

def continuous_baseline_missing(tools,baseline,*,read=None):
 def actual(pid):
  current=dict(tools.process_identity(pid));current["uid"]=(pathlib.Path("/proc")/str(pid)).stat().st_uid;return current
 read=read or actual
 missing=[]
 for pid,previous in baseline.items():
  try:
   current=read(pid)
   if any(current.get(k)!=previous.get(k) for k in ('pid','starttime_ticks','uid','cgroup')):missing.append({'pid':pid,'expected':previous,'actual':current})
  except BaseException as exc:missing.append({'pid':pid,'expected':previous,'identity_unavailable':True,'error_type':type(exc).__name__,'error':str(exc)})
 return missing
