"""Actual file barriers: teacher ready cannot close global startup sampling."""
import json,os,pathlib,time
from contracts import require

def publish(path,value,*,replace=False):
 path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
 temporary=path.with_name(path.name+'.tmp')
 fd=os.open(str(temporary),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 try:
  with os.fdopen(fd,'w') as f:json.dump(value,f,sort_keys=True,allow_nan=False);f.flush();os.fsync(f.fileno())
  if replace:os.replace(temporary,path)
  else:os.link(temporary,path)
  d=os.open(str(path.parent),os.O_DIRECTORY);os.fsync(d);os.close(d)
 finally:
  if temporary.exists():temporary.unlink()

def wait(path,children,errors,timeout=1800):
 began=time.monotonic();path=pathlib.Path(path)
 while not path.exists():
  require(not errors,'owned monitor failed: '+repr(errors))
  require(all(p.poll() is None for p in children),'owned process exited before actual barrier '+path.name)
  require(time.monotonic()-began<timeout,'actual barrier timeout '+path.name);time.sleep(.2)
 require(path.is_file() and not path.is_symlink() and path.stat().st_mode & 0o077==0,'unsafe actual barrier')
 return json.loads(path.read_text())

class StartupPhases:
 def __init__(self,guard):self.guard=guard;self.state='E5_ONLY';self.teacher_ready=False;self.worker_ready=False
 def arm(self):require(self.state=='E5_ONLY','startup rearmed');self.guard.closed=False;self.state='OWNER_CONTEXT_AND_ACTOR_STARTUP'
 def teacher(self):require(self.state=='OWNER_CONTEXT_AND_ACTOR_STARTUP','teacher phase ordering');self.teacher_ready=True
 def workers(self,receipts,validate):
  require(self.teacher_ready and not self.worker_ready,'global startup readiness order');validate(receipts)
  self.worker_ready=True;self.guard.close_startup();self.state='GLOBAL_WORKERS_READY'
 def ready_check(self):return self.worker_ready

def join_monitor(stop,monitor,errors,timeout=180):
 stop.set();monitor.join(timeout=timeout)
 require(not monitor.is_alive(),'Owned active monitor still live at terminal boundary')
 # Errors may arrive while join waits; closure is not permission to discard them.
 require(not errors,'Strict active monitor failed during terminal closure '+repr(errors))
 return True


def hold_algorithm_frame(trainer,mode):
 """Keep trainer/actual worker references alive before fit returns into main_task."""
 require(trainer is not None,'Original trainer frame required')
 run=pathlib.Path(os.environ['TOPD32B_RUN'])
 publish(run/'driver-algorithm-returned.json',dict(status='ORIGINAL_FIT_COMPLETED_TRAINER_HELD_PENDING_OWNED_TERMINAL_EXIT',mode=mode,pid=os.getpid(),job=os.environ['SLURM_JOB_ID']))
 permission=wait(run/'authorized-terminal-exit.json',[],[],3600)
 require(permission['job']==os.environ['SLURM_JOB_ID'] and permission['parent_pid']==os.getppid()
         and permission['actual_driver_return_sha256']==__import__('hashlib').sha256((run/'driver-algorithm-returned.json').read_bytes()).hexdigest()
         and permission['phase']=='NO_MORE_GPU_WORK_OWNED_EXIT','Owned held-trainer terminal permission mismatch')
 return trainer
