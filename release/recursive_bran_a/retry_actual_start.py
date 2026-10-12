"""Single actual HKUST1 startup attempt at a time; retry initialization OOM only."""
import argparse,fcntl,json,os,signal,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser()
for k in ('base','source','gate','previous-owner','previous-session','previous-receipt'):p.add_argument('--'+k,required=True)
a=p.parse_args();b=Path(a.base);source=Path(a.source);assert os.getuid()==1005
lock=(b/'receipts/eight-resource-recovery-queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
status=b/'receipts/actual-start-retry.json';attempt=0

def write(v):
 v.update(time=time.time(),supervisor_pid=os.getpid(),attempt=attempt,source=str(source))
 tmp=status.with_suffix('.pending');tmp.write_text(json.dumps(v,indent=2));os.replace(tmp,status)
 print(json.dumps(v),flush=True)

def own_processes(run):
 result=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit() or p.name=='2239529':continue
  try:
   if p.stat().st_uid!=1005:continue
   env=(p/'environ').read_bytes().split(b'\0')
   if not any(x in env for x in [b'E1_RUN_ID='+run.encode(),b'SEARCH_R1_BASELINE_RUN_ID='+run.encode()]):continue
   raw=(p/'stat').read_text().rsplit(') ',1)[1].split()
   if raw[0]=='Z':continue
   result.append((int(p.name),int(raw[19])))
  except (FileNotFoundError,ProcessLookupError,PermissionError):continue
 return result

def release_own(run):
 # Immutable run env + UID + fresh starttick + pidfd, never foreign processes.
 for pid,tick in own_processes(run):
  fd=os.pidfd_open(pid)
  try:
   raw=Path('/proc',str(pid),'stat').read_text().rsplit(') ',1)[1].split()
   assert int(raw[19])==tick
   signal.pidfd_send_signal(fd,signal.SIGTERM)
  except (FileNotFoundError,ProcessLookupError):pass
  finally:os.close(fd)
 deadline=time.time()+20
 while own_processes(run) and time.time()<deadline:time.sleep(1)
 return not own_processes(run)

while True:
 attempt+=1
 if (b/'checkpoints/training_state/global_step_5/manifest.json').exists():
  write(dict(state='HOLD_NEW_COMPLETE_CHECKPOINT',reason='Never roll back newly persisted progress toCP4'));break
 cmd=[os.sys.executable,'-B',str(source/'release/recursive_bran_a/launch_search_only_eight.py')]
 for k in ('base','source','gate','previous_owner','previous_session','previous_receipt'):cmd+=['--'+k.replace('_','-'),getattr(a,k)]
 cmd+=['--attempt-contention'];write(dict(state='ACTUAL_START_ATTEMPT'))
 r=subprocess.run(cmd,capture_output=True,text=True)
 if r.returncode:
  write(dict(state='LAUNCH_FAILED',returncode=r.returncode,stdout=r.stdout,stderr=r.stderr));time.sleep(5);continue
 proof=json.loads(r.stdout.splitlines()[-1]);run=proof['run_id'];log=Path(proof['log']);session=proof['session']
 write(dict(state='MODEL_STARTING',launch=proof))
 while True:
  pane=subprocess.check_output(['tmux','display-message','-pt',session,'#{pane_dead}:#{pane_dead_status}'],text=True).strip()
  text=log.read_text(errors='replace') if log.exists() else ''
  migrated='RECURSIVE_A_STATE_MIGRATED=' in text
  write(dict(state='TRAINING_AFTER_COMPLETE_RESTORE' if migrated else 'MODEL_STARTING',run_id=run,session=session,log=str(log),log_bytes=log.stat().st_size if log.exists() else 0,pane=pane))
  if pane.startswith('1:'):break
  time.sleep(5)
 if pane=='1:0':write(dict(state='TRAINING_PROCESS_EXIT0_UNACCEPTED',run_id=run));break
 memory=any(x in text.lower() for x in ('out of memory','outofmemoryerror','no available memory for the cache blocks','cublas_status_alloc_failed'))
 released=release_own(run)
 write(dict(state='INITIALIZATION_OOM_RETRY' if memory and not migrated and released else 'HOLD_FAILURE',run_id=run,initialization_only=not migrated,memory_failure=memory,own_release=released,actual_terminal=pane))
 if not (memory and not migrated and released):break
 time.sleep(10)
