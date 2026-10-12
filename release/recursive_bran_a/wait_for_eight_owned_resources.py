"""CPU-only, single queued recovery; never signal or use an occupied GPU."""
import argparse,fcntl,json,os,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--source',required=True);p.add_argument('--gate',required=True);p.add_argument('--previous-owner',required=True);p.add_argument('--previous-session',required=True);p.add_argument('--previous-receipt',required=True);a=p.parse_args()
b=Path(a.base).resolve();source=Path(a.source).resolve();assert os.getuid()==1005 and source.parent==b
lock=(b/'receipts/eight-resource-recovery-queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
status=b/'receipts/eight-resource-recovery-queue.json'
def write(v):
 v.update(time=time.time(),pid=os.getpid(),uid=os.getuid(),source=str(source),scope='CPU-only wait; no GPU allocation or process signals')
 tmp=status.with_suffix('.pending-'+str(os.getpid()));tmp.write_text(json.dumps(v,indent=2));os.replace(tmp,status)
while True:
 assert subprocess.check_output(['tmux','display-message','-pt',a.previous_session,'#{pane_dead}:#{pane_dead_status}'],text=True).strip()=='1:1'
 assert not Path('/proc',a.previous_owner).exists(),'Previous owner still present'
 apps=set(subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).split())
 free=list(map(int,subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).split()))
 foreign=sorted(apps-{'2239529'})
 write(dict(state='WAITING_RESOURCES',other_compute_pids=foreign,directfree_mib=free))
 if not foreign and len(free)==8 and min(free)>=20000:break
 time.sleep(30)
command=[os.sys.executable,'-B',str(source/'release/recursive_bran_a/launch_search_only_eight.py')]
for name in ('base','source','gate','previous_owner','previous_session','previous_receipt'):command.extend(['--'+name.replace('_','-'),str(getattr(a,name))])
write(dict(state='EXECUTING_REVIEWED_LAUNCH',command=command))
r=subprocess.run(command,capture_output=True,text=True)
write(dict(state='LAUNCH_COMMAND_SUCCEEDED_NOT_TRAINING_ACCEPTANCE' if r.returncode==0 else 'LAUNCH_COMMAND_FAILED',returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))
print(r.stdout,flush=True);print(r.stderr,flush=True);raise SystemExit(r.returncode)
