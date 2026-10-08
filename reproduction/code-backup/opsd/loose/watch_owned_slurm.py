"""Archive the scheduler's actual terminal text before its retention window expires."""
import argparse,datetime,json,re,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('job',type=int);p.add_argument('--hours',type=float,default=169);a=p.parse_args();assert a.job>0
base=Path(__file__).resolve().parent;deadline=time.monotonic()+a.hours*3600
while time.monotonic()<deadline:
 try:
  r=subprocess.run(['python3',str(base/'head.py'),f'scontrol show job {a.job}'],capture_output=True,text=True,timeout=90)
  if r.returncode==0 and re.search(r'\bJobId='+str(a.job)+r'\b',r.stdout):
   (base/f'job{a.job}-slurm-watch.txt').write_text(r.stdout)
   if re.search(r'\bJobState=(COMPLETED|FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED|BOOT_FAIL)\b',r.stdout):
    final=base/f'job{a.job}-slurm-final.txt';temp=final.with_suffix('.partial');temp.write_text(r.stdout);temp.replace(final);print('ACTUAL_TERMINAL_ARCHIVED',final,flush=True);raise SystemExit(0)
  else:
   (base/f'job{a.job}-watch-last-error.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}))
 except (subprocess.TimeoutExpired,OSError) as e:
  print(type(e).__name__,str(e),flush=True)
 time.sleep(15)
raise SystemExit('WATCH_TIMEOUT_WITHOUT_TERMINAL_PROOF')
