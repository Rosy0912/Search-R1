"""Local five-minute supervision, one locked owner, with explicit reload advancement."""
import fcntl,json,subprocess,time
from pathlib import Path
base=Path(__file__).resolve().parent.parent
lock=(base/'work/supervise_m.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
command='/dataset/shaoshitong/search-r1-rl/cosearch/env/bin/python /dataset/shaoshitong/search-r1-rl/cosearch/tools/supervise_m.py'
while True:
 result=subprocess.run(['python3',str(base/'work/remote.py'),'--command',command,'--timeout','90'],capture_output=True,text=True)
 if result.returncode:
  print(json.dumps(dict(supervision_route_error=result.stderr[-1800:])),flush=True)
 else:
  try:record=json.loads(result.stdout)
  except ValueError:
   print(json.dumps(dict(invalid_supervision_output=result.stdout[-1800:],stderr=result.stderr[-1000:])),flush=True)
  else:
   (base/'outputs/evidence/m-supervision-latest.json').write_text(json.dumps(record,indent=2)+'\n')
   steps=[r['step'] for r in record.get('outer',[])]
   summary=dict(checked_at=record['checked_at'],job=2993,slurm=record['slurm'],outer_steps=steps,probe_passed=record.get('probe',{}).get('passed'),driver_exit=record.get('driver_exit'),acceptance=record.get('acceptance'),rank=record.get('rank'),progress=record.get('progress'))
   (base/'outputs/evidence/m-supervision-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
   state=json.loads((base/'outputs/STATE.json').read_text());state.update(last_supervision=record['checked_at'],actual_outer_updates=len(set(steps)),status='M2993 active native long-context gate' if not record.get('probe') else 'M2993 five-update training/reload ongoing');(base/'outputs/STATE.json').write_text(json.dumps(state,indent=2)+'\n')
   print(json.dumps(summary),flush=True)
   if record.get('driver_exit',{}).get('returncode') not in (None,0):
    state['status']='M2993 failed; inspect preserved logs and fix necessary issue';(base/'outputs/STATE.json').write_text(json.dumps(state,indent=2)+'\n');break
   if record.get('acceptance'):
    print('ACCEPTANCE_REQUIRES_AGENT_REVIEW',flush=True);break
 time.sleep(300)
