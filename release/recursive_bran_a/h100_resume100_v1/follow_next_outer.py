"""Wait for one next actual outer and subsequent generated tokens; no production mutations."""
import json,re,time
from pathlib import Path
base=Path('/dataset/shaoshitong/recursive-bran-a-h100-20261012/recovery-from100-20261013')
binding=json.loads((base/'final51713-supervisor-recovery-v1/binding.json').read_text())
assert binding['job_id']==3393
owner=binding['owner'];p=Path('/proc')/str(owner['pid'])
def live():
 try:
  s=(p/'stat').read_text().rsplit(')',1)[1].split()
  return p.stat().st_uid==owner['uid'] and int(s[19])==owner['tick'] and s[0]!='Z'
 except FileNotFoundError:return False
assert live(),'Owner not live; inspect actual terminal instead of blind restart'
log=base/'logs/training.log';latest=100
with log.open(errors='replace') as stream:
 for line in stream:
  a=re.search(r'\bstep:(\d+)\b',line);b=re.search(r'training/actual_outer_updates:\s*(\d+)',line)
  if a and b and a.group(1)==b.group(1) and 'timing_s/step:' in line:latest=max(latest,int(a.group(1)))
 offset=stream.tell()
target=min(latest+1,200);pending='';complete=None;started=time.monotonic()
print(json.dumps(dict(event='verified_start',time=time.time(),owner=owner,initial_complete=latest,target_complete=target)),flush=True)
while time.monotonic()-started<1800:
 with log.open(errors='replace') as stream:stream.seek(offset);pending+=stream.read();offset=stream.tell()
 lines=pending.split('\n');pending=lines.pop()
 for line in lines:
  a=re.search(r'\bstep:(\d+)\b',line);b=re.search(r'training/actual_outer_updates:\s*(\d+)',line)
  if a and b and int(a.group(1))==int(b.group(1))==target and 'timing_s/step:' in line:
   complete=line;print(json.dumps(dict(event='outer_complete',outer=target,time=time.time(),native_metrics=line)),flush=True)
  if complete and 'TREE_TIMING=' in line:
   try:d=json.JSONDecoder().raw_decode(line.split('TREE_TIMING=',1)[1])[0]
   except ValueError:continue
   if d.get('run_id')==binding['run_id'] and d.get('outer')==target+1 and d.get('event')=='trajectory_complete' and d.get('new_generated_tokens',d.get('policy_tokens',0))>0:
    print(json.dumps(dict(accepted=True,completed_outer=target,next_outer=target+1,next_generation=d,time=time.time())),flush=True);raise SystemExit(0)
 if complete and target==200:
  print(json.dumps(dict(training200_logged=True,final_still_requires_acceptance=True,time=time.time())),flush=True);raise SystemExit(0)
 if not live():
  print(json.dumps(dict(owner_terminal=True,target_complete=target,completed=complete is not None,time=time.time(),limits='Observer terminal is not a claim about training exit code')),flush=True);raise SystemExit(1)
 time.sleep(15)
print(json.dumps(dict(observation_window_elapsed=True,owner_live=live(),target_complete=target,completed=complete is not None,time=time.time())),flush=True)
