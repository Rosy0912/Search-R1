"""Metadata-only raw scheduler observer, never sends signals or changes training."""
import datetime,hashlib,json,subprocess,sys,time
from pathlib import Path
r=Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004')
job=sys.argv[1];assert job.isdigit()
out=r/f'logs/quality10-native-terminal-{job}-raw-v1.json'
log=r/f'logs/quality10-native-scheduler-watch-{job}-raw-v1.jsonl'
assert not out.exists() and not log.exists(),'Existing observer evidence: inspect, do not overwrite'
terminal={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
with log.open('x') as f:
 for _ in range(400):
  p=subprocess.run(['scontrol','show','job',job,'-o'],capture_output=True,text=True)
  snapshot={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'args':['scontrol','show','job',job,'-o'],'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
  f.write(json.dumps(snapshot)+'\n');f.flush()
  fields=dict(t.split('=',1) for t in p.stdout.split() if '=' in t)
  if p.returncode==0 and fields.get('JobState') in terminal:
   assert fields.get('JobId')==job and fields.get('NodeList')=='h100-8'
   snapshot.update(fields=fields,scope='Actual raw terminal observation only; not independent acceptance')
   with out.open('x') as g:g.write(json.dumps(snapshot,indent=2)+'\n')
   print(json.dumps({'path':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'state':fields['JobState'],'exit_code':fields.get('ExitCode')}),flush=True)
   break
  time.sleep(20)
 else:raise TimeoutError('Observer timeout: inspect original job, never resubmit')
