"""Retire only the never-started own3350 after human explicitly selected node11."""
import subprocess,json,datetime,os
from pathlib import Path
assert os.uname().nodename=='h100-5'
r=Path('/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-baseline-aligned-em200-20261009');receipt=r/'scheduler-head/formal-rayip-v2/formal-submit-receipt.json'
b=json.loads(receipt.read_text());assert b['job']=='3350' and b['token']=='5fc47201ff6b4212ab6b73dee2bfefb5'
record=r/'scheduler-head/formal-rayip-v2/node11-retire3350.json'
if record.exists():print(record.read_text());raise SystemExit(0)
p=subprocess.run(['scontrol','show','job','3350','-o'],capture_output=True,text=True,check=True,timeout=15)
assert 'JobState=PENDING ' in p.stdout and 'RunTime=00:00:00 ' in p.stdout and 'NodeList=(null)' in p.stdout and 'BAPO_ALIGNED_EM200:formal:080ebbc46b89d61567cd33e1a5e2f1b40afd8b84:5fc47201ff6b4212ab6b73dee2bfefb5' in p.stdout
run=Path(b['run']);assert not (run/'train.log').exists()
intent={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'human_authorization':'那么就在H100-11启动，速度快一些行不？','scope':'only never-started own pending3350; goal remains ACTIVE; replace placement, no foreign action','before':p.stdout,'actual_outer':0,'actual_adam':0}
with record.open('x') as f:json.dump(intent,f);f.flush();os.fsync(f.fileno())
p=subprocess.run(['scancel','3350'],capture_output=True,text=True,timeout=15);intent['cancel']={'exit':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
q=subprocess.run(['squeue','-h','-j','3350','-o','%i|%T|%R|%k'],capture_output=True,text=True,timeout=15);intent['after_queue']={'exit':q.returncode,'stdout':q.stdout,'stderr':q.stderr}
record.write_text(json.dumps(intent,indent=2));print(json.dumps(intent))
if p.returncode:raise SystemExit(p.returncode)
