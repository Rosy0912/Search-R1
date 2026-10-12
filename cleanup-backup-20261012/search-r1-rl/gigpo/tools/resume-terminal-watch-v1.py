"""Observe only job3130; freeze actual terminal scheduler proof before expiry."""
from pathlib import Path
import json,subprocess,hashlib,datetime,time,sys
ROOT=Path('/mnt/data/shaoshitong/search-r1-rl/gigpo')
RUN=ROOT/'runs/gigpo-resume200to300-job3130-1791086878335123421'
OUT=ROOT/'logs/resume3130-terminal-scheduler-observation-v1.json'
COLLECTOR=ROOT/'tools/resume-final-evidence-collect-v2.py'
assert not OUT.exists(),'Read existing terminal observation, never overwrite'
assert hashlib.sha256(COLLECTOR.read_bytes()).hexdigest()=='5d1daa3ba0d6bca984e804bc7be9822b15a4b63d6c3372d84a812d2da694c0c8'
def command(args):
 p=subprocess.run(args,capture_output=True,text=True)
 return dict(args=args,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
deadline=time.monotonic()+7200
while time.monotonic()<deadline:
 snapshot=command(['scontrol','show','job','3130'])
 fields={}
 for token in snapshot['stdout'].split():
  if '=' in token:
   key,value=token.split('=',1);fields[key]=value
 state=fields.get('JobState')
 if snapshot['returncode']==0 and state in {'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}:
  assert fields.get('JobId')=='3130' and fields.get('NodeList')=='h100-8','Wrong scheduler identity'
  result_path=RUN/'run-result.json'
  result_bytes=result_path.read_bytes() if result_path.exists() else None
  obj=dict(captured_at=datetime.datetime.now().isoformat(),job_id='3130',run=str(RUN),scope='Actual terminal scheduler observation only; not final acceptance',scheduler=snapshot,fields=fields,queue=command(['squeue','-j','3130','-o','%.18i %.9T %.10M %.20R']),run_result_text=None if result_bytes is None else result_bytes.decode(),run_result_sha256=None if result_bytes is None else hashlib.sha256(result_bytes).hexdigest())
  with OUT.open('x') as stream:json.dump(obj,stream,indent=2)
  print(json.dumps(dict(terminal_observation=str(OUT),sha256=hashlib.sha256(OUT.read_bytes()).hexdigest(),state=state,exit_code=fields.get('ExitCode'))),flush=True)
  assert state=='COMPLETED' and fields.get('ExitCode')=='0:0','Training or final reload did not exit successfully; preserve observation, no retry'
  result=json.loads(result_bytes)
  assert result['status']=='CHECKPOINT_RELOADED_UNREVIEWED' and result['cleanup_residual_states']=={},'Missing real reload/cleanup evidence'
  assert hashlib.sha256(COLLECTOR.read_bytes()).hexdigest()=='5d1daa3ba0d6bca984e804bc7be9822b15a4b63d6c3372d84a812d2da694c0c8'
  subprocess.run([sys.executable,str(COLLECTOR)],check=True)
  break
 time.sleep(10)
else:
 raise TimeoutError('Observer deadline reached; inspect same job3130, never resubmit')
