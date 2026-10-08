"""Stop only own slow3339 after byte-bound completed6; save actual evidence."""
import json, shlex
from datetime import datetime, timezone
from pathlib import Path
import ssh_retained18 as ssh
HERE=Path(__file__).resolve().parent
WORKSPACE=HERE.parents[1]
AUTH=WORKSPACE/'outputs/tree-fresh200/RETAINED18_HUMAN_SPEED_TRANSITION3339_TO_NATIVE_PREFIX_AUTHORITY_20261008.json'
authority=json.loads(AUTH.read_text())
code=r"""from pathlib import Path
import json,subprocess,socket,hashlib,time
root=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo');run=root/'runs/tree-current200-continue-3339'
assert socket.gethostname().split('.')[0]=='h100-10'
def query():
 r=subprocess.run(['scontrol','--local','show','job','3339'],capture_output=True,text=True,timeout=20)
 return dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
def native():
 result={}
 for rel in ['launch-manifest.json','checkpoints/training_state/rolling-completed/latest.json','checkpoints/training_state/rolling-completed/global_step_6/manifest.json','checkpoints/training_state/attempt-results/attempt-000006.json','checkpoints/training_state/attempt-results/attempt-000007.json','run-result.json','cost.json','termination-cost.json','output/train200-terminal.json']:
  p=run/rel
  if p.exists():
   b=p.read_bytes();result[rel]=dict(sha256=hashlib.sha256(b).hexdigest(),value=json.loads(b))
 return result
before=query();assert before['returncode']==0 and 'JobState=RUNNING' in before['stdout'] and 'NodeList=h100-10' in before['stdout']
values=native();assert values['launch-manifest.json']['value']['code_commit']=='62271081ba1bb74d8fc3f4958d623758efa74fa5'
assert values['checkpoints/training_state/rolling-completed/global_step_6/manifest.json']['sha256']=='2421b7f1f8f8db0d5b73e342e9de1a9c8b61a5fe08cb6596da492a0d3a04fbc3'
assert values['checkpoints/training_state/rolling-completed/latest.json']['value']['manifest_sha256']=='2421b7f1f8f8db0d5b73e342e9de1a9c8b61a5fe08cb6596da492a0d3a04fbc3'
assert values['checkpoints/training_state/attempt-results/attempt-000006.json']['value']['accepted_outer']==6
assert 'checkpoints/training_state/attempt-results/attempt-000007.json' not in values
journal=json.loads((root/'instruct300/SUBMISSION_ONCE-retained18-current200-speed-v13-6227108.json').read_text())
assert journal['status']=='SUBMITTED' and journal['job_id']==3339 and journal['manifest_sha256']=='cbca5e1de3188eaf071ce02c295d2607475ec003c53352110a4c6fbf72cbfa63'
preserved={}
for rel in ['stage.log','guard.log','retriever.log','resource-samples.jsonl']:
 p=run/rel
 if p.is_file():
  b=p.read_bytes();preserved[rel]=dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),tail=b[-40000:].decode('utf8',errors='replace'))
r=subprocess.run(['scancel','3339'],capture_output=True,text=True,timeout=20)
assert r.returncode==0
print(json.dumps(dict(status='SCANCEL_ONLY_OWN3339_REQUESTED',observed_epoch=time.time(),job_id=3339,before_scheduler=before,before_native=values,before_logs=preserved,scancel_returncode=r.returncode,scancel_stdout=r.stdout,scancel_stderr=r.stderr,after_scheduler=query(),other_jobs_modified=False,signals_to_individual_pids=False)),flush=True)
"""
result=ssh.run('python3 -c '+shlex.quote(code),timeout=80)
record=dict(observed_at=datetime.now(timezone.utc).isoformat(),authority=authority,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,no_blind_retry=True)
out=WORKSPACE/'outputs/tree-fresh200/RETAINED18_MANUAL_STOP_SLOW3339_REQUEST_20261008.json'
out.write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(dict(returncode=result.returncode,stdout=result.stdout[-1800:],stderr=result.stderr)))
if result.returncode:raise SystemExit(result.returncode)
