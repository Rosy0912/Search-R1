
import json,sys,hashlib,io,contextlib,os,time,signal,subprocess
from pathlib import Path
p=json.load(sys.stdin);keeper=p['keeper'];base=Path(p['new_base']);repo=base/'repo'
assert hashlib.sha256(p['reader'].encode()).hexdigest()==p['reader_sha256']
assert hashlib.sha256((base/'logs/actor-probe-production-deployment-review.json').read_bytes()).hexdigest()==p['review_sha256']
assert hashlib.sha256((repo/'reproduction/a800/owned_processes.py').read_bytes()).hexdigest()=='df83d747f3b86a187eb2dc4267ac146a6afeab7338f274ab5ecd8e7de9e2969e'
sys.path.insert(0,str(repo))
from reproduction.a800.owned_processes import signal_owned,identity
output=io.StringIO();original=sys.stdin
try:
 sys.stdin=io.StringIO(json.dumps(keeper))
 with contextlib.redirect_stdout(output):exec(compile(p['reader'],'<published-bounded-read>','exec'),{'__name__':'__main__'})
finally:sys.stdin=original
reading=json.loads(output.getvalue());print(json.dumps(dict(role='pre_release_current_resource_read',observation=reading)),flush=True)
assert reading['actual_owned_60GiB_each_verified'] and reading['post_READY_retention_sample_verified']
assert reading['process']['alive'] and not reading['foreign_training_cuda_applications'] and not reading['foreign_auxiliary_cuda_applications'],'Currentassigned GPU foreigncontexts: preserve holder; no signal or launch'
actual=identity(keeper['identity']['pid']);assert all(actual[k]==keeper['identity'][k] for k in ['pid','start_ticks','sid','uid'])
old_claim=Path(keeper['namespace'])/'release-for-reviewed-training-once.json'
old=json.loads(old_claim.read_text());assert old['identity']==keeper['identity'] and old['new_base']==str(base) and old['review_sha256']==p['review_sha256']
assert hasattr(os,'pidfd_open') and hasattr(signal,'pidfd_send_signal')
claim=Path(keeper['namespace'])/'release-runtime-adaptation-once.json'
data=dict(role='owned_capacity_release_for_reviewed_training',identity=keeper['identity'],new_base=str(base),review_sha256=p['review_sha256'],pre_release_observation_sha256=hashlib.sha256((json.dumps(reading,sort_keys=True)+'\n').encode()).hexdigest(),signal='SIGTERM',requested_unix=time.time(),no_foreign_signal=True,prior_attempt_signal_sent=False,prior_failure='bootstrap Python lacks pidfd_open; known pre-signal exception',prior_claim_sha256=hashlib.sha256(old_claim.read_bytes()).hexdigest(),python_executable=sys.executable)
fd=os.open(claim,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
with os.fdopen(fd,'w') as f:json.dump(data,f,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
fd=os.open(claim.parent,os.O_RDONLY|os.O_DIRECTORY)
try:os.fsync(fd)
finally:os.close(fd)
signal_owned(keeper['identity'],signal.SIGTERM)
print(json.dumps(dict(role='exact_owned_keeper_SIGTERM_sent',identity=keeper['identity'],checked_unix=time.time(),claim_sha256=hashlib.sha256(claim.read_bytes()).hexdigest())),flush=True)
def inventory():
 raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True,timeout=10)
 mapping={row.split(',')[1].strip().lower():int(row.split(',')[0]) for row in raw.splitlines()}
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=10)
 rows=[]
 for row in apps.splitlines():
  fields=[x.strip() for x in row.split(',')];idx=mapping[fields[0].lower()]
  if idx in [0,1,2,3,5,6]:rows.append(dict(gpu_index=idx,pid=int(fields[1]),used_mib=fields[2],uuid=fields[0]))
 return rows
start=time.monotonic();samples=[]
while True:
 try:proc=identity(keeper['identity']['pid']);samebirth=all(proc[k]==keeper['identity'][k] for k in ['pid','start_ticks','sid','uid']);alive=samebirth and proc['state'] not in ['Z','X']
 except (FileNotFoundError,ProcessLookupError):proc=None;samebirth=False;alive=False
 rows=inventory();owned=[x for x in rows if x['pid']==keeper['identity']['pid']]
 samples.append(dict(checked_unix=time.time(),same_birth_alive=alive,owned_cuda_count=len(owned),assigned_cuda_count=len(rows)))
 if not alive and not owned:
  result=dict(role='actual_owned_capacity_exit_and_CUDA_drain',identity=keeper['identity'],owned_birth_exited=True,owned_CUDA_released=True,all_assigned_contexts_absent=not rows,assigned_contexts=rows,last_process=proc,bounded_samples=samples,checked_unix=time.time(),new_trainer_started=False,native_algorithm_accepted=False)
  print(json.dumps(result),flush=True);break
 assert time.monotonic()-start<60,'Ownedrelease actualdrain unconfirmed within bound; do notlaunch'
 time.sleep(2)
