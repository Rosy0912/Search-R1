OBSERVATION={'observed_at': '2026-10-07T07:26:50.975930+00:00', 'observation_sha256': 'bbff615064a522b452c806621072a639756f4ae3cbfa65dd828b02df3b7c9265', 'returncode': 0, 'stderr': '', 'fields': {'BatchHost': 'h100-9', 'CPUs/Task': '32', 'EndTime': '2026-10-07T15:26:24', 'ExitCode': '1:0', 'JobId': '3322', 'JobState': 'FAILED', 'MinMemoryNode': '256G', 'NodeList': 'h100-9', 'NumCPUs': '192', 'NumNodes': '1', 'NumTasks': '1', 'Reason': 'NonZeroExitCode', 'Requeue': '0', 'RunTime': '00:47:52', 'StartTime': '2026-10-07T14:38:32', 'SubmitTime': '2026-10-07T14:38:31', 'TimeLimit': '7-00:00:00'}, 'raw_stdout_retained': False}
import json,hashlib,subprocess,datetime,socket
from pathlib import Path
ROOT=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
RUN=ROOT/'runs/tree-fresh200-train-3322'
assert socket.gethostname().split('.')[0]=='h100-9'
def desc(p,expected=None):
 raw=p.read_bytes();sha=hashlib.sha256(raw).hexdigest()
 if expected: assert sha==expected,(str(p),sha)
 return dict(path=str(p),sha256=sha),json.loads(raw)
result,r=desc(RUN/'run-result.json','02ed5f95cad896770d51245667a3f02d8ce811702c80ac1b9e9b7e79f0e8e80e')
cost,c=desc(RUN/'cost.json','86f24918e79a1a102e4b6961f8b32b3dcd9903e140128c064ac0c51f9d4d947a')
journal,j=desc(ROOT/'instruct300/SUBMISSION_ONCE-fresh200-direct-a18f420-v3.json')
assert str(r['job_id'])=='3322' and r['cleanup_residual']=={} and r['original_eight_pid_start_times_preserved'] is True
assert j['status']=='SUBMITTED' and j['job_id']==3322 and j['manifest_sha256']=='3f21730c9665dc25072d71c8e79b7e91becfcf5e96ad0aa136df26545cb2fad4'
q=subprocess.run(['squeue','--local','-h','-w','h100-9','-o','%i|%j|%T|%b'],text=True,capture_output=True,timeout=20)
assert q.returncode==0 and not q.stderr
assert not any(line.split('|')[1].startswith('tree-') for line in q.stdout.splitlines() if line.strip())
partial,p=desc(RUN/'checkpoints/training_state/partial_attempt_000001_outer_000001/manifest.json','b9928363790c101054a986a17614d314e57337cd77fc8d2a3abfb21b1e8d81a8')
proof=dict(status='OWN_DIRECT_TRAIN_FAILED_WITH_CLEANUP',job_id=3322,code_commit='a18f4200ba4fd3e76ceeeae0cf409a43e25f7b98',scheduler_state='FAILED',scheduler_exit_code='1:0',observed_at=OBSERVATION['observed_at'],positive_scheduler_observation=OBSERVATION,cleanup_residual=r['cleanup_residual'],original_eight_pid_start_times_preserved=r['original_eight_pid_start_times_preserved'],run_result=result,cost=cost,once_journal=journal,empty_own_queue_observation=dict(observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),returncode=q.returncode,stderr=q.stderr,stdout=q.stdout),direct_partial_zero_observation=dict(manifest=partial,actual_optimizer_calls_per_rank=[0]*8,actual_scheduler_steps_per_rank=[0]*8,credited_outer=0))
raw=(json.dumps(proof,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
path=RUN/'direct-training-predecessor-proof.json'
if path.exists(): assert path.read_bytes()==raw,'Existing different proof cannot be overwritten'
else:
 with path.open('xb') as stream:stream.write(raw)
print(json.dumps(dict(proof_path=str(path),proof_sha256=hashlib.sha256(raw).hexdigest(),once_journal_sha256=journal['sha256'],proof=proof),allow_nan=False))
