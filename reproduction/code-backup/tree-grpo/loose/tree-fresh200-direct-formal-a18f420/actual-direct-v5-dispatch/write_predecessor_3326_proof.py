OBSERVATION={'observed_at': '2026-10-07T08:30:33.022521+00:00', 'observation_sha256': '490d349f60ddc58952a54858990a25e18e9b29ee9181ad5e89f1ca2107ff45fb', 'returncode': 0, 'stderr': '', 'fields': {'BatchHost': 'h100-9', 'CPUs/Task': '32', 'EndTime': '2026-10-07T16:29:14', 'ExitCode': '1:0', 'JobId': '3326', 'JobState': 'FAILED', 'MinMemoryNode': '256G', 'NodeList': 'h100-9', 'NumCPUs': '192', 'NumNodes': '1', 'NumTasks': '1', 'Reason': 'NonZeroExitCode', 'Requeue': '0', 'RunTime': '00:50:13', 'StartTime': '2026-10-07T15:39:01', 'SubmitTime': '2026-10-07T15:39:00', 'TimeLimit': '7-00:00:00'}, 'raw_stdout_retained': False}
import json,hashlib,subprocess,datetime,socket
from pathlib import Path
ROOT=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo');RUN=ROOT/'runs/tree-fresh200-train-3326'
assert socket.gethostname().split('.')[0]=='h100-9'
def desc(p,expected=None):
 raw=p.read_bytes();sha=hashlib.sha256(raw).hexdigest()
 if expected:assert sha==expected,(str(p),sha)
 return dict(path=str(p),sha256=sha),json.loads(raw)
result,r=desc(RUN/'run-result.json','c39d9f8f089bdd2208fe14e1288cdbc7fd469e1b61d56d82a71deea4326969f1')
cost,c=desc(RUN/'cost.json','5dcc71ef878a6c650e6c4553f93a7bb8727e0c1fb28d2b784c3d93bc9f5ed0e0')
journal,j=desc(ROOT/'instruct300/SUBMISSION_ONCE-fresh200-direct-2e26b55-v4.json')
assert r['code_commit']=='2e26b5505ce8ddfad8b627bba74337fbf4e23373' and str(r['job_id'])=='3326'
assert r['cleanup_residual']=={} and r['original_eight_pid_start_times_preserved'] is True
assert j['status']=='SUBMITTED' and j['job_id']==3326 and j['manifest_sha256']=='b9e7de77b1d1e168dd02af2f3be474af7d0dd2a510d13e4ee812fe6d6c4b9108'
a,a_data=desc(RUN/'checkpoints/training_state/attempt-results/attempt-000001.json','940c86ddd270085406f539a02a79c03cb4fdaaa837ccbe220ad7af3a12679d11')
assert a_data['status']=='COMPLETED_OPPORTUNITIES_OUTER_CREDITED'
allr=a_data['all_rank_result'];ranks=allr['ranks'];assert allr['decision']=='COMPLETED_OUTER' and len(ranks)==8
assert sorted(row['rank'] for row in ranks)==list(range(8))
assert all(row['optimizer_calls']==20 and row['opportunities_processed']==20 and row['opportunities_skipped']==0 and row['scheduler_before']==0 and row['scheduler_after']==1 for row in ranks)
q=subprocess.run(['squeue','--local','-h','-w','h100-9','-o','%i|%j|%T|%b'],text=True,capture_output=True,timeout=20)
assert q.returncode==0 and not q.stderr
assert not any(line.split('|')[1].startswith('tree-') for line in q.stdout.splitlines() if line.strip())
proof=dict(status='OWN_DIRECT_TRAIN_FAILED_WITH_CLEANUP',job_id=3326,code_commit=r['code_commit'],scheduler_state='FAILED',scheduler_exit_code='1:0',observed_at=OBSERVATION['observed_at'],positive_scheduler_observation=OBSERVATION,cleanup_residual={},original_eight_pid_start_times_preserved=True,run_result=result,cost=cost,once_journal=journal,direct_completed_outer_observation=dict(attempt_result=a,credited_outer=1,actual_optimizer_calls_per_rank=[20]*8,actual_scheduler_steps_per_rank=[1]*8,skips_per_rank=[0]*8,logical_optimizer_writes=20,physical_rank_optimizer_calls=160),model_optimizer_scheduler_checkpoint_available=False,continuation_mode='Cannot resume3326: no model or Adam checkpoint was saved before first milestone',empty_own_queue_observation=dict(observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),returncode=q.returncode,stderr=q.stderr,stdout=q.stdout))
raw=(json.dumps(proof,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode();path=RUN/'direct-training-predecessor-proof.json'
assert not path.exists(),'Proof alreadyexists; neveroverwrite'
with path.open('xb') as stream:stream.write(raw)
print(json.dumps(dict(proof_path=str(path),proof_sha256=hashlib.sha256(raw).hexdigest(),once_journal_sha256=journal['sha256'],proof=proof),allow_nan=False))
