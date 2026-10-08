OBSERVATION={'observed_at': '2026-10-07T11:02:14.133646+00:00', 'observation_sha256': '1028ad0189ebac9dd08ce3c56a745cf89f81134178b641a7377595996d5fc156', 'returncode': 0, 'stderr': '', 'fields': {'BatchHost': 'h100-9', 'CPUs/Task': '32', 'EndTime': '2026-10-07T19:01:32', 'ExitCode': '1:0', 'JobId': '3330', 'JobState': 'FAILED', 'MinMemoryNode': '256G', 'NodeList': 'h100-9', 'NumCPUs': '192', 'NumNodes': '1', 'NumTasks': '1', 'Reason': 'NonZeroExitCode', 'Requeue': '0', 'RunTime': '00:02:28', 'StartTime': '2026-10-07T18:59:04', 'SubmitTime': '2026-10-07T18:59:03', 'TimeLimit': '7-00:00:00'}, 'raw_stdout_retained': True, 'stdout': 'JobId=3330 JobName=tree-fresh200-train UserId=root(0) GroupId=root(0) MCS_label=N/A Priority=4294898736 Nice=0 Account=(null) QOS=(null) JobState=FAILED Reason=NonZeroExitCode Dependency=(null) Requeue=0 Restarts=0 BatchFlag=1 Reboot=0 ExitCode=1:0 RunTime=00:02:28 TimeLimit=7-00:00:00 TimeMin=N/A SubmitTime=2026-10-07T18:59:03 EligibleTime=2026-10-07T18:59:03 AccrueTime=2026-10-07T18:59:03 StartTime=2026-10-07T18:59:04 EndTime=2026-10-07T19:01:32 Deadline=N/A SuspendTime=None SecsPreSuspend=0 LastSchedEval=2026-10-07T18:59:04 Scheduler=Main Partition=gpu AllocNode:Sid=h100-9:1766995 ReqNodeList=h100-9 ExcNodeList=(null) NodeList=h100-9 BatchHost=h100-9 NumNodes=1 NumCPUs=192 NumTasks=1 CPUs/Task=32 ReqB:S:C:T=0:0:*:* TRES=cpu=192,mem=256G,node=1,billing=192 Socks/Node=* NtasksPerN:B:S:C=0:0:*:* CoreSpec=* MinCPUsNode=32 MinMemoryNode=256G MinTmpDiskNode=0 Features=(null) DelayBoot=00:00:00 OverSubscribe=NO Contiguous=0 Licenses=(null) Network=(null) Command=(null) WorkDir=/root StdErr=/root/slurm-3330.out StdIn=/dev/null StdOut=/root/slurm-3330.out Power= TresPerNode=gres:gpu:8 \n', 'raw_stdout_evidence_sha256': '4f00c0725ebba5ec17c4db36da56c824a9d20fe71aaabdb1619b3a96c8ec73b1'}
"""Root-only native3330 startup failure proof; JSON reads, queue/PID checks, one immutable proof write."""
import json,hashlib,subprocess,datetime,socket,os
from pathlib import Path
ROOT=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo');RUN=ROOT/'runs/tree-current200-continue-3330'
assert socket.gethostname().split('.')[0]=='h100-9' and not any(name.startswith('SLURM_') for name in os.environ)
def desc(path,expected=None):
 assert path.is_file() and not path.is_symlink()
 raw=path.read_bytes();sha=hashlib.sha256(raw).hexdigest()
 if expected:assert sha==expected,(str(path),sha)
 return dict(path=str(path),sha256=sha),json.loads(raw)
result,r=desc(RUN/'run-result.json','354324b39e0bf5cd9ca7c6fdc758a5e681081b650743b13562a045aeca45e4b2')
cost,c=desc(RUN/'cost.json','a74e23582aa65ded0f153b90c4fc3a0fd9588fe2149eab5acb476365bbe842ae')
journal,j=desc(ROOT/'instruct300/SUBMISSION_ONCE-current-fresh200-chain-mask-v7-9693f0c.json')
assert r['status']=='FAILED' and r['code_commit']=='9693f0c846fab483a03e1d70ce1109aeddc5e118' and str(r['job_id'])=='3330'
assert r['cleanup_residual']=={} and r['original_eight_pid_start_times_preserved'] is True
assert r.get('optimizer_committed') is None and r.get('scheduler_advanced') is None
assert j['status']=='SUBMITTED' and j['job_id']==3330 and j['cluster_name']=='wuhan-h100' and j['manifest_sha256']=='5f7071c0c6886b3a5b88abd216cd3b18bc0b14a1fd6293418717d742ece4ccff'
assert OBSERVATION['returncode']==0 and OBSERVATION['stderr']==''
assert all(OBSERVATION['fields'].get(k)==v for k,v in dict(JobId='3330',JobState='FAILED',ExitCode='1:0',BatchHost='h100-9',NodeList='h100-9',EndTime='2026-10-07T19:01:32',RunTime='00:02:28').items())
assert not (RUN/'output/current-chain-migrated.json').exists() and not (RUN/'output/fresh-initialization.json').exists()
assert not list((RUN/'checkpoints/training_state/attempt-results').glob('attempt-*.json')) and not list((RUN/'output/training-rollout-cost').glob('attempt-*.json'))
q=subprocess.run(['squeue','--local','-h','-w','h100-9','-o','%i|%j|%T|%b'],text=True,capture_output=True,timeout=20);assert q.returncode==0 and not q.stderr
assert not any(line.split('|')[1].startswith('tree-') for line in q.stdout.splitlines() if line.strip())
for pid,start in {**{p:'1502352181' for p in (1680782,1680783,1680784,1680785)},**{p:'1502398762' for p in (1696443,1696444,1696445,1696446)}}.items():
 text=Path(f'/proc/{pid}/stat').read_text();fields=text[text.rindex(')')+2:].split();assert fields[19]==start and fields[0] not in ('Z','X')
proof=dict(status='OWN_DIRECT_TRAIN_FAILED_WITH_CLEANUP',job_id=3330,code_commit=r['code_commit'],scheduler_state='FAILED',scheduler_exit_code='1:0',observed_at=OBSERVATION['observed_at'],positive_scheduler_observation=OBSERVATION,cleanup_residual={},original_eight_pid_start_times_preserved=True,run_result=result,cost=cost,once_journal=journal,prior_manifest_sha256=j['manifest_sha256'],phase='BEFORE_CURRENT_CHAIN_MIGRATION_AND_ACTOR_STARTUP',fixed_failure_message="'FuncWrapper' object has no attribute 'options'",native_optimizer_committed=r.get('optimizer_committed'),native_scheduler_advanced=r.get('scheduler_advanced'),native_null_counters_not_converted_to_zero=True,actual_migration_receipt_present=False,actual_initialization_receipt_present=False,child_training_current_counts=None,unpublished_rank_optimizer_calls_unknown=True,current_chain_parent_job_id=3328,empty_own_queue_observation=dict(observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),returncode=q.returncode,stderr=q.stderr,stdout=q.stdout))
raw=(json.dumps(proof,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode();path=RUN/'direct-training-predecessor-proof.json';assert not path.exists()
with path.open('xb') as stream:stream.write(raw)
print(json.dumps(dict(proof_path=str(path),proof_sha256=hashlib.sha256(raw).hexdigest(),once_journal_sha256=journal['sha256'],proof=proof),allow_nan=False))
