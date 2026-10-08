OBSERVATION={'observed_at': '2026-10-07T10:37:13.249410+00:00', 'observation_sha256': 'b82054dde7ee2e8b66aa59eae21ad9dd524fcbcdab9a4cd939da73f0f073c79d', 'returncode': 0, 'stderr': '', 'fields': {'BatchHost': 'h100-9', 'CPUs/Task': '32', 'EndTime': '2026-10-07T18:37:07', 'ExitCode': '1:0', 'JobId': '3328', 'JobState': 'FAILED', 'MinMemoryNode': '256G', 'NodeList': 'h100-9', 'NumCPUs': '192', 'NumNodes': '1', 'NumTasks': '1', 'Reason': 'NonZeroExitCode', 'Requeue': '0', 'RunTime': '01:44:53', 'StartTime': '2026-10-07T16:52:14', 'SubmitTime': '2026-10-07T16:52:14', 'TimeLimit': '7-00:00:00'}, 'raw_stdout_retained': True, 'stdout': 'JobId=3328 JobName=tree-fresh200-train UserId=root(0) GroupId=root(0) MCS_label=N/A Priority=4294898738 Nice=0 Account=(null) QOS=(null) JobState=FAILED Reason=NonZeroExitCode Dependency=(null) Requeue=0 Restarts=0 BatchFlag=1 Reboot=0 ExitCode=1:0 RunTime=01:44:53 TimeLimit=7-00:00:00 TimeMin=N/A SubmitTime=2026-10-07T16:52:14 EligibleTime=2026-10-07T16:52:14 AccrueTime=2026-10-07T16:52:14 StartTime=2026-10-07T16:52:14 EndTime=2026-10-07T18:37:07 Deadline=N/A SuspendTime=None SecsPreSuspend=0 LastSchedEval=2026-10-07T16:52:14 Scheduler=Main Partition=gpu AllocNode:Sid=h100-9:1715767 ReqNodeList=h100-9 ExcNodeList=(null) NodeList=h100-9 BatchHost=h100-9 NumNodes=1 NumCPUs=192 NumTasks=1 CPUs/Task=32 ReqB:S:C:T=0:0:*:* TRES=cpu=192,mem=256G,node=1,billing=192 Socks/Node=* NtasksPerN:B:S:C=0:0:*:* CoreSpec=* MinCPUsNode=32 MinMemoryNode=256G MinTmpDiskNode=0 Features=(null) DelayBoot=00:00:00 OverSubscribe=NO Contiguous=0 Licenses=(null) Network=(null) Command=(null) WorkDir=/root StdErr=/root/slurm-3328.out StdIn=/dev/null StdOut=/root/slurm-3328.out Power= TresPerNode=gres:gpu:8 \n', 'raw_stdout_evidence_sha256': '7e921fa3fb3377d7b30f8a4a2e7e08fd2f85b0af6da54810e77e708764b5c2ef'}
"""One native3328 terminal closure; run only by root on the original h1009."""
import json,hashlib,subprocess,datetime,socket,os
from pathlib import Path
ROOT=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo');RUN=ROOT/'runs/tree-fresh200-train-3328'
assert socket.gethostname().split('.')[0]=='h100-9'
assert not any(name.startswith('SLURM_') for name in os.environ), 'Execute from management shell, never within another allocation'
def desc(path,expected=None):
 assert path.is_file() and not path.is_symlink()
 raw=path.read_bytes();sha=hashlib.sha256(raw).hexdigest()
 if expected:assert sha==expected,(str(path),sha)
 return dict(path=str(path),sha256=sha),json.loads(raw)
result,r=desc(RUN/'run-result.json','7386e708ee492f618085b9dcede97d22704dd58bc6676abf1a5650fa8a41cbf6')
cost,c=desc(RUN/'cost.json','abd0336c14967af8eac5c01327f427b529ba8f021c7ac3143a2d10b3f3dc00b2')
journal,j=desc(ROOT/'instruct300/SUBMISSION_ONCE-fresh200-direct-4a65e9a-v6.json')
parent,m=desc(RUN/'checkpoints/training_state/rolling-completed/global_step_2/manifest.json','067d71ee3c7fa0e00da471bebbd9e8913ec7e5de0b1c98bdb7598d7a022565a4')
assert r['status']=='FAILED' and r['code_commit']=='4a65e9a3ab87e84e1e9dd283b664cdef3a224c8a' and str(r['job_id'])=='3328'
assert r['cleanup_residual']=={} and r['original_eight_pid_start_times_preserved'] is True
assert j['status']=='SUBMITTED' and j['job_id']==3328 and j['cluster_name']=='wuhan-h100' and j['manifest_sha256']=='ef2817200f999c74deca43ea3de501f6bf18276ea838b38b35342ebf3368265f'
assert OBSERVATION['returncode']==0 and OBSERVATION['stderr']==''
assert all(OBSERVATION['fields'].get(k)==v for k,v in dict(JobId='3328',JobState='FAILED',ExitCode='1:0',BatchHost='h100-9',NodeList='h100-9',EndTime='2026-10-07T18:37:07',RunTime='01:44:53').items())
progress=r['actual_progress_evidence'];counts={str(rank):40 for rank in range(8)}
assert progress['actual_completed_outer_from_attempt_results']==2 and progress['training_attempts_seen']==2 and progress['completed_actual_optimizer_calls_per_rank']==counts and not progress['errors']
assert m['status']=='COMPLETE_NATIVE_ROLLING_STATE_SAVED' and m['identity']['completed_step']==2 and m['identity']['code_commit']==r['code_commit'] and m['driver']['next_step']==3
assert len(m['worker_states'])==8 and all(w['scheduler_last_epoch']==2 and w['actual_total_optimizer_calls']==40 and w['full_worker_state_finite'] is True for w in m['worker_states'])
q=subprocess.run(['squeue','--local','-h','-w','h100-9','-o','%i|%j|%T|%b'],text=True,capture_output=True,timeout=20);assert q.returncode==0 and not q.stderr
assert not any(line.split('|')[1].startswith('tree-') for line in q.stdout.splitlines() if line.strip())
for pid,start in {**{p:'1502352181' for p in (1680782,1680783,1680784,1680785)},**{p:'1502398762' for p in (1696443,1696444,1696445,1696446)}}.items():
 text=Path(f'/proc/{pid}/stat').read_text();fields=text[text.rindex(')')+2:].split();assert fields[19]==start and fields[0] not in ('Z','X')
proof=dict(status='OWN_DIRECT_TRAIN_FAILED_WITH_CLEANUP',job_id=3328,code_commit=r['code_commit'],scheduler_state='FAILED',scheduler_exit_code='1:0',observed_at=OBSERVATION['observed_at'],positive_scheduler_observation=OBSERVATION,cleanup_residual={},original_eight_pid_start_times_preserved=True,run_result=result,cost=cost,once_journal=journal,prior_manifest_sha256=j['manifest_sha256'],phase='FAILED_THIRD_GENERATION_AFTER_COMPLETED_ROLLING2',native_optimizer_committed=r.get('optimizer_committed'),native_scheduler_advanced=r.get('scheduler_advanced'),native_null_counters_not_converted_to_zero=True,direct_completed_outer_observation=dict(credited_outer=2,training_attempts=2,actual_optimizer_calls_per_rank=[40]*8,actual_scheduler_steps_per_rank=[2]*8,logical_optimizer_writes=40,physical_rank_optimizer_calls=320),current_chain_parent_manifest=parent,parent_failed_attempt=dict(attempt_id=3,completed_outer_credit=0,optimizer_writes=None,published_rank_optimizer_record=False),model_optimizer_scheduler_checkpoint_available=True,automatic_resume_supported=False,exact_executor_resume_verified=False,empty_own_queue_observation=dict(observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),returncode=q.returncode,stderr=q.stderr,stdout=q.stdout))
raw=(json.dumps(proof,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode();path=RUN/'direct-training-predecessor-proof.json';assert not path.exists()
with path.open('xb') as stream:stream.write(raw)
print(json.dumps(dict(proof_path=str(path),proof_sha256=hashlib.sha256(raw).hexdigest(),once_journal_sha256=journal['sha256'],proof=proof),allow_nan=False))
