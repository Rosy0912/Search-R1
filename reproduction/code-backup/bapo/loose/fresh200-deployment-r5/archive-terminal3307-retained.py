from pathlib import Path
import hashlib,json,os,subprocess,tarfile
R=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-safety-r5-20261006');run=R/'runs/formal200-6a29642035ff40b2b7787de206f886c0-3307';source='aec5e057093b4c491bfb0a47abd7605d98981b56'
assert os.uname().nodename=='h100-5' and (R/'source-expected.txt').read_text().strip()==source and run.is_dir()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
observed=subprocess.run(['scontrol','show','job','3307'],text=True,capture_output=True)
slurm='JobId=3307 JobName=bapo-fresh200-formal200 UserId=root(0) GroupId=root(0) MCS_label=N/A Priority=4294898759 Nice=0 Account=(null) QOS=(null) JobState=FAILED Reason=NonZeroExitCode Dependency=(null) Requeue=0 Restarts=0 BatchFlag=1 Reboot=0 ExitCode=1:0 RunTime=00:16:41 TimeLimit=UNLIMITED TimeMin=N/A SubmitTime=2026-10-06T20:37:39 EligibleTime=2026-10-06T20:37:39 AccrueTime=2026-10-06T20:37:39 StartTime=2026-10-06T20:37:39 EndTime=2026-10-06T20:54:20 Deadline=N/A SuspendTime=None SecsPreSuspend=0 LastSchedEval=2026-10-06T20:37:39 Scheduler=Main Partition=gpu AllocNode:Sid=h100-16:3744832 ReqNodeList=h100-5 ExcNodeList=(null) NodeList=h100-5 BatchHost=h100-5 NumNodes=1 NumCPUs=192 NumTasks=1 CPUs/Task=32 ReqB:S:C:T=0:0:*:* TRES=cpu=192,mem=512G,node=1,billing=192 Socks/Node=* NtasksPerN:B:S:C=0:0:*:* CoreSpec=* MinCPUsNode=32 MinMemoryNode=512G MinTmpDiskNode=0 Features=(null) DelayBoot=00:00:00 OverSubscribe=NO Contiguous=0 Licenses=(null) Network=(null) Command=/dataset/shaoshitong/search-r1-rl/bapo-fresh200-safety-r5-20261006/tools/formal200.sbatch WorkDir=/dataset/shaoshitong/search-r1-rl/bapo-fresh200-safety-r5-20261006 Comment=BAPO_FRESH200:formal200:aec5e057093b4c491bfb0a47abd7605d98981b56:6a29642035ff40b2b7787de206f886c0  StdErr=/dataset/shaoshitong/search-r1-rl/bapo-fresh200-safety-r5-20261006/logs/slurm-formal200-3307.err StdIn=/dev/null StdOut=/dataset/shaoshitong/search-r1-rl/bapo-fresh200-safety-r5-20261006/logs/slurm-formal200-3307.out Power= TresPerNode=gres:gpu:h100:8 \n'
assert 'JobState=FAILED' in slurm and 'ExitCode=1:0' in slurm
cleanup=json.loads((run/'own-process-cleanup.json').read_text());assert cleanup['remaining']==[] and cleanup['own_marker']==str(run)
record=R/'logs/diagnostic3307-terminal-record.json';assert not record.exists()
accounting=subprocess.run(['sacct','-j','3307','--format=JobID,State,ExitCode,Elapsed,Start,End,NodeList','-P'],text=True,capture_output=True)
record.write_text(json.dumps({'scontrol':slurm,'scontrol_provenance':{'scope':'PREVIOUS_ACTUAL_TERMINAL_HEALTH7_NOT_FRESH_SLURM_QUERY','local_evidence_sha256':'6abff9dfd77e59a3e13bd0388daa45bb8721c779e5fd9cc6f16823c9114edf25'},'current_scontrol':{'exit_code':observed.returncode,'stdout':observed.stdout,'stderr':observed.stderr},'sacct':{'exit_code':accounting.returncode,'stdout':accounting.stdout,'stderr':accounting.stderr},'source_commit':source,'run_root':str(run),'formal_outer':0,'actual_adam_writes_per_rank':0,'reserved_gpu_seconds':1001*8,'reserved_gpu_hours':1001*8/3600,'cleanup':cleanup},indent=2))
files=[p for p in run.iterdir() if p.is_file() and not p.is_symlink()]
files += [p for p in (run/'diagnostics').rglob('*') if p.is_file() and not p.is_symlink() and p.suffix!='.pt'] if (run/'diagnostics').exists() else []
files += [R/'logs'/n for n in ['slurm-formal200-3307.out','slurm-formal200-3307.err','launch-approval.json','tools-manifest.json','target-cpu-gate.json','context-budget-audit.json']]
files += [R/'logs/runtime-tests.actual.json',R/'logs/runtime-tests.actual.log',R/'logs/runtime-executable.json',R/'logs/frozen-policy-safety-config.json',R/'logs/user-fixed-policy-thresholds.json'];files.append(record);assert all(p.is_file() and not p.is_symlink() for p in files)
files += [p for p in (run/'checkpoints/diagnostics').rglob('*') if p.is_file() and not p.is_symlink() and p.suffix!='.pt']
tensor_hashes=[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in (run/'diagnostics').rglob('*.pt')] if (run/'diagnostics').exists() else []
tensor_hashes += [{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in (run/'checkpoints/diagnostics').rglob('*.pt')]
manifest=R/'logs/diagnostic3307-terminal-archive-manifest.json';assert not manifest.exists();manifest.write_text(json.dumps({'files':{str(p.relative_to(R)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in files},'remote_retained_initial_tensors':tensor_hashes,'runtime_executable_file_exists':(R/'logs/runtime-executable.json').exists(),'scope':'RAW_FAILED_SAMPLING_PROBABILITY_ARCHIVE_NOT_NATIVE_TRAINING_ACCEPTANCE'},indent=2))
archive=R/'logs/diagnostic3307-terminal-raw.tar.gz';assert not archive.exists()
with tarfile.open(archive,'x:gz') as t:
 for p in files+[manifest]:t.add(p,arcname=str(p.relative_to(R)),recursive=False)
print(json.dumps({'path':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),'files':len(files)+1,'retained_tensor_hashes':len(tensor_hashes)}))
