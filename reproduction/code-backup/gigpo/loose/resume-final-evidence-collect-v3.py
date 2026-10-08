"""Read-only freeze of the completed original200->300 run, excluding model/data files."""
from pathlib import Path
import json,subprocess,hashlib,base64,gzip,datetime,os
ROOT=Path('/mnt/data/shaoshitong/search-r1-rl/gigpo');RUN=ROOT/'runs/gigpo-resume200to300-job3130-1791086878335123421'
def command(args):
 p=subprocess.run(args,capture_output=True,text=True);return dict(args=args,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
result=json.loads((RUN/'run-result.json').read_text());assert result['status']=='CHECKPOINT_RELOADED_UNREVIEWED' and result['cleanup_residual_states']=={}
scheduler=command(['scontrol','show','job','3130'])
terminal_path=ROOT/'logs/resume3130-terminal-scheduler-observation-v1.json'
terminal_bytes=terminal_path.read_bytes()
assert hashlib.sha256(terminal_bytes).hexdigest()=='b93203f0f8ac9a36b4fca185c572701990086faf20844cd5b89ca3ee00791136'
terminal=json.loads(terminal_bytes)
assert terminal['fields']['JobId']=='3130' and terminal['fields']['NodeList']=='h100-8' and terminal['fields']['JobState']=='COMPLETED' and terminal['fields']['ExitCode']=='0:0'
assert terminal['scheduler']['returncode']==0
assert json.loads(terminal['run_result_text'])==result
# Live scheduler may expire; the SHA-bound raw terminal observation is authoritative.
paths=set(RUN.glob('*.json'))|set(RUN.glob('*.jsonl'))|set((RUN/'audit').rglob('*.jsonl'))|set(RUN.glob('*.yaml'))
paths.add(RUN/'audit/entry-config.yaml')
paths.add(terminal_path)
paths.update(RUN/name for name in ['retriever.log','training.log','guard.log','checkpoint205-cpu-verification.log','checkpoint-cpu-verification.log','ray.log'])
for step in (205,225,250,275,300):
 paths.add(RUN/'checkpoints/actor'/f'global_step_{step}'/'training_state/manifest.json')
 paths.add(RUN/'checkpoints/algorithm'/f'global_step_{step}'/'state.json')
 paths.add(ROOT/'logs'/f'resume3130-checkpoint{step}-cpu-state-verification-v1.json') if step!=205 else None
paths.update(ROOT/'logs'/name for name in ['resume3130-205-independent-review.json','resume-original200-submission-v2.json','deployment-binding-resume-60a472b-v2.json','resume-header-deployment-independent-review-v2.json','resume-checkpoint-inspector-independent-review-v3.json','resume-original200-3130.log','original2992-checkpoint200-resume-inventory.json','source-activation-resume-60a472b-v1.json'])
for p in (RUN/'ray').rglob('worker-*.out'):
 if b'SEARCH_R1_RECOVERY_COMPLETE_JSON=' in p.read_bytes():paths.add(p)
raw={};missing=[]
for p in sorted(paths):
 if not p.is_file():missing.append(str(p));continue
 b=p.read_bytes();raw[str(p)]=dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),base64=base64.b64encode(b).decode())
launch=json.loads((RUN/'launch-manifest.json').read_text());scope=Path(launch['cgroup'])
assert str(scope)=='/sys/fs/cgroup/freezer/slurm_h100-8/uid_0/job_3130/step_0'
try:
 os.stat(scope);scope_absent=False
except FileNotFoundError:
 scope_absent=True
assert scope_absent,'Own Slurm step still exists: wait for removal'
parent=Path('/sys/fs/cgroup/freezer/slurm_h100-8')
os.stat(parent)
parent_entries=os.listdir(parent) # prove actual readable Slurm mount ancestor
assert 'uid_0' not in parent_entries or not scope.exists()
baseline_bytes=(ROOT/'logs/resume3130-205-evidence-v1.json').read_bytes();assert hashlib.sha256(baseline_bytes).hexdigest()=='8d6d656832c265e4f403250643e2f5d0bac5f5d906660cf043b79e16e2a778b6'
ready=json.loads(baseline_bytes)['ready'];workers=[(int(x['pause_identity']['pid']),int(x['pause_identity']['start_ticks'])) for x in ready['identity']['files'] if 'pause_identity' in x];assert len(workers)==8 and len({pid for pid,tick in workers})==8
worker_exit=[]
for pid,tick in workers:
 try:
  stat=Path('/proc')/str(pid)/'stat';line=stat.read_text();actual=int(line[line.rfind(')')+2:].split()[19]);same=actual==tick
 except FileNotFoundError:
  actual=None;same=False
 assert not same,'Own training worker remains live'
 worker_exit.append(dict(pid=pid,expected_start_ticks=tick,current_start_ticks=actual,same_worker_live=same))
obj=dict(cleanup_after_batch=dict(cgroup=str(scope),cgroup_absent=scope_absent,readable_parent=str(parent),parent_entries=parent_entries,workers=worker_exit),captured_at=datetime.datetime.now().isoformat(),run=str(RUN),job_id='3130',source=str(ROOT/'runs/gigpo-formal500-job2992-1791016340597739432/checkpoints/actor/global_step_200'),raw_files=raw,missing_optional_paths=missing,live_scheduler=scheduler,live_queue=command(['squeue','-j','3130','-o','%.18i %.9T %.10M %.20R']),gpu=command(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader']),code_head=command(['git','-C',str(ROOT/'repo'),'rev-parse','HEAD']),code_status=command(['git','-C',str(ROOT/'repo'),'status','--porcelain']))
p=ROOT/'logs/resume3130-final-evidence-v3.json.gz';assert not p.exists(),'Existing final evidence must be reused'
blob=gzip.compress(json.dumps(obj,indent=2).encode(),mtime=0);p.write_bytes(blob);print(json.dumps(dict(path=str(p),bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest(),raw_count=len(raw),missing_optional_paths=missing)))
