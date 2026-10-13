"""Current own training progress and error evidence only; no tensor downloads."""
import importlib.util,json,shlex,sys
from pathlib import Path
spec=importlib.util.spec_from_file_location('ssh',Path(__file__).with_name('ssh_retained18.py'))
ssh=importlib.util.module_from_spec(spec);spec.loader.exec_module(ssh)
code='''import json,subprocess,time,sys
from pathlib import Path
r=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo');source='0befbaf99fc44942105831b3fbec94e4bf1b8ccb';unit='sst-tree-nested-branch-fresh200-'+source[:8]+'.service';job='not-slurm;systemd:'+source;run=r/'runs'/('tree-nested-branch-fresh200-'+job);tools=r/'instruct300/candidates'/('nested-branch-fresh200-'+source)
def tail(p,n=3500):
 if not p.exists():return None
 with p.open('rb') as f:f.seek(0,2);f.seek(max(0,f.tell()-n));return f.read().decode(errors='replace')
def read(p):return json.loads(p.read_text()) if p.is_file() else None
sys.path.insert(0,str(tools));from train_monitor import progress_evidence
sample=tail(run/'resource-samples.jsonl',32768)
lines=sample.splitlines() if sample else []
guard=json.loads(lines[-1]) if lines and sample.endswith('\\n') else json.loads(lines[-2]) if len(lines)>1 else None
rankpaths=sorted((run/'output/tree-rank-audit').glob('step-*-attempt-*-rank-*.json'))
last=[read(p) for p in rankpaths[-8:]]
rank_summary=[{k:v for k,v in row.items() if k in ('outer_step','attempt_id','rank','successful_optimizer_calls','skipped_optimizer_calls','opportunities_processed','scheduler_step','learning_rate','finite_grad_norms','nonzero_grad_norms')} for row in last]
dev=[]
for p in (run/'output/validation-cost').glob('validation-*.json'):
 d=read(p)
 if d:dev.append(dict(path=p.name,outer=d.get('actual_completed_outer'),phase=d.get('phase'),status=d.get('status'),rows=d.get('evaluated_rows'),metrics=d.get('metrics')))
startup=[dict(path=str(p),value=read(p)) for p in (r/'instruct300/startup-audits').glob('*'+source+'*') if p.is_file()]
d=dict(epoch=time.time(),node='h100-10',source_commit=source,unit=unit,job_id=job,run=str(run),systemd=subprocess.check_output(['systemctl','show',unit,'--property=LoadState,ActiveState,SubState,MainPID,ExecMainStatus,MemoryCurrent,CPUUsageNSec'],text=True),progress=progress_evidence(run),direct_rank_records=len(rankpaths),last_rank_summaries=rank_summary,latest_guard=guard,logs={n:tail(run/n) for n in ('stage.log','retriever.log','guard.log','ray.log')},run_result=read(run/'run-result.json'),development=sorted(dev,key=lambda x:x['outer'] or 0)[-1:] if dev else [],startup=startup,journal=subprocess.check_output(['journalctl','-u',unit,'-n','12','--no-pager','-o','cat'],text=True),gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True),native_manifests=[str(p) for p in (run/'checkpoints/training_state').glob('*/manifest.json')],own_output_files={p.name:p.stat().st_size for p in run.glob('*') if p.is_file()})
d['output_metadata']=[dict(path=str(p.relative_to(run)),bytes=p.stat().st_size,mtime=p.stat().st_mtime) for p in (run/'output').glob('*/*') if p.is_file()]
d['rollout_cost']=[read(p) for p in sorted((run/'output/training-rollout-cost').glob('attempt-*.json'))[-1:]]
with (run/'stage.log').open(errors='replace') as stream:
 d['first_rollout_entered']=any('epoch 0, step 1' in line for line in stream)
print(json.dumps(d))
'''
r=ssh.run('env CUDA_VISIBLE_DEVICES= python3 -c '+shlex.quote(code),timeout=60)
if r.returncode:raise RuntimeError(r.stdout+r.stderr)
d=json.loads(r.stdout)
Path(sys.argv[1]).write_text(json.dumps(d,indent=2)+'\n')
print(json.dumps(dict(epoch=d['epoch'],systemd=d['systemd'],progress=d['progress'],direct_rank_records=d['direct_rank_records'],run_result=d['run_result'],journal=d['journal'][-1600:],stage_log=d['logs']['stage.log'])))
