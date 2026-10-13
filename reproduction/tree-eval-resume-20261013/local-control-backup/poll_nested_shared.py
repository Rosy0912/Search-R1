"""Current own training progress and small restoration receipts; no tensor downloads."""
import argparse, importlib.util, json, os, re, shlex
from pathlib import Path

LEGACY_SOURCE = '0befbaf99fc44942105831b3fbec94e4bf1b8ccb'
NATIVE1_SOURCE = 'ba9f6627ec9a4e9749830d7b6acca53f6feec6b4'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output')
    parser.add_argument('--source', '--source-commit', dest='source', default=os.environ.get('NESTED_SOURCE_COMMIT', LEGACY_SOURCE))
    parser.add_argument('--unit')
    parser.add_argument('--run-directory')
    parser.add_argument('--tools-directory')
    parser.add_argument('--source-checkout')
    parser.add_argument('--source-branch')
    parser.add_argument('--native1-continuation', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'[0-9a-f]{40}', args.source):
        parser.error('source must be a full 40-character Git SHA')
    native = args.native1_continuation or args.source == NATIVE1_SOURCE
    root = '/dataset/shaoshitong/search-r1-rl/tree-grpo'
    job = 'not-slurm;systemd:' + args.source
    config = dict(source=args.source, unit=args.unit or 'sst-tree-nested-branch-fresh200-' + args.source[:8] + '.service', job=job,
        run=args.run_directory or root + '/runs/tree-nested-branch-fresh200-' + job,
        tools=args.tools_directory or root + '/instruct300/candidates/nested-branch-fresh200-' + args.source,
        source_checkout=str(Path(args.source_checkout or ('work/tree-speed-repair-20261008/nested-native1-recovery-source' if native else 'work/tree-speed-repair-20261008/nested-branch-source')).resolve()),
        source_branch=args.source_branch or ('codex/tree-nested-native1-recovery-20261014' if native else 'codex/tree-nested-branch-20261012'),
        native1_continuation_requested=native, experiment_origin='fresh')
    spec = importlib.util.spec_from_file_location('ssh', Path(__file__).with_name('ssh_retained18.py'))
    ssh = importlib.util.module_from_spec(spec); spec.loader.exec_module(ssh)
    code = r'''import json,subprocess,time,sys,re
from pathlib import Path
config=__CONFIG__
r=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo');source=config['source'];unit=config['unit'];job=config['job'];run=Path(config['run']);tools=Path(config['tools'])
def tail(p,n=3500):
 if not p.exists():return None
 with p.open('rb') as f:f.seek(0,2);f.seek(max(0,f.tell()-n));return f.read().decode(errors='replace')
def read(p):return json.loads(p.read_text()) if p.is_file() else None
sys.path.insert(0,str(tools));from train_monitor import progress_evidence
progress=progress_evidence(run)
receipt=read(run/'output/nested-native1-loaded.json')
native_summary=None
if receipt is not None:
 worker_fields=('status','protocol','rank','world_size','parent_identity','parent_manifest_sha256','state_path','state_file_sha256','state_file_bytes','actual_file_sha256_verified','serialized_logical_state_sha256','actual_loaded_logical_state_sha256','actual_cold_native_state_equal','adam_steps','scheduler_step','actual_total_optimizer_calls','child_code_commit','executor_equivalence','exact_executor_resume')
 native_summary={k:receipt.get(k) for k in ('status','protocol','provenance','driver','executor_equivalence','exact_executor_resume','initial_evaluation_added')}
 native_summary['workers']=[{k:v for k,v in worker.items() if k in worker_fields} for worker in receipt.get('workers',[])]
 native_summary['worker_count']=len(native_summary['workers'])
 native_summary['worker_ranks']=sorted(w.get('rank') for w in native_summary['workers'])
 native_summary['receipt_path']=str(run/'output/nested-native1-loaded.json')
 native_summary['receipt_sha256']=(progress.get('inherited_native1') or {}).get('receipt_sha256')
sample=tail(run/'resource-samples.jsonl',32768);lines=sample.splitlines() if sample else []
guard=json.loads(lines[-1]) if lines and sample.endswith('\n') else json.loads(lines[-2]) if len(lines)>1 else None
rankpaths=sorted((run/'output/tree-rank-audit').glob('step-*-attempt-*-rank-*.json'));last=[read(p) for p in rankpaths[-8:]]
rank_summary=[{k:v for k,v in row.items() if k in ('outer_step','attempt_id','rank','successful_optimizer_calls','skipped_optimizer_calls','skipped_empty_minibatches','opportunities_processed','opportunities_skipped','scheduler_step','learning_rate','finite_grad_norms','nonzero_grad_norms','local_shard_sample_delta')} for row in last]
dev=[]
for p in (run/'output/validation-cost').glob('validation-*.json'):
 value=read(p)
 if value:dev.append(dict(path=p.name,outer=value.get('actual_completed_outer'),phase=value.get('phase'),status=value.get('status'),rows=value.get('evaluated_rows'),metrics=value.get('metrics')))
startup=[dict(path=str(p),value=read(p)) for p in (r/'instruct300/startup-audits').glob('*'+source+'*') if p.is_file()]
d=dict(epoch=time.time(),node='h100-10',source_commit=source,source_checkout=config['source_checkout'],source_branch=config['source_branch'],experiment_origin=config['experiment_origin'],native1_continuation_requested=config['native1_continuation_requested'],native1_load_receipt=native_summary,unit=unit,job_id=job,run=str(run),systemd=subprocess.check_output(['systemctl','show',unit,'--property=LoadState,ActiveState,SubState,MainPID,ExecMainStatus,MemoryCurrent,MemoryMax,CPUUsageNSec,Result'],text=True),progress=progress,direct_rank_records=len(rankpaths),last_rank_summaries=rank_summary,latest_guard=guard,logs={n:tail(run/n) for n in ('stage.log','retriever.log','guard.log','ray.log')},run_result=read(run/'run-result.json'),development=sorted(dev,key=lambda x:x['outer'] or 0)[-1:] if dev else [],startup=startup,journal=subprocess.check_output(['journalctl','-u',unit,'-n','12','--no-pager','-o','cat'],text=True),gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True),own_output_files={p.name:p.stat().st_size for p in run.glob('*') if p.is_file()})
d['output_metadata']=[dict(path=str(p.relative_to(run)),bytes=p.stat().st_size,mtime=p.stat().st_mtime) for p in (run/'output').glob('*/*') if p.is_file()]
d['rollout_cost']=[read(p) for p in sorted((run/'output/training-rollout-cost').glob('attempt-*.json'))[-1:]]
d['first_rollout_entered']=False;iteration_metrics=[]
if (run/'stage.log').is_file():
 with (run/'stage.log').open(errors='replace') as stream:
  for line in stream:
   if re.search(r'epoch [0-9]+, step [0-9]+',line):d['first_rollout_entered']=True
   if 'timing_s/step:' in line:
    iteration_metrics.append({k:float(v) for k,v in re.findall(r'([a-zA-Z0-9_./-]+):(-?[0-9]+(?:[.][0-9]+)?(?:[eE][+-]?[0-9]+)?)',line)})
d['iteration_metrics']=iteration_metrics[-3:]
d['native_manifests']=[str(p) for p in (run/'checkpoints/training_state').rglob('manifest.json')]
print(json.dumps(d))
'''.replace('__CONFIG__', repr(config))
    response = ssh.run('env CUDA_VISIBLE_DEVICES= python3 -c ' + shlex.quote(code), timeout=60)
    if response.returncode:
        raise RuntimeError(response.stdout + response.stderr)
    data = json.loads(response.stdout)
    Path(args.output).write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps(dict(epoch=data['epoch'], systemd=data['systemd'], progress=data['progress'], native1_load_receipt=data['native1_load_receipt'], direct_rank_records=data['direct_rank_records'], run_result=data['run_result'], journal=data['journal'][-1600:], stage_log=data['logs']['stage.log'])))


if __name__ == '__main__':
    main()
