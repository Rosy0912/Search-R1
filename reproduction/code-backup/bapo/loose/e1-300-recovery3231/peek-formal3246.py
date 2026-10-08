"""Read-only actual formal progress; inherited25 and new completed updates remain distinct."""
import pathlib,json,hashlib,time
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-recovery-20261005');run=root/'runs/bapo-instruct300-formal300-3246';source='7de6494bff08f35ffeefbf23ed9a9e1b73e5612e'
assert run==run.resolve() and (run/'source-commit.txt').read_text().strip()==source
out={'job_id':3246,'source_commit':source,'run_root':str(run),'time_unix':time.time(),'inherited_checkpoint_outer':25}
for name in ['recovery-permission-check.json','own-process-cleanup.json','retriever-health.json']:
 p=run/name
 if p.is_file()and p.stat().st_size:
  try:out[name]=json.loads(p.read_text())
  except json.JSONDecodeError:out[name]='incomplete write'
for name in ['phase.log','train.log','retriever.log','resource-watch.jsonl']:
 p=run/name
 if not p.is_file():continue
 if name=='train.log':
  raw=p.read_text(errors='replace');rows=[];decoder=json.JSONDecoder()
  for line in raw.splitlines():
   if 'BAPO_TRAIN_METRICS_JSON='in line:
    rows.append(decoder.raw_decode(line.split('BAPO_TRAIN_METRICS_JSON=',1)[1].lstrip())[0])
  out['completed_new_driver_steps']=[x['step']for x in rows]
  out['last_driver_metrics']=rows[-1]if rows else None
  out['rank_restore_markers']=raw.count('BAPO_RECOVERY_RANK_JSON=');out['driver_restore_markers']=raw.count('BAPO_RECOVERY_DRIVER_JSON=')
  out['actual_learning_path_outer']=rows[-1]['step']if rows else 25
  out['train.log_tail']=raw[-3500:]
 else:
  with p.open('rb')as f:f.seek(max(0,p.stat().st_size-2500));out[name+'_tail']=f.read().decode(errors='replace')
p=root/'logs/slurm-formal300-3246.err';out['stderr_tail']=p.read_text()[-2200:]
out['files']=sorted(p.name for p in run.iterdir())
print(json.dumps(out,allow_nan=False))
