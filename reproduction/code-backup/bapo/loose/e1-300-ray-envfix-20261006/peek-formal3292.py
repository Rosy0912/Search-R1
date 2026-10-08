"""Read-only actual formal progress; inherited25 and new completed updates remain distinct."""
import pathlib,json,hashlib,time
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-ray-envfix-20261006');run=root/'runs/bapo-instruct300-formal300-3292';source='a2568d0df306ead6e059760e28b5bd53ee3f6876'
assert run==run.resolve() and (run/'source-commit.txt').read_text().strip()==source
out={'job_id':3292,'source_commit':source,'run_root':str(run),'time_unix':time.time(),'inherited_checkpoint_outer':25}
for name in ['recovery-permission-check.json','own-process-cleanup.json','retriever-health.json']:
 p=run/name
 if p.is_file()and p.stat().st_size:
  try:out[name]=json.loads(p.read_text())
  except json.JSONDecodeError:out[name]='incomplete write'
for name in ['phase.log','train.log','retriever.log','resource-watch.jsonl']:
 p=run/name
 if not p.is_file():continue
 if name=='train.log':
  rows=[];decoder=json.JSONDecoder();rank_markers=0;driver_markers=0
  for line in p.open(errors='replace'):
   rank_markers+=line.count('BAPO_RECOVERY_RANK_JSON=');driver_markers+=line.count('BAPO_RECOVERY_DRIVER_JSON=')
   if 'BAPO_TRAIN_METRICS_JSON='in line:
    rows.append(decoder.raw_decode(line.split('BAPO_TRAIN_METRICS_JSON=',1)[1].lstrip())[0])
  out['completed_new_driver_steps']=[x['step']for x in rows]
  out['last_driver_metrics']=rows[-1]if rows else None
  out['rank_restore_markers']=rank_markers;out['driver_restore_markers']=driver_markers
  out['driver_metrics_history']=rows
  out['actual_learning_path_outer']=rows[-1]['step']if rows else 25
  with p.open('rb') as tailfile:tailfile.seek(max(0,p.stat().st_size-3500));out['train.log_tail']=tailfile.read().decode(errors='replace')
 else:
  with p.open('rb')as f:f.seek(max(0,p.stat().st_size-2500));out[name+'_tail']=f.read().decode(errors='replace')
p=root/'logs/slurm-formal300-3292.err';out['stderr_tail']=p.read_text()[-2200:]
out['files']=sorted(p.name for p in run.iterdir())
w=run/'recovery-witness';out['restore_witnesses']={}
if w.is_dir():
 for f in sorted(w.glob('*.json')):
  try:
   v=json.loads(f.read_text());out['restore_witnesses'][f.name]={k:v.get(k) for k in ('status','step','next_step','rank','world_size','target_job_id','scheduler_last_epoch','lr','verify_only','sampling_continuity','target_source_commit','target_run_root')}
  except json.JSONDecodeError:out['restore_witnesses'][f.name]='incomplete write'
watch=run/'resource-watch.jsonl';events=[];samples=[]
if watch.is_file():
 for line in watch.read_text().splitlines():
  try:r=json.loads(line)
  except json.JSONDecodeError:continue
  if 'event' in r:
   if r['event'] not in ('owned_stop_signal','owned_freeze_signal','owned_resume_signal'):
    events.append({k:v for k,v in r.items() if k in ('event','reason','time','elapsed_seconds','reserved_gpu_hours','recovery_seconds','result','error')})
  else:samples.append({k:v for k,v in r.items() if k in ('status','reason','time','action','foreign_gpu_percent','foreign_memory_gib')})
 out['last_unknown_resource_samples']=[{k:v for k,v in r.items() if k in ('reason','error','time','pause')} for r in (json.loads(x) for x in watch.read_text().splitlines() if x.startswith('{') and x.endswith('}')) if r.get('reason')=='monitor_unknown_fail_closed'][-2:]
 out['resource_events']=events;out['last_resource_sample']=samples[-1] if samples else None;out['resource_sample_count']=len(samples)
print(json.dumps(out,allow_nan=False))
