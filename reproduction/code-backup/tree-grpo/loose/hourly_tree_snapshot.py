import json,re,time,math,hashlib
from pathlib import Path
p=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-formal-2971');raw=(p/'stage.log').read_text();steps=[]
for line in raw.splitlines():
 m=re.search(r'(?<![\w/])step:(\d+) - ',line)
 if not m:continue
 fields={}
 for field in line[m.end():].split(' - '):
  k,sep,v=field.partition(':')
  try:fields[k]=float(v)
  except ValueError:continue
 if 'training/actual_outer_updates' in fields:steps.append(dict(step=int(m[1]),metrics=fields))
n=steps[-1]['step'];ranks=[]
for f in sorted((p/'output/tree-rank-audit').glob('*.json')):
 x=json.loads(f.read_text())
 if x['outer_step']<=n:
  assert x['job_id']=='2971' and x['code_commit']=='4d133efeb1f675139d2558640f1e21d50780f133'
  assert x['successful_optimizer_calls']==x['finite_grad_norms']==x['nonzero_grad_norms']==12 and x['skipped_empty_minibatches']==0
  assert all(math.isfinite(v) and v>0 for v in x['existing_grad_norms'])
  ranks.append(x)
assert {(x['outer_step'],x['rank']) for x in ranks}=={(s,r) for s in range(1,n+1) for r in range(8)}
guard=[json.loads(v) for v in (p/'resource-samples.jsonl').read_text().splitlines()];last=guard[-1];peak=max(g['total_used_memory_bytes'] for s in guard for g in s.get('gpus',[]))/2**30
latest=[x for x in ranks if x['outer_step']==n]
current=re.findall(r'epoch (\d+), step (\d+)',raw)
q=p/'output/checkpoints/tree_grpo-qwen2.5-3b/actor'
result=dict(epoch=time.time(),run=str(p),actual_outer_updates=n,current_epoch_step=current[-1] if current else None,rank_records_complete=len(ranks),last_three_steps=steps[-3:],periodic_metric_steps=[{'step':s['step'],'metrics':{k:v for k,v in s['metrics'].items() if k.startswith('val/')}} for s in steps if any(k.startswith('val/') for k in s['metrics'])],last_three_step_wall_mean=sum(x['metrics']['timing_s/step'] for x in steps[-3:])/len(steps[-3:]),latest_rank_nonzero_sample_counts={str(x['rank']):x['local_shard_sample_delta']['nonzero'] for x in latest},all_completed_rank_calls_finite_nonzero=True,all_console_metrics_finite=all(math.isfinite(v) for s in steps for v in s['metrics'].values()),guard_samples=len(guard),guard_events=[s.get('event') for s in guard if s.get('event')],guard_age_seconds=time.time()-last['epoch'],sampled_gpu_peak_gib=peak,foreign_memory_fraction_peak=max(g['foreign_memory_fraction'] for s in guard for g in s.get('gpus',[])),foreign_sm_peak=max(g['foreign_sm_upper'] for s in guard for g in s.get('gpus',[])),checkpoint_steps=sorted(int(d.name.split('_')[-1]) for d in q.glob('global_step_*') if d.is_dir()),traceback_present='Traceback (most recent call last)' in raw,validation_reports=[json.loads(f.read_text()) for f in sorted((p/'output/validation-cost').glob('*.json'))],selected_latest_quality_lines=[line for line in raw.splitlines() if 'EVALUATION_METRICS_JSON=' in line or 'SEARCH_R1_FINAL_METRICS_JSON=' in line][-2:])
print(json.dumps(result,ensure_ascii=False),flush=True)
