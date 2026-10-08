import json,math,pathlib,re,time
r=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo/runs/bapo-formal500-2948')
lines=(r/'train.log').read_text().splitlines(); rows=[]
for l in lines:
 if 'BAPO_TRAIN_METRICS_JSON=' in l:rows.append(json.loads(l.split('BAPO_TRAIN_METRICS_JSON=',1)[1]))
issues=[]
for x in rows:
 m=x['metrics']; step=x['step']
 if m.get('bapo/actual_outer_updates')!=step:issues.append([step,'outer_mismatch'])
 if m.get('bapo/stage_evaluated_count')!=64:issues.append([step,'stage_count'])
 for k,v in m.items():
  if isinstance(v,(float,int)) and not math.isfinite(v):issues.append([step,k,'nonfinite'])
 for k in ('bapo/mask_binary_errors','bapo/mask_on_observation_tokens','bapo/mask_on_padding_tokens'):
  if m.get(k)!=0:issues.append([step,k,m.get(k)])
final_start=next((i for i,l in enumerate(lines) if 'Final validation dataset size:' in l),None)
batches=[]
if final_start is not None:
 for l in lines[final_start+1:]:
  if 'ACTIVE_TRAJ_NUM:' in l:
   a=json.loads(l.split('ACTIVE_TRAJ_NUM:',1)[1].strip());batches.append(a[0])
final=[json.loads(l.split('SEARCH_R1_FINAL_METRICS_JSON=',1)[1]) for l in lines if 'SEARCH_R1_FINAL_METRICS_JSON=' in l]
print(json.dumps({'time':time.time(),'steps':[x['step'] for x in rows],'contiguous500':[x['step'] for x in rows]==list(range(1,501)),'issues':issues,'total_training_step_seconds':sum(x['metrics']['timing_s/step'] for x in rows),'final_generation_batches_logged':len(batches),'final_generation_rows_logged':sum(batches),'generation_count_scope':'loop completion log before reward/collect_qa; not authoritative final evaluated_count','final_metrics':final,'final_eval_complete':bool(final)}))
