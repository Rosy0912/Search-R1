import hashlib,json,math,re,time
from pathlib import Path
p=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-instruct300-3154/stage.log')
h=hashlib.sha256();rows=[];prefix_bytes=0
with p.open('rb') as f:
 for raw in f:
  if not raw.endswith(b'\n'):break
  h.update(raw);prefix_bytes+=len(raw);line=raw.decode(errors='replace');m=re.search(r'(?<![\w/])step:(\d+) - ',line)
  if not m:continue
  fields={}
  for field in line[m.end():].split(' - '):
   k,sep,v=field.partition(':')
   try:fields[k]=float(v)
   except ValueError:pass
  if 'training/actual_outer_updates' not in fields:continue
  step=int(m[1]);assert fields['training/actual_outer_updates']==step
  assert fields['actor/scheduler_step_before_outer']==step-1 and fields['actor/scheduler_step_after_outer']==step
  assert math.isclose(fields['actor/lr_before_outer'],1e-6*min((step-1)/57,1),rel_tol=1e-10,abs_tol=1e-18)
  assert math.isclose(fields['actor/lr'],1e-6*min(step/57,1),rel_tol=1e-10,abs_tol=1e-18)
  assert fields['actor/optimizer_steps']==24
  rows.append({'step':step,'fields':fields})
assert [r['step'] for r in rows]==list(range(1,len(rows)+1))
keys=['timing_s/step','timing_s/gen','timing_s/update_actor','timing_s/save_checkpoint','rollout/generated_tokens','rollout/prefill_tokens','rollout/retrieval_calls','rollout/padded_rows','actor/optimizer_steps']
sums={k:sum(r['fields'].get(k,0) for r in rows) for k in keys}
last=rows[-1]['fields']
print(json.dumps({'status':'ACTUAL_LOGGED_COST_SCHEDULER_PREFIX_ONLY','read_epoch':time.time(),'actual_logged_updates':len(rows),'prefix_bytes':prefix_bytes,'prefix_sha256':h.hexdigest(),'once_per_outer_scheduler_and_fixed57_verified':True,'last_lr_before':last['actor/lr_before_outer'],'last_lr_after':last['actor/lr'],'cumulative_logged_metrics':sums,'limitations':'Actor optimizer_steps is a per-rank aggregate metric, not all-rank independent proof. Outer timing excludes initial evaluation, native tests, acceptance gate and service startup; whole-job Slurm cost separate. Token events are not unique prefixes, KV cache savings or FLOPs.'}))
