"""Read-only actual recovery cost prefix; no final or utilization claim."""
import json,re,time,hashlib
from pathlib import Path
root=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-recovery300-3234')
steps=[];h=hashlib.sha256();lines=0
with (root/'stage.log').open('rb') as f:
 for raw in f:
  if not raw.endswith(b'\n'):break
  h.update(raw);lines+=1
  line=re.sub(r'\x1b\[[0-9;]*m','',raw.decode());m=re.search(r'(?<![\w/])step:(\d+) - ',line)
  if not m:continue
  fields={}
  for item in line[m.end():].split(' - '):
   k,sep,v=item.partition(':')
   if sep:
    try:fields[k]=float(v)
    except ValueError:continue
  if 'training/actual_outer_updates' in fields:steps.append((int(m[1]),fields))
assert [n for n,x in steps]==list(range(101,101+len(steps)))
assert all(x['actor/optimizer_steps']==10 and x['training/actual_outer_updates']==n and x['actor/scheduler_step_after_outer']==n and x['actor/scheduler_step_before_outer']==n-1 for n,x in steps)
keys=['timing_s/step','timing_s/gen','timing_s/ref','timing_s/update_actor','rollout/generated_tokens','rollout/prefill_tokens','rollout/retrieval_calls']
resource_hash=hashlib.sha256();samples=[]
with (root/'resource-samples.jsonl').open('rb') as f:
 for raw in f:
  if not raw.endswith(b'\n'):break
  resource_hash.update(raw);samples.append(json.loads(raw))
assert samples and all(not x.get('event') and len(x['gpus'])==8 for x in samples)
assert all(g['foreign_memory_fraction']<=.15 and g['foreign_sm_upper']<=15 for x in samples for g in x['gpus'])
peak=max((g['total_used_memory_bytes']/1024**3,g['total_used_memory_bytes']/g['total_memory_bytes']) for x in samples for g in x['gpus'])
print(json.dumps(dict(epoch=time.time(),job_id=3234,new_outer_count=len(steps),completed_boundary=100+len(steps),optimizer_calls_per_rank=10*len(steps),totals={k:sum(x.get(k,0) for n,x in steps) for k in keys},log_complete_lines=lines,log_prefix_sha256=h.hexdigest(),resource_complete_samples=len(samples),resource_prefix_sha256=resource_hash.hexdigest(),guard_sample_age_seconds=time.time()-samples[-1]['epoch'],sampled_whole_gpu_peak_gib=peak[0],sampled_whole_gpu_peak_fraction=peak[1],guard_prefix_events=0,scope='Actual successful outer log totals and 5-second whole-GPU sample prefix; startup/restored eval/acceptance/CPU reviews excluded from outer totals; no final cost, no unique token/FLOP or sufficient memory-headroom claim')),flush=True)
