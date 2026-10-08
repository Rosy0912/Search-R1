import json,re,time
from pathlib import Path
p=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-five-2956');log=p/'stage.log'
steps=[]
if log.exists():
 for line in log.read_text().splitlines():
  m=re.search(r'(?<![\w/])step:(\d+) - ',line)
  if not m:continue
  fields={}
  for field in line[m.end():].split(' - '):
   k,sep,v=field.partition(':')
   try:fields[k]=float(v)
   except ValueError:continue
  if 'training/actual_outer_updates' in fields:steps.append(dict(step=int(m[1]),wall=fields.get('timing_s/step'),gen=fields.get('timing_s/gen'),actor=fields.get('timing_s/update_actor'),grad=fields.get('actor/grad_norm'),calls=fields.get('actor/optimizer_steps'),generated=fields.get('rollout/generated_tokens'),queries=fields.get('rollout/retrieval_calls')))
result=dict(epoch=time.time(),steps=steps,rank_files=len(list((p/'output/tree-rank-audit').glob('*.json'))))
if (p/'run-result.json').exists():result['run_result']=json.loads((p/'run-result.json').read_text())
if (p/'resource-samples.jsonl').exists():
 last=json.loads((p/'resource-samples.jsonl').read_text().splitlines()[-1]);result['guard_event']=last.get('event');result['guard_sample_age_seconds']=time.time()-last['epoch'];result['foreign_memory_max']=max((r['foreign_memory_fraction'] for r in last.get('gpus',[])),default=None);result['foreign_sm_max']=max((r['foreign_sm_upper'] for r in last.get('gpus',[])),default=None);result['sampled_gpu_memory_max_gib']=max((r['total_used_memory_bytes']/1024**3 for r in last.get('gpus',[])),default=None)
print(json.dumps(result),flush=True)
