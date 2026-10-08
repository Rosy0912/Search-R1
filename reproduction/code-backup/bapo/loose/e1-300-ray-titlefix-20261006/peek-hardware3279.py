import json
from pathlib import Path
r=Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-ray-titlefix-20261006');run=r/'runs/bapo-instruct300-hardware-3279';out={'job_id':3279,'source':'484c0b30fa73fa5354bdb3452f007134ec5db13f'}
for n in ['recovery-permission-check.json','identity-linux-fixture/result.json','recovery-witness/driver.json','recovery-witness/driver-reload.json']:
 p=run/n
 if p.exists():
  try:
   v=json.loads(p.read_text());out[n]={k:v[k] for k in ['status','source_commit','job_id','verify_only','step','next_step','foreign_signals','foreign_exit_code','actual_ray_actor_executed','gpu_initialized']if k in v}
   if 'records'in v:out[n]['records_count']=len(v['records'])
  except ValueError:out[n]='PRESENT_NOT_COMPLETE_JSON'
for n in ['identity-linux-fixture.log','hardware.log']:
 p=run/n
 if p.exists():out[n]={'bytes':p.stat().st_size,'tail':p.read_text(errors='replace')[-1200:]}
for n in ['slurm-hardware-3279.err']:
 p=r/'logs'/n
 if p.exists():out[n]={'bytes':p.stat().st_size,'tail':p.read_text(errors='replace')[-1200:]}
print(json.dumps(out,ensure_ascii=False))
