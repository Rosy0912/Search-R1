from pathlib import Path
import json,math,hashlib
O=Path(__file__).resolve().parent;P=O/'actual3154-files/runs/tree-instruct300-3154';cfg=hashlib.sha256((P/'resolved.yaml').read_bytes()).hexdigest();rows=[];peak=0;maxret=0;maxlogical=0;wall=0
for step in [1,2]:
 for rank in range(8):
  j=json.loads((P/f'output/tree-rank-audit/memory-profile/step-{step:06d}-rank-{rank:02d}.json').read_text());a=json.loads((P/f'output/tree-rank-audit/step-{step:06d}-rank-{rank:02d}.json').read_text())
  assert j['kind']=='tree_detached_storage_observer' and j['outer_step']==step and j['rank']==rank and j['world_size']==8 and j['job_id']=='3154' and j['pid']==a['pid'] and j['config_sha256']==cfg and j['code_commit']=='f994af33c6c93b6b9844863be912cab2a4020dc1' and j['observer_sha256']=='6b7e3cb0ae998b0248323efe1a8dd1815e81985412fdbf087134752f417265a0'
  assert len(j['micro_records'])==192 and len(j['optimizer_records'])==24 and [m['pending_microbatches'] for m in j['micro_records']]==list(range(1,9))*24 and all(v['pending_after_clear']==0 for v in j['optimizer_records'])
  mems=[j['start'],j['end']]+[m['memory'] for m in j['micro_records']]+[v[k] for v in j['optimizer_records'] for k in ['before_clear','after_clear']]
  for m in mems:
   assert m['cuda_available'] and all(math.isfinite(m[k]) and m[k]>=0 for k in ['allocated','reserved','lifetime_peak_allocated','lifetime_peak_reserved']) and m['reserved']>=m['allocated'] and m['lifetime_peak_allocated']>=m['allocated'] and m['lifetime_peak_reserved']>=m['reserved']
  assert math.isfinite(j['observer_wall_seconds']) and j['observer_wall_seconds']>=0 and math.isfinite(j['observer_cpu_seconds']) and j['observer_cpu_seconds']>=0
  peak=max(peak,max(v['lifetime_peak_allocated'] for v in mems));maxret=max(maxret,max(m['unique_retained_cuda_bytes'] for m in j['micro_records']));maxlogical=max(maxlogical,max(m['logical_cuda_tensor_bytes'] for m in j['micro_records']));wall=max(wall,j['observer_wall_seconds'])
  rows.append({'step':step,'rank':rank,'pid':j['pid'],'micro_records':192,'clears':24,'observer_wall_seconds':j['observer_wall_seconds']})
result={'status':'PASS_TWO_PREFIX16_PROFILE_RECORDS_ONLY','job_id':'3154','records':rows,'max_unique_retained_cuda_bytes':maxret,'max_logical_tensor_bytes':maxlogical,'max_lifetime_torch_allocated_GiB':peak/2**30,'max_recorded_micro_observer_wall_seconds':wall,'limits':['Backing storage can already be owned elsewhere, not necessarily extra exclusive allocation','Torch allocator excludes FAISS/NVML and other processes','lifetime high-water marks have not been reset','no matched 3B on-off control or failure path capture proven here']}
(O/'TWO_PROFILE_RECOMPUTED.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='records'}))
