import json,hashlib,os
from pathlib import Path
root=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-recovery300-3234/output/tree-rank-audit');records=[]
for step in json.loads(os.environ['INDEPENDENT_REVIEW_STEPS']):
 for rank in range(8):
  p=root/f'memory-profile/step-{step:06d}-rank-{rank:02d}.json';raw=p.read_bytes();x=json.loads(raw);r=json.loads((root/f'step-{step:06d}-rank-{rank:02d}.json').read_bytes())
  assert (x['outer_step'],x['rank'],x['world_size'],x['job_id'],x['pid'],x['code_commit'],x['config_sha256'])==(step,rank,8,'3234',r['pid'],r['code_commit'],r['config_sha256'])
  assert len(x['micro_records'])==240 and len(x['optimizer_records'])==10
  assert [z['pending_microbatches'] for z in x['micro_records']]==list(range(1,25))*10
  assert all(z['pending_after_clear']==0 for z in x['optimizer_records'])
  records.append(dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),step=step,rank=rank,pid=x['pid'],observer_sha256=x['observer_sha256'],micro_records=240,optimizer_clears=10,all_pending_cleared=True,start=x['start'],end=x['end'],retained_unique_max_bytes=max(z['unique_retained_cuda_bytes'] for z in x['micro_records']),observer_cpu_seconds=x['observer_cpu_seconds'],observer_wall_seconds=x['observer_wall_seconds']))
print(json.dumps(dict(status='PASS_INCREMENTAL_PROFILE_ONLY',records=records)))
