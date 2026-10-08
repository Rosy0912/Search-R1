import json,hashlib,math,time
from pathlib import Path
p=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-instruct300-3154/output/tree-rank-audit')
records=[];issues=[];previous={}
for step in range(1,76):
 for rank in range(8):
  f=p/f'step-{step:06d}-rank-{rank:02d}.json'
  b=f.read_bytes();x=json.loads(b);d=x['local_shard_sample_delta']
  checks={'identity':x['outer_step']==step and x['rank']==rank and x['job_id']=='3154' and x['code_commit']=='f994af33c6c93b6b9844863be912cab2a4020dc1' and x['config_sha256']=='63296d0602c5978d1ff67d258b401054ce587e3a9a2c55052c35fbeac626c198' and x['data_identity']=='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24','batch':(x['world_size'],x['local_rows'],x['local_minibatch'],x['local_microbatch'],x['epochs'])==(8,384,16,2,1),'calls':x['successful_optimizer_calls']==24 and x['skipped_empty_minibatches']==0,'grad':len(x['existing_grad_norms'])==24 and all(math.isfinite(v) and v>0 for v in x['existing_grad_norms']) and x['finite_grad_norms']==24 and x['nonzero_grad_norms']==24,'sample':d['count']==8192 and d['finite']==8192 and (d['nonzero']>0 if step>1 else d['nonzero']==0) and x['local_shard_sample_layout_unchanged'] is True}
  if rank in previous:
   before=previous[rank];checks['continuity']=before['pid']==x['pid'] and before['local_shard_sample_after_sha256']==x['local_shard_sample_before_sha256'] and before['local_shard_sample_layout']==x['local_shard_sample_layout']
  previous[rank]=x
  issues.extend({'step':step,'rank':rank,'check':k} for k,v in checks.items() if not v)
  records.append({'step':step,'rank':rank,'path':str(f),'sha256':hashlib.sha256(b).hexdigest(),'pid':x['pid'],'checks':checks,'nonzero_sample_delta':d['nonzero']})
print(json.dumps({'epoch':time.time(),'status':'OWNER_DIRECT_RANK_PREFIX_CHECK_PASS' if not issues else 'OWNER_DIRECT_RANK_PREFIX_CHECK_FAIL','through_step':75,'records':records,'issues':issues,'optimizer_calls_per_rank':75*24,'scope':'Direct rank records1..75, finite positive recorded norms and8192-coordinate parameter samples, identities/batch and same PID/sample continuity. Does not prove all parameter tensors/losses finite or final scientific effect; independent final acceptance remains pending.'}))
