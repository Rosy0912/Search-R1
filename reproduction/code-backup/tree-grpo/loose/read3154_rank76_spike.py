import json,math,hashlib,time
from pathlib import Path
p=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-instruct300-3154/output/tree-rank-audit')
out=[]
for rank in range(8):
 f=p/f'step-000076-rank-{rank:02d}.json';b=f.read_bytes();x=json.loads(b);before=json.loads((p/f'step-000075-rank-{rank:02d}.json').read_bytes());norms=x['existing_grad_norms'];d=x['local_shard_sample_delta']
 out.append({'rank':rank,'sha256':hashlib.sha256(b).hexdigest(),'norms':norms,'finite_norms':all(math.isfinite(v) for v in norms),'successful_optimizer_calls':x['successful_optimizer_calls'],'skipped_empty_minibatches':x['skipped_empty_minibatches'],'sample_delta':d,'same_pid':before['pid']==x['pid'],'sample_chain_equal':before['local_shard_sample_after_sha256']==x['local_shard_sample_before_sha256']})
print(json.dumps({'epoch':time.time(),'step':76,'records':out,'scope':'Recorded gradient norms and8192-coordinate parameter delta samples; not full parameter finite check.'}))
