import json,hashlib,time,math
from pathlib import Path
r=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-recovery300-3234/output/tree-rank-audit');out=[]
for rank in range(8):
 p=r/f'step-000110-rank-{rank:02d}.json';raw=p.read_bytes();x=json.loads(raw);old=json.loads((r/f'step-000109-rank-{rank:02d}.json').read_bytes())
 assert (x['outer_step'],x['rank'],x['world_size'],x['job_id'])==(110,rank,8,'3234')
 assert x['code_commit']=='502e9fafa8e3aba722dbfcb493abc2b9d3ece326' and x['config_sha256']=='e54d322ba3635beb8d930c4f8282997f1867b4dbf5a0ca727e54a569bb198f35'
 assert (x['local_rows'],x['local_minibatch'],x['local_microbatch'],x['successful_optimizer_calls'],x['finite_grad_norms'],x['nonzero_grad_norms'])==(480,48,2,10,10,10)
 assert all(math.isfinite(n) and 0<n<1000 for n in x['existing_grad_norms'])
 assert x['pid']==old['pid'] and x['local_shard_sample_before_sha256']==old['local_shard_sample_after_sha256']
 d=x['local_shard_sample_delta'];assert d['count']==d['finite']==8192 and d['nonzero']>0
 out.append(dict(rank=rank,sha256=hashlib.sha256(raw).hexdigest(),pid=x['pid'],norms=x['existing_grad_norms'],sample_delta=d))
print(json.dumps(dict(status='OWNER110_DIRECT_RANK_PREFIX_PASS_NOT_EVALUATION',epoch=time.time(),job_id=3234,step=110,records=out)))
