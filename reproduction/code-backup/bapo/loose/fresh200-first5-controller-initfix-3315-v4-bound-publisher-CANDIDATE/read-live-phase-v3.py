import json,datetime
from pathlib import Path
R=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-controller-initfix-20261006/runs/formal200-bab8981d9fd14afc98923eb4ebb89e33-3315')
v={'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job_id':3315}
for n in ['train.log','retriever.log','resource-watch.jsonl']:
 p=R/n
 with p.open('rb')as f:f.seek(max(0,p.stat().st_size-4194304));b=f.read()
 if n=='resource-watch.jsonl':
  lines=b.splitlines();vals=[]
  for line in lines[1:]:
   try:vals.append(json.loads(line))
   except ValueError:pass
  samples=[x for x in vals if isinstance(x,dict)and 'pause'in x and 'reason'in x and x.get('slurm_job_id')=='3315'];x=samples[-1]if samples else {};v['resource']={k:x.get(k)for k in ['time','pause','reason','slurm_job_id']};v['resource_complete_sample_found']=bool(samples)
 else:v[n+'_tail']=b.decode(errors='replace')[-6000:]
print(json.dumps(v))
