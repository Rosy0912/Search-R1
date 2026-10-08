from pathlib import Path
import json,datetime
run=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-controller-initfix-20261006/runs/formal200-bab8981d9fd14afc98923eb4ebb89e33-3315')
out={'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job_id':3315,'scope':'read-only running log stat/latest guard sample'}
for n in ('train.log','retriever.log','resource-watch.jsonl'):
 p=run/n;st=p.stat();out[n]={'bytes':st.st_size,'mtime_ns':st.st_mtime_ns}
 if n=='resource-watch.jsonl':
  with p.open('rb')as f:f.seek(max(0,st.st_size-4194304));lines=f.read().splitlines()[1:]
  samples=[]
  for line in lines:
   try:v=json.loads(line)
   except ValueError:continue
   if isinstance(v,dict)and 'pause'in v and 'reason'in v and v.get('slurm_job_id')=='3315':samples.append(v)
  out['latest_guard']=samples[-1]if samples else None
print(json.dumps(out,allow_nan=False))
