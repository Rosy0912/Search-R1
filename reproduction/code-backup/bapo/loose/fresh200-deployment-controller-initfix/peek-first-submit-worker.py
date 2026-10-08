from pathlib import Path
import json,subprocess,datetime
R=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-controller-initfix-20261006');p=R/'scheduler-head/formal200-receipt.json';s=json.loads(p.read_text());assert s['source']=='60f566b3ddabc88ce9de56ead5ad30a3d6795dd2'and str(s['job_id'])=='3315';run=R/'runs'/('formal200-'+s['nonce']+'-3315')
out={'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'receipt':s,'run_root':str(run),'files':[p.name for p in run.iterdir()]if run.exists()else[]}
for n in ['phase.log','train.log','worker-start.json','retriever-health.json']:
 p=run/n
 if p.is_file():
  with p.open('rb')as f:f.seek(max(0,p.stat().st_size-5000));out[n]=f.read().decode(errors='replace')
print(json.dumps(out))
