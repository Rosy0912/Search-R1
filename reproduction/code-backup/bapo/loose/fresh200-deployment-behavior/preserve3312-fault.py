from pathlib import Path
import json,hashlib,subprocess,datetime
R=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-behavior-20261006'); run=R/'runs/formal200-8c29851b54174f2083023a98cde9c19b-3312'
out={'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job_id':3312,'files':[],'texts':{}}
for p in sorted(run.rglob('*')):
 if p.is_file():
  n=p.relative_to(run).as_posix();z=p.stat().st_size
  item={'path':n,'bytes':z}
  if z<20000000:
   b=p.read_bytes();item['sha256']=hashlib.sha256(b).hexdigest()
   if p.suffix in ('.log','.txt','.json','.yaml','.err') and 'runtime/' not in n:
    out['texts'][n]=b.decode(errors='replace')
  out['files'].append(item)
for name in ['slurm-formal200-3312.err','slurm-formal200-3312.out']:
 p=R/'logs'/name
 if p.exists():out['texts']['rootlogs/'+name]=p.read_text(errors='replace')
print(json.dumps(out))
