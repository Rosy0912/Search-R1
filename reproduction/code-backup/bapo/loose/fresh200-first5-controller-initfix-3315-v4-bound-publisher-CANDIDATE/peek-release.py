import json,hashlib,datetime,subprocess,os
from pathlib import Path
R=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-controller-initfix-20261006');RUN=R/'runs/formal200-bab8981d9fd14afc98923eb4ebb89e33-3315'
assert subprocess.check_output(['git','-C',str(R/'repo'),'rev-parse','HEAD'],text=True).strip()=='60f566b3ddabc88ce9de56ead5ad30a3d6795dd2'
p=RUN/'logs/fresh200-step5-independent-release.json';b=p.read_bytes();assert hashlib.sha256(b).hexdigest()=='2eb2da04ef635260348b1e43526d19021324b29186a06b8874e067d717248c21'
v=json.loads(b);markers=[]
with (RUN/'train.log').open()as f:
 for line in f:
  for m in ['FRESH200_STEP5_RELEASED_JSON=','FRESH200_STEP5_PAUSED_JSON=']:
   if m in line:markers.append(dict(marker=m,value=json.JSONDecoder().raw_decode(line.split(m,1)[1])[0]))
print(json.dumps(dict(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),job_id=3315,driver_pid=v['driver_pid'],driver_live=Path('/proc/'+str(v['driver_pid'])).exists(),markers=markers,release_sha256=hashlib.sha256(b).hexdigest(),slurm=subprocess.check_output(['scontrol','show','job','3315'],text=True))))
