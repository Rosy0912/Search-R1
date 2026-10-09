"""Observe exact own3351 without signalling or modifying runtime."""
import json,os,subprocess,datetime
from pathlib import Path
assert os.uname().nodename=='h100-11'
r=Path('/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-20261009');b=json.loads((r/'scheduler-head/formal-submit-receipt.json').read_text());assert b['job']=='3351'and b['token']=='b9633490cbad4d3ba60b7f662ad25d58'
run=Path(b['run']);out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job':'3351','run':str(run),'logs':{}}
for k,args in [('slurm',['scontrol','show','job','3351','-o']),('queue',['squeue','-h','-j','3351','-o','%i|%T|%R|%k'])]:
 p=subprocess.run(args,capture_output=True,text=True,timeout=15);out[k]={'exit':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
for n in ['formal-supervisor.log','retriever.log','ray-head.log','train.log','resource-watch.jsonl','original-exit-code.txt','cleanup-exit-code.txt']:
 p=run/n
 if p.is_file():
  with p.open('rb')as f:f.seek(max(0,p.stat().st_size-18000));out['logs'][n]=f.read().decode(errors='replace')
for name in ['out','err']:
 p=r/'logs'/('slurm-formal-3351.'+name)
 if p.exists():out['logs']['slurm_'+name]=p.read_text()[-10000:]
print(json.dumps(out))
