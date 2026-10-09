"""Observe exact own3352 without signalling or modifying runtime."""
import json,os,subprocess,datetime
from pathlib import Path
assert os.uname().nodename=='h100-11'
r=Path('/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v2-20261009');b=json.loads((r/'scheduler-head/formal-submit-receipt.json').read_text());assert b['job']=='3352'and b['token']=='09f93f1e773f4419928c0662e4537f32'
run=Path(b['run']);out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job':'3352','run':str(run),'logs':{}}
for k,args in [('slurm',['scontrol','show','job','3352','-o']),('queue',['squeue','-h','-j','3352','-o','%i|%T|%R|%k'])]:
 p=subprocess.run(args,capture_output=True,text=True,timeout=15);out[k]={'exit':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
for n in ['formal-supervisor.log','retriever.log','ray-head.log','train.log','resource-watch.jsonl','original-exit-code.txt','cleanup-exit-code.txt']:
 p=run/n
 if p.is_file():
  with p.open('rb')as f:f.seek(max(0,p.stat().st_size-18000));out['logs'][n]=f.read().decode(errors='replace')
for name in ['out','err']:
 p=r/'logs'/('slurm-formal-3352.'+name)
 if p.exists():out['logs']['slurm_'+name]=p.read_text()[-10000:]
birth=run/'own-formal-retriever-birth.json'
if birth.exists():
 value=json.loads(birth.read_text());out['retriever_birth']=value
 ident=value['identity'];proc=Path('/proc')/str(ident['pid'])
 for name in ['stat','status','wchan','io']:
  try:out['retriever_'+name]=(proc/name).read_text()
  except OSError as exc:out['retriever_'+name]=repr(exc)
 try:out['retriever_open_files']=[os.readlink(q) for q in (proc/'fd').iterdir()]
 except OSError as exc:out['retriever_open_files']=repr(exc)
print(json.dumps(out))
