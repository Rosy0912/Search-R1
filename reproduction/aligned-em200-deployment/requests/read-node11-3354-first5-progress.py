"""Observe exact own3352 without signalling or modifying runtime."""
import json,os,subprocess,datetime
from pathlib import Path
assert os.uname().nodename=='h100-11'
r=Path('/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v4-20261009');b=json.loads((r/'scheduler-head/formal-submit-receipt.json').read_text());assert b['source']=='7a501b38b7ed6b06bf39935b97fcb7bbd56ecc47';job=str(b['job'])
run=Path(b['run']);out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job':job,'run':str(run),'logs':{}}
for k,args in [('slurm',['scontrol','show','job',job,'-o']),('queue',['squeue','-h','-j',job,'-o','%i|%T|%R|%k'])]:
 p=subprocess.run(args,capture_output=True,text=True,timeout=15);out[k]={'exit':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
for n in ['driver-time.log','cleanup.json','formal-supervisor.log','retriever.log','ray-head.log','train.log','resource-watch.jsonl','original-exit-code.txt','cleanup-exit-code.txt']:
 p=run/n
 if p.is_file():
  with p.open('rb')as f:f.seek(max(0,p.stat().st_size-18000));out['logs'][n]=f.read().decode(errors='replace')
for name in ['out','err']:
 p=r/'logs'/('slurm-formal-'+job+'.'+name)
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
out['ledger']={'metrics':[],'training_em':[],'evaluations':[],'attempts':[],'pauses':[]}
p=run/'train.log'
if p.exists():
 with p.open('rb')as f:
  for line in f:
   if not line.endswith(b'\n'):continue
   for label,key in [(b'BAPO_TRAIN_METRICS_JSON=','metrics'),(b'BAPO_TRAIN_EM_AUDIT_JSON=','training_em'),(b'BASELINE_EVALUATION_METRICS_JSON=','evaluations'),(b'FRESH200_ATTEMPT_JSON=','attempts'),(b'FRESH200_STEP5_PAUSED_JSON=','pauses')]:
    index=line.find(label)
    if index>=0:
     try:value,_=json.JSONDecoder().raw_decode(line[index+len(label):].decode().lstrip());out['ledger'][key].append(value)
     except (ValueError,UnicodeDecodeError):out.setdefault('incomplete_markers',[]).append(key)
out['first5_manifests']=[str(p)for p in (run/'checkpoints/full_state/global_step_5').glob('*manifest*.json')] if (run/'checkpoints/full_state/global_step_5').is_dir()else []

out['effective_cuda_env']=[]
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  env=(proc/'environ').read_bytes().split(b'\0')
  if ('BAPO_RUN_ID='+str(run)).encode()in env:
   out['effective_cuda_env'].append({'pid':int(proc.name),'cmdline':(proc/'cmdline').read_bytes().decode(errors='replace').replace('\0',' ')[:300],'launch_blocking':next((v.decode(errors='replace') for v in env if v.startswith(b'CUDA_LAUNCH_BLOCKING=')),None)})
 except (OSError,PermissionError):pass
print(json.dumps(out))
