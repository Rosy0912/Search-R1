import gzip,base64
import pathlib,json,hashlib,time,subprocess
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/tree-grpo');run=root/'runs/tree-fixed-two-3095';files={};paths=[]
for p in run.rglob('*'):
 if not p.is_file():continue
 rel=str(p.relative_to(run))
 if p.suffix in {'.json','.jsonl','.yaml','.log'} and not rel.startswith(('wandb/','cache/','runtime/')) and p.stat().st_size<30*1024**2:paths.append(p)
for d in root.glob('runs/tree-fixed-two-cpu-3095-*'):
 paths.extend(p for p in d.rglob('*') if p.is_file() and p.suffix in {'.txt','.json','.log'})
for p in root.glob('logs/tree-fixed-two*309*.out'):
 if any(k in p.name for k in ('3095','3096','3097')):paths.append(p)
for p in paths:
 data=p.read_bytes()
 try:text=data.decode()
 except UnicodeDecodeError:continue
 files[str(p)]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'text':text}
slurm={}
for job in ['3095','3096','3097']:
 p=subprocess.run(['scontrol','show','job',job],capture_output=True,text=True);slurm[job]={'exit':p.returncode,'text':p.stdout+p.stderr}
print(base64.b64encode(gzip.compress(json.dumps({'epoch':time.time(),'files':files,'slurm':slurm}).encode())).decode())
