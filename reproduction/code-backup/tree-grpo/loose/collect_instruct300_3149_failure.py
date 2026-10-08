import pathlib,json,hashlib,subprocess,time
r=pathlib.Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-instruct300-3149');files={}
for p in r.rglob('*'):
 if p.is_file() and p.stat().st_size<10*1024**2:
  b=p.read_bytes();files[str(p.relative_to(r))]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'text':b.decode(errors='replace')}
print(json.dumps({'job_id':3149,'epoch':time.time(),'files':files,'scontrol':subprocess.run(['scontrol','show','job','3149'],capture_output=True,text=True).stdout,'gpu_processes':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader'],text=True)}))
