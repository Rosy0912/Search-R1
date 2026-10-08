import json,subprocess,time
from pathlib import Path
name='tree-policy-safety-native'
r=subprocess.run(['squeue','-h','-n',name+','+name+'-v2','-o','%i|%j|%T|%N'],capture_output=True,text=True)
root=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
logs=[]
for p in sorted((root/'logs').glob(name+'-*.out')):
 logs.append(dict(path=str(p),mtime=p.stat().st_mtime,tail=p.read_text(errors='replace')[-8000:]))
runs=[]
for p in sorted((root/'runs').glob(name+'-*')):
 result=p/'result.json'
 runs.append(dict(path=str(p),result=json.loads(result.read_text()) if result.exists() else None,
  gpu_log=(p/'gpu.log').read_text(errors='replace')[-8000:] if (p/'gpu.log').exists() else None))
print(json.dumps(dict(epoch=time.time(),squeue_rc=r.returncode,squeue=r.stdout,squeue_stderr=r.stderr,logs=logs,runs=runs)))
