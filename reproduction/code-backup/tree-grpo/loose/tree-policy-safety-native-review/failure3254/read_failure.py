import json,hashlib,subprocess,time
from pathlib import Path
root=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo');run=root/'runs/tree-policy-safety-native-3254';files={}
for n in ['result.json','allocation.txt','gpu.log','guard.log','resources.jsonl','resources.ready']:
 p=run/n
 if p.exists():
  raw=p.read_bytes();files[str(p)]=dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),text=raw.decode(errors='replace'))
for p in [root/'logs/tree-policy-safety-native-3254.out',root/'logs/tree-policy-safety-native-3254.err',root/'candidates/tree-policy-safety-native-0980ec3/manifest.json']:
 if p.exists():
  raw=p.read_bytes();files[str(p)]=dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),text=raw.decode(errors='replace'))
slurm=subprocess.run(['scontrol','show','job','3254'],capture_output=True,text=True)
pids=[1680782,1680783,1680784,1680785,1696443,1696444,1696445,1696446];identity={}
for pid in pids:
 p=Path('/proc')/str(pid)/'stat';raw=p.read_text();fields=raw.rsplit(')',1)[1].split();identity[str(pid)]=dict(state=fields[0],start_ticks=int(fields[19]),stat_raw=raw)
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],capture_output=True,text=True)
ps=subprocess.run(['ps','-p',','.join(map(str,pids)),'-o','pid=,lstart='],capture_output=True,text=True)
print(json.dumps(dict(epoch=time.time(),files=files,slurm=dict(rc=slurm.returncode,stdout=slurm.stdout,stderr=slurm.stderr),protected_process_identity=identity,gpu_processes=dict(rc=gpu.returncode,stdout=gpu.stdout,stderr=gpu.stderr),protected_ps=ps.stdout,rank_files=[str(p) for p in (run/'gpu-records').glob('rank-*.json')],safety_events=[str(p) for p in (run/'gpu-records').rglob('event-*.json')])),flush=True)
