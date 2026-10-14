"""Small read-only allocation, GPU and reusable asset state; no model loading."""
import hashlib,json,socket,subprocess,time,shutil
from pathlib import Path
B=Path('/dataset/shaoshitong/recursive-bran-a-h100-20261012')
R=B/'recovery-from100-20261013'
commands={}
for name,args in [('job',['scontrol','show','job','3393']),('gpu',['nvidia-smi','--query-gpu=index,uuid,memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits']),('processes',['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'])]:
 p=subprocess.run(args,capture_output=True,text=True,timeout=15);commands[name]=dict(rc=p.returncode,stdout=p.stdout,stderr=p.stderr)
files={}
for path in [R/'receipts/configuration.json',R/'receipts/resolved.yaml',R/'retriever/receipts/source-identity.json',R/'retriever/receipts/full-scan-http-gate.json',R/'retriever/receipts/active.json']:
 if path.is_file():
  raw=path.read_bytes();files[str(path)]=dict(sha256=hashlib.sha256(raw).hexdigest(),content=raw.decode())
print(json.dumps(dict(time=time.time(),hostname=socket.gethostname(),commands=commands,files=files,disk_free=shutil.disk_usage(B).free)),flush=True)
