"""Read the actual new allocation and pinned completed100 state."""
import hashlib,json,subprocess,time
from pathlib import Path
b=Path('/dataset/shaoshitong/recursive-bran-a-h100-20261012');r=b/'recovery-from100-20261013'
commands={}
for key,args in [('job',['scontrol','show','job','3393']),('gpu',['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'])]:
 p=subprocess.run(args,text=True,capture_output=True,timeout=15);commands[key]=dict(rc=p.returncode,stdout=p.stdout,stderr=p.stderr)
files={}
for path in [b/'receipts/configuration.json',b/'receipts/resolved.yaml',b/'checkpoints/training_state/global_step_100/post-evaluation/manifest.json',r/'allocation-job.txt']:
 if path.exists():
  raw=path.read_bytes();files[str(path)]=dict(sha256=hashlib.sha256(raw).hexdigest(),content=raw.decode())
print(json.dumps(dict(time=time.time(),commands=commands,files=files)))
