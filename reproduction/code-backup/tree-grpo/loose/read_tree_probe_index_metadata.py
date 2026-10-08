"""Read only frozen original HF index JSON and current queue metadata on h100-9."""
import hashlib,json,socket,subprocess,time
from pathlib import Path
assert socket.gethostname().split('.')[0]=='h100-9'
p=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-recovery300-3234/output/checkpoints/tree-recovery300/actor/global_step_110/model.safetensors.index.json')
b=p.read_bytes();assert hashlib.sha256(b).hexdigest()=='3a899e7f6ef595e15c34b5ffc7e4a1df6131d7c371418b1a6e8f823ab8a7302d'
q=subprocess.run(['squeue','-h','-w','h100-9','-o','%i %j %T %M %N %b'],capture_output=True,text=True,timeout=15)
print(json.dumps(dict(epoch=time.time(),index=dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),text=b.decode(),bytes=len(b)),queue=dict(rc=q.returncode,stdout=q.stdout,stderr=q.stderr),model_tensors_loaded=False,optimizer_writes=0),sort_keys=True))
