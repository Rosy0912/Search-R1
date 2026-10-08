import json,hashlib,subprocess,time,socket
from pathlib import Path
r=Path("/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-instruct300-3154")
files={}
for pat in ["allocation.txt","launch-manifest.json","resolved.yaml","run-result.json","native-fullstate/rank-*.json","gpu-kl-proof/rank-*.json","native-fullstate.log","guard.log","gpu-kl.log","cost.json","termination-cost.json"]:
 for p in sorted(r.glob(pat)):
  b=p.read_bytes();files[str(p)]={"sha256":hashlib.sha256(b).hexdigest(),"text":b.decode(errors="replace")}
for p in [r/"resource-samples.jsonl",Path("/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/slurm-3154.out"),Path("/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/slurm-3154.err")]:
 if p.exists():
  b=p.read_bytes();files[str(p)]={"sha256":hashlib.sha256(b).hexdigest(),"bytes":len(b),"tail":b[-16000:].decode(errors="replace")}
s=subprocess.run(["scontrol","show","job","3154"],capture_output=True,text=True)
print(json.dumps({"epoch":time.time(),"hostname":socket.gethostname(),"slurm":s.stdout,"slurm_error":s.stderr,"files":files,"output_files":[str(p) for p in sorted((r/"output").rglob("*")) if p.is_file()],"native_files":[str(p) for p in sorted((r/"native-fullstate").glob("*"))],"kl_records":[str(p) for p in sorted((r/"gpu-kl-proof").glob("*.json"))]}))
