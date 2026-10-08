import hashlib,json,time,subprocess
from pathlib import Path
root=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-instruct300-3154')
patterns=['allocation.txt','launch-manifest.json','resolved.yaml','retrieval-gate.json','retriever-coexistence.json','output/entry-audit/*.jsonl','output/entry-audit/*.yaml','output/tree-rank-audit/step-00000[1-5]-rank-*.json','output/tree-rank-audit/memory-profile/step-00000[1-5]-rank-*.json','output/tree-audit/step_[1-5]_*.json','output/acceptance-step5/*.json','output/acceptance-step5/*.log','output/validation-cost/*.json']
records={}
for pattern in patterns:
 for path in sorted(root.glob(pattern)):
  raw=path.read_bytes();records[str(path)]=dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),text=raw.decode())
ready=root/'output/acceptance-step5/ready.json'
if ready.exists():
 r=json.loads(ready.read_bytes());mp=Path(r['full_state_manifest']);raw=mp.read_bytes()
 records[str(mp)]=dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),text=raw.decode())
slurm=subprocess.run(['scontrol','show','job','3154'],capture_output=True,text=True)
print(json.dumps(dict(epoch=time.time(),slurm=slurm.stdout,slurm_error=slurm.stderr,ready_exists=ready.exists(),files=records)))
