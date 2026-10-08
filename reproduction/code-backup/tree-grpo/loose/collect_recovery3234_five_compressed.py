import hashlib,json,time,subprocess,gzip,sys
from pathlib import Path
root=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-recovery300-3234')
patterns=['allocation.txt','launch-manifest.json','resolved.yaml','retrieval-gate.json','retriever-coexistence.json','output/recovery-loaded.json','native-fullstate/rank-*.json','gpu-kl-proof/*.json','output/trajectories/outer-00010[1-5]/manifest.json','output/entry-audit/*.jsonl','output/entry-audit/*.yaml','output/tree-rank-audit/step-00010[1-5]-rank-*.json','output/tree-rank-audit/memory-profile/step-00010[1-5]-rank-*.json','output/tree-rank-audit/ratio-tail/step-00010[1-5]-rank-*-optimizer-*.json','output/tree-audit/step_10[1-5]_*.json','output/acceptance-step105/*.json','output/acceptance-step105/*.log','output/validation-cost/*.json']
records={}
for pattern in patterns:
 for path in sorted(root.glob(pattern)):
  raw=path.read_bytes();records[str(path)]=dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),text=raw.decode())
ready=root/'output/acceptance-step105/ready.json'
if ready.exists():
 r=json.loads(ready.read_bytes());mp=Path(r['full_state_manifest']);raw=mp.read_bytes()
 records[str(mp)]=dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),text=raw.decode())
slurm=subprocess.run(['scontrol','show','job','3234'],capture_output=True,text=True)
sys.stdout.buffer.write(gzip.compress(json.dumps(dict(epoch=time.time(),slurm=slurm.stdout,slurm_error=slurm.stderr,ready_exists=ready.exists(),files=records)).encode()))
