from pathlib import Path
import json,hashlib,subprocess,time,socket,gzip,sys
r=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-instruct300-3154');files={}
patterns=['resolved.yaml','launch-manifest.json','allocation.txt','*coexistence*.json','retrieval-gate.json','run-result.json','cost.json','output/entry-audit/*','output/validation-cost/*','output/tree-rank-audit/step-000001-rank-*.json','output/tree-rank-audit/step-000002-rank-*.json','output/tree-rank-audit/memory-profile/step-000001-rank-*.json','output/tree-rank-audit/memory-profile/step-000002-rank-*.json','output/tree-audit/step_1_*','output/tree-audit/step_2_*']
for pattern in patterns:
 for p in sorted(r.glob(pattern)):
  if not p.is_file():continue
  b=p.read_bytes();files[str(p)]={'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'text':b.decode(errors='replace')}
for p in [r/'stage.log',r/'resource-samples.jsonl',r/'guard.log',r/'retriever.log']:
 if p.exists():
  b=p.read_bytes();files[str(p)]={'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'text':b.decode(errors='replace')}
s=subprocess.run(['scontrol','show','job','3154'],capture_output=True,text=True)
x={'epoch':time.time(),'hostname':socket.gethostname(),'slurm':s.stdout,'slurm_error':s.stderr,'files':files,'output_inventory':[str(p.relative_to(r)) for p in r.rglob('*.json')]}
sys.stdout.buffer.write(gzip.compress(json.dumps(x).encode()))
