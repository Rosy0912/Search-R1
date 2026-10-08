from pathlib import Path
import json,hashlib,subprocess,time
r=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-instruct300-3154');gate=r/'output/acceptance-step5';files={}
for p in gate.glob('*'):
 if p.is_file() and (p.suffix=='.json' or p.suffix=='.log'):
  b=p.read_bytes();files[str(p)]={'sha256':hashlib.sha256(b).hexdigest(),'text':b.decode(errors='replace')}
s=subprocess.run(['scontrol','show','job','3154'],capture_output=True,text=True)
print(json.dumps({'epoch':time.time(),'slurm':s.stdout,'slurm_error':s.stderr,'files':files,'rank_count':len(list((r/'output/tree-rank-audit').glob('step-*-rank-*.json'))),'result':json.loads((r/'run-result.json').read_text()) if (r/'run-result.json').exists() else None}))
