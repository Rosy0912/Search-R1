import json,subprocess,time
from pathlib import Path
r=Path('/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v11-20261009');x=json.loads((r/'scheduler-head/eval150-submit-receipt.json').read_text());run=Path(x['run']);o=dict(x,time=time.time(),job_state=subprocess.getoutput('squeue -j'+x['job']+' -h -o "%i %T %N %M"'))
for n in ['supervisor.log','retriever.log','ray-head.log','eval.log','eval.err','progress.jsonl','final51713.json','original-exit-code.txt','cleanup-exit-code.txt']:
 p=run/n
 if p.exists():o[n]={'bytes':p.stat().st_size,'tail':p.read_text(errors='replace')[-4500:]}
for suffix in ['out','err']:
 p=r/'logs'/('slurm-eval150-'+x['job']+'.'+suffix)
 if p.exists():o['slurm.'+suffix]=p.read_text(errors='replace')[-1500:]
print(json.dumps(o))
