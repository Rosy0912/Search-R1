from pathlib import Path
import json,hashlib,subprocess,time
r=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-instruct300-3154');files={}
for pattern in ['output/acceptance-step5/ready.json','output/acceptance-step5/approved.json','output/acceptance-step5/released.json','output/tree-rank-audit/step-000005-rank-*.json','output/tree-rank-audit/step-000006-rank-*.json','stage.log','resource-samples.jsonl','run-result.json','cost.json']:
 for p in r.glob(pattern):
  b=p.read_bytes();files[str(p)]={'sha256':hashlib.sha256(b).hexdigest(),'text':b.decode(errors='replace')}
s=subprocess.run(['scontrol','show','job','3154'],text=True,capture_output=True)
print(json.dumps(dict(epoch=time.time(),slurm=s.stdout,error=s.stderr,rank6_count=len(list((r/'output/tree-rank-audit').glob('step-000006-rank-*.json'))),files=files)))
