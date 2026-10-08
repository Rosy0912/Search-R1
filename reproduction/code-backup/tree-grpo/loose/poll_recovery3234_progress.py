"""Read-only recovery status from actual Slurm and complete log records."""
import json,re,time,subprocess,hashlib
from pathlib import Path
root=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-recovery300-3234')
slurm=subprocess.run(['scontrol','show','job','3234'],capture_output=True,text=True)
records=[]
path=root/'stage.log'
if path.exists():
 with path.open() as stream:
  for raw in stream:
   if not raw.endswith('\n'):continue
   line=re.sub(r'\x1b\[[0-9;]*m','',raw)
   m=re.search(r'(?<![\w/])step:(\d+) - ',line)
   if not m:continue
   fields={}
   for field in line[m.end():].split(' - '):
    k,sep,v=field.partition(':')
    if not sep:continue
    try:fields[k]=float(v)
    except ValueError:continue
   records.append(dict(step=int(m[1]),metrics=fields,raw_line_sha256=hashlib.sha256(raw.encode()).hexdigest()))
training=[x for x in records if 'training/actual_outer_updates' in x['metrics']]
scores=[x for x in records if 'val/f1/all' in x['metrics']]
ledgers=[]
for p in sorted((root/'output/validation-cost').glob('*.jsonl')):
 events=[]
 with p.open() as f:
  for line in f:
   if line.endswith('\n'):events.append(json.loads(line))
 ledgers.append(dict(path=str(p),complete_events=len(events),submitted_rows=sum(e.get('rows',0) for e in events if e.get('kind')=='real_sample_batch'),last=events[-1] if events else None,failures=[e for e in events if e.get('kind','').endswith('_failed')]))
print(json.dumps(dict(epoch=time.time(),slurm_returncode=slurm.returncode,slurm=slurm.stdout,slurm_stderr=slurm.stderr,successful_new_outer=len(training),latest_training=training[-3:],score_records=scores,ledgers=ledgers,trajectory_manifests=[dict(path=str(p),manifest=json.loads(p.read_bytes())) for p in sorted((root/'output/trajectories').glob('outer-*/manifest.json'))],ready105_exists=(root/'output/acceptance-step105/ready.json').exists(),result=json.loads((root/'run-result.json').read_bytes()) if (root/'run-result.json').exists() else None)),flush=True)
