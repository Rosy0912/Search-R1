from pathlib import Path
import json,subprocess,datetime
r=Path('/mnt/data/shaoshitong/search-r1-rl/gigpo')
receipt=json.loads((r/'logs/resume-original200-submission-v2.json').read_text());job=receipt['job_id'].split(';')[0]
runs=sorted((r/'runs').glob('gigpo-resume200to300-job'+job+'-*'));run=runs[-1] if runs else None
obj=dict(time=datetime.datetime.now().isoformat(),job=job,run=str(run) if run else None,queue=subprocess.check_output(['squeue','-j',job,'-o','%.18i %.9T %.10M %.20R'],text=True),gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True))
if run:
 for name in ['launch-manifest.json','run-result.json','ray-audit-environment-verification.json','resume205-ready.json','checkpoint205-cpu-verification.json']:
  p=run/name
  if p.exists():obj[name]=json.loads(p.read_text())
 obj['audit']={}
 for p in sorted((run/'audit').glob('*.jsonl')):
  rows=[]
  for line in p.read_text().splitlines():
   try:rows.append(json.loads(line))
   except json.JSONDecodeError:pass
  obj['audit'][p.name]={'rows':len(rows),'outer_steps':[x.get('outer_step') for x in rows],'last':rows[-1] if rows else None}
 for name in ['training.log','guard.log','retriever.log']:
  p=run/name
  if p.exists():obj[name]=p.read_text(errors='replace')[-5000:]
p=r/'logs'/('resume'+job+'-live.json');p.write_text(json.dumps(obj));print(json.dumps(obj))
