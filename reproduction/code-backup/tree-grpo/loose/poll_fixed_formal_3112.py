import json,subprocess,time,hashlib
from pathlib import Path
p=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-fixed-formal-3112');o={'epoch':time.time(),'slurm':subprocess.run(['scontrol','show','job','3112'],capture_output=True,text=True).stdout,'kl':[]}
for f in sorted((p/'gpu-kl-proof').glob('rank-*.json')):
 d=json.loads(f.read_text());o['kl'].append({'rank':d['rank'],'status':d['status'],'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'dtypes':[{'dtype':r['dtype'],'exact_points':r['native_dtype_points_compared'],'stable_points':r['fp64_stable_gradient_points'],'boundary_points':len(r['fp64_boundary_points'])} for r in d['records']]})
for name in ('retriever.log','ray-start.log','stage.log','guard.log'):
 f=p/name
 if f.exists():o[name+'_tail']=f.read_text(errors='replace')[-1600:]
for name in ('run-result.json','retrieval-gate.json'):
 f=p/name
 if f.exists():
  d=json.loads(f.read_text())
  o[name]=d if name=='run-result.json' else {'status':d.get('status'),'asset_identity_verified':d.get('asset_identity_verified'),'source_sha256':d.get('source_sha256'),'baseline_sha256':d.get('baseline_sha256'),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}
f=p/'resource-samples.jsonl'
if f.exists():
 lines=f.read_text().splitlines();o['guard_samples']=len(lines);o['latest_guard']=json.loads(lines[-1]);o['guard_events']=[json.loads(s)['event'] for s in lines if json.loads(s).get('event')]
o['rank_records']=len(list((p/'output/tree-rank-audit').glob('step*.json')))
print(json.dumps(o))
