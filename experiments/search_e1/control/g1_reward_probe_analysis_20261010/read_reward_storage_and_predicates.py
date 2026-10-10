"""Read-only exact reward predicates, persistence schema, and marginal bounds."""
from pathlib import Path
import datetime, hashlib, json, re
BASE=Path('/data/shaoshitong/Search-G1-a800-fresh200-a20f0475ec824c69ba1554f39e9ff512')
run=json.loads((BASE/'logs/run-manifest.json').read_text())
out=dict(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),run_id=run['run_id'],source_commit=run['source_commit'],readonly=True,source=[],scalar_rows=[])
for rel,ranges in [('verl/trainer/main_ppo_eco.py',[(75,100),(390,425),(445,555)]),('reproduction/common8/behavior_trace.py',[(15,110)]),('verl/trainer/ppo/ray_trainer.py',[]),('reproduction/common8/behavior_runtime.py',[])]:
 p=BASE/'repo'/rel;raw=p.read_bytes();lines=raw.decode().splitlines();excerpts=[]
 for a,b in ranges:excerpts.append(dict(first_line=a,source='\n'.join(lines[a-1:b])))
 if rel.endswith('ray_trainer.py'):
  for i,l in enumerate(lines):
   if any(t in l for t in ("batch['token_level_scores']", "batch['token_level_rewards']", "metrics.update(self.reward_fn.get_metrics", "non_tensor_batch['uid']",'behavior_runtime','critic/rewards')):excerpts.append(dict(first_line=max(0,i-4)+1,source='\n'.join(lines[max(0,i-4):i+6])))
 if rel.endswith('behavior_runtime.py'):
  for i,l in enumerate(lines):
   if any(t in l for t in ('make_trace(', 'make_actor_row(', 'write_private_evidence(',"token_level_rewards","advantages")):excerpts.append(dict(first_line=max(0,i-6)+1,source='\n'.join(lines[max(0,i-6):i+14])))
 out['source'].append(dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),bound_sha256=run['source_sha256'].get(rel),excerpts=excerpts))
log=BASE/'logs/search-g1-common8-training.log'
for ln,raw in enumerate(log.open(errors='replace'),1):
 m=re.search(r'\bstep:(\d+)\s*-',raw)
 if not m or int(m[1])>150:continue
 vals={k:float(v) for k,v in re.findall(r'([\w/]+):(-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)',raw)}
 out['scalar_rows'].append(dict(line=ln,step=int(m[1]),metrics={k:v for k,v in vals.items() if any(t in k for t in ('critic/rewards','critic/score','train/perf/em','actor/kl_coef','critic/advantages'))}))
out['log_sha256']=hashlib.sha256(log.read_bytes()).hexdigest()
out['logs_filename_inventory']=[p.name for p in (BASE/'logs').iterdir() if p.is_file() and any(x in p.name for x in ['reward','group','advantage','update'])]
print(json.dumps(out,ensure_ascii=False))
