"""Read-only G1 group-statistic availability and exact GRPO source evidence."""
from pathlib import Path
import datetime, hashlib, json, re, socket
BASE = Path('/data/shaoshitong/Search-G1-a800-fresh200-a20f0475ec824c69ba1554f39e9ff512')
run = json.loads((BASE/'logs/run-manifest.json').read_text())
out = dict(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), node=socket.gethostname(), run_id=run['run_id'], source_commit=run['source_commit'], readonly=True, source=[], calibration=[], native_field_availability=[])
for rel, functions in [('verl/trainer/ppo/core_algos.py',['compute_grpo_outcome_advantage']),('verl/trainer/ppo/ray_trainer.py',['compute_advantage']),('reproduction/common8/behavior_trace.py',['make_trace','make_actor_row','write_private_evidence']),('reproduction/common8/behavior_runtime.py',[]),('ts_probe_dump.py',[])]:
 p=BASE/'repo'/rel; raw=p.read_bytes(); lines=raw.decode().splitlines(); excerpts=[]
 for fn in functions:
  starts=[i for i,l in enumerate(lines) if re.match(r'(?:    )?def '+fn+r'\(',l)]
  for a in starts:
   indent=len(lines[a])-len(lines[a].lstrip()); b=next((i for i in range(a+1,len(lines)) if re.match(r'\s*def \w+\(',lines[i]) and len(lines[i])-len(lines[i].lstrip())<=indent),len(lines))
   excerpts.append(dict(first_line=a+1, function=fn, source='\n'.join(lines[a:b])))
 if rel.endswith('behavior_runtime.py'):
  for i,l in enumerate(lines):
   if any(t in l for t in ('reward','advantage','TRACE_FIELDS','capture','write_private','score')): excerpts.append(dict(first_line=i+1, source='\n'.join(lines[max(0,i-2):i+3])))
 if rel=='ts_probe_dump.py':
  for a,b in [(329,359),(393,440)]: excerpts.append(dict(first_line=a,source='\n'.join(lines[a-1:b])))
 out['source'].append(dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),bound_sha256=run['source_sha256'].get(rel),excerpts=excerpts))
for directory in sorted((BASE/'artifacts').glob('round_*')):
 mp=directory/'version-manifest.json'
 if not mp.is_file(): continue
 m=json.loads(mp.read_text())
 if m['round'] not in [0,50,100,150]: continue
 proof_path=directory/'dumps/calibration-label-proof.json'; proof=json.loads(proof_path.read_text())
 rawpaths=list((directory/'dumps').iterdir())
 out['calibration'].append(dict(round=m['round'],directory=str(directory),files=[dict(name=p.name,bytes=p.stat().st_size) for p in rawpaths if p.is_file()],proof_sha256=hashlib.sha256(proof_path.read_bytes()).hexdigest(),proof_keys=list(proof),full_actual_search_count_length=len(proof['actual_retrieval_counts']),full_group_size=proof['group_size'],valid_searched_count=proof['searched_valid_trajectory_count']))
for root in [BASE/'artifacts/behavior-traces',BASE/'logs']:
 paths=sorted(root.glob('attempt-*.json')) if root.name=='behavior-traces' else sorted(root.glob('behavior-probability-attempt-*.json'))
 if not paths: continue
 for p in paths[:1]+paths[-1:]:
  prefix=p.open('rb').read(32768)
  out['native_field_availability'].append(dict(path=str(p),bytes=p.stat().st_size,prefix_bytes=len(prefix),key_names=sorted(set(x.decode() for x in re.findall(rb'"([A-Za-z_][A-Za-z_0-9]*)"\s*:',prefix))),bounded_prefix_not_full_absence_proof=True))
print(json.dumps(out,ensure_ascii=False))
