import pathlib,json,datetime
base=pathlib.Path('/mnt/data/shaoshitong/search-g1-resume5-475fbb6-20261011')
rows=[]
for pool in (base/'artifacts/mixed-candidate-pools').iterdir():
 p=pool/'progress.json'
 if p.exists():
  d=json.loads(p.read_text());rows.append(dict(attempt_id=pool.name,step=d['step'],reserved_candidates=len(d['reserved']),complete_candidate_groups=len(d['complete_ordinals']),last_observed_source_consumed=d['source_consumed'],progress_mtime_ns=p.stat().st_mtime_ns))
 else:rows.append(dict(attempt_id=pool.name,first_candidate_wave_in_progress=True,before_generation_snapshots=len(list(pool.glob('*-before-generation.pt')))))
print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),job_id='3368',candidate_pools=rows,no_complete_outer_claim=True)))
