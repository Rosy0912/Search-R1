import hashlib,importlib.util,json,pathlib
base=pathlib.Path(__file__).resolve().parent
snapshot=base/'formal3210-first5-observed-20261004T171042Z';run=snapshot/'run'
spec=importlib.util.spec_from_file_location('gate_actual',base.parent/'opsd-e1-300/search_r1/baselines/checkpoint_gate.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
m=json.loads((run/'full_state/global_step_5/manifest.json').read_bytes());pids=set()
for rank in range(8):
 rec=m['files'][f'actor/training_state/rank_{rank}.json'];p=run/'checkpoints/actor/global_step_5/training_state'/f'rank_{rank}.json';assert hashlib.sha256(p.read_bytes()).hexdigest()==rec['sha256'];v=json.loads(p.read_bytes());pt=m['files'][f'actor/training_state/rank_{rank}.pt'];assert v['sha256']==v['state_sha256']==pt['sha256'] and v['path']==pt['path'];assert v['status']=='EXACT_FULL_STATE_RELOAD_OK' and v['version']==3 and (v['rank'],v['world_size'],v['step'])==(rank,8,5);assert v['pid'] not in pids;pids.add(v['pid'])
rows=[];pause=[];evaluation=[]
for line in (run/'train.log').read_text().splitlines():
 for marker,dest in [('OPSD_E1_OUTER_JSON=',rows),('OPSD_E1_STEP5_PAUSED_JSON=',pause),('BASELINE_EVALUATION_METRICS_JSON=',evaluation)]:
  if marker in line:dest.append(g.parse_log_marker(line,marker,driver_pid=m['driver_pid'],job_id=m['job_id'],worker_pids=pids,checkpoint_warning_path='/mnt/data/shaoshitong/search-r1-rl/opsd/env/lib/python3.10/site-packages/torch/utils/checkpoint.py'))
assert len(rows)==5 and len(pause)==1
for step,v in enumerate(rows,1):
 assert v['actual_outer_updates']==v['global_step']==step and v['source_commit']==m['source_commit'] and v['config_sha256']==m['config_sha256'];g.validate_outer(v['metrics'])
out={'scope':'Root strict actual 5 logged outer and SHA-pinned native report identities only; full PT/HF CPU content and independent release pending','job_id':'3210','worker_pids':sorted(pids),'manifest_sha256':hashlib.sha256((run/'full_state/global_step_5/manifest.json').read_bytes()).hexdigest(),'train_log_sha256':hashlib.sha256((run/'train.log').read_bytes()).hexdigest(),'rows':rows,'pause':pause,'evaluation':evaluation}
(base/'formal3210-root-strict-first5-prefix-review.json').write_text(json.dumps(out,indent=2));print(json.dumps({'actual_outer':len(rows),'pause_count':len(pause),'evaluation':evaluation,'worker_pids':sorted(pids)}))
