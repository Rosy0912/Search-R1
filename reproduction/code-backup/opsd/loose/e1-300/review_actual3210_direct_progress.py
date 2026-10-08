"""Strict original own-driver evidence join; no training mutation or completion inference."""
import pathlib,json,hashlib,importlib.util,sys,math
H=pathlib.Path(__file__).resolve().parent;p=pathlib.Path(sys.argv[1]);s=importlib.util.spec_from_file_location('gate',H.parent/'opsd-e1-300/search_r1/baselines/checkpoint_gate.py');g=importlib.util.module_from_spec(s);s.loader.exec_module(g);workers=json.loads((H/'formal3210-root-strict-first5-prefix-review.json').read_bytes())['worker_pids'];rows=[];released=[]
for line in (p/'driver-direct.out').read_text().splitlines():
 for marker,dest in [('OPSD_E1_OUTER_JSON=',rows),('OPSD_E1_STEP5_RELEASED_JSON=',released)]:
  if marker in line:dest.append(g.parse_log_marker(line,marker,driver_pid=591546,job_id='3210',worker_pids=workers,checkpoint_warning_path='/mnt/data/shaoshitong/search-r1-rl/opsd/env/lib/python3.10/site-packages/torch/utils/checkpoint.py'))
assert [v['actual_outer_updates'] for v in rows]==list(range(1,len(rows)+1)) and len(rows)<=300
assert len(released)==1 and released[0]['receipt_sha256']=='25b41a15d3e70958b6335f43350ece1c63472478f5d23d9a8c6f78b1a4af433f'
for step,v in enumerate(rows,1):
 assert v['global_step']==step and v['source_commit']=='a0968c2d24beeaa0400f60e4e34677fe6deef9c9' and v['config_sha256']=='a863b6521ffc43ff377625786c7c96c78b3b659fb9cb2cb600ed0da811fee342';m=v['metrics'];g.validate_outer(m);assert math.isclose(m['actor/lr'],1e-6*min(step,57)/57,rel_tol=1e-14)
 for rank in range(8):assert m['teacher/cuda_update_entries_cumulative_per_rank/rank_'+str(rank)]==step
out={'job_id':'3210','source_commit':rows[-1]['source_commit'],'driver_pid':591546,'actual_outer':len(rows),'original_direct_stdout_sha256':hashlib.sha256((p/'driver-direct.out').read_bytes()).hexdigest(),'released':released[0],'rows':rows,'scope':'Strict contiguous original driver logs + actual first5 signed release; no storedHF10/Adam200/sampler5120 or formal300/final claim'};(p/'root-strict-progress-review.json').write_text(json.dumps(out,indent=2));print(json.dumps({'actual_outer':len(rows),'release_receipt_sha256':released[0]['receipt_sha256'],'latest_reward':rows[-1]['metrics']['critic/score/mean'],'latest_loss':rows[-1]['metrics']['actor/opsd_loss']}))
