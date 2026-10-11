import json,sys,shlex,time,datetime,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from search_g1_restore5_transport import remote
root=Path(__file__).resolve().parents[1];state=Path('/Users/shaoshitong/Documents/Search-G1/preparation/mixed-first-fresh200-state.json')
observer=(root/'work/read_search_g1_job3368_actual_template.py').read_text()
last=None
while True:
 tag=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
 evidence=root/'outputs/gigpo-e1-instruct200'/('SEARCH-G1-JOB3368-ACTUAL-'+tag+'.json')
 try:
  raw=remote('env -i PATH=/usr/bin:/bin /usr/bin/python3 -E -B -S -c '+shlex.quote(observer),str(evidence),emit=False)
  d=json.loads(raw);native=d.get('native_status','');stage=d.get('logs/pipeline-status.json',{}).get('stage');updates=d['actual-updates.jsonl'];attempts=d['actual-attempts.jsonl'];workers=d.get('logs/full-state-restored-workers.json',{}).get('ranks',[]);driver=d.get('logs/full-state-restored-driver.json');counts=[v.get('actual_last_write_count') for v in d['ranks']]
  complete=5
  if attempts.get('last') and attempts['last'].get('complete_outer') and attempts['last'].get('all_ranks_agreed'):
   complete=attempts['last']['completed_outer']
  if complete>5 and (not updates.get('last') or updates['last']['step']<complete): complete=max(5,updates['last']['step'] if updates.get('last') else 5)
  summary=dict(utc=d['utc'],job='3368',stage=stage,native_running='JobState=RUNNING' in native,native_failed='JobState=FAILED' in native or stage=='FAILED',restored_ranks=len(workers),individual_restored_ranks=sum(v is not None for v in d.get('individual_restore_receipts',[])),driver_restored=bool(driver),completed_outer=complete,actual_adam_counts=counts,updates=updates['count'],attempts=attempts['count'],acceptance=d.get('logs/five-resumed-updates-acceptance.json'),evidence=str(evidence))
  sd=json.loads(state.read_text())
  assert sd['slurm_job_id']=='3368'
  sd.update(updated_utc=d['utc'],last_observed_utc=d['utc'],last_observed_stage=stage,native_job_state='FAILED1:0' if summary['native_failed'] else ('RUNNING' if summary['native_running'] else native),training_currently_running=summary['native_running'] and not summary['native_failed'],actual_progress_evidence=str(evidence),completed_outer=complete,backend_completed_outer=complete,driver_metric_rows=updates['count'])
  if len(workers)==8:
   assert all(v['actual_optimizer_writes']==100 and v['scheduler_last_epoch']==5 and v['actual_adam_scheduler_readback_equal'] and v['actual_rng_readback_equal'] for v in workers)
   assert next(v for v in workers if v['rank']==0)['model_hf_exact_tensors_checked_on_rank0']==435
   sd['current_training_gpu_worker_pids']=[v['process_identity']['pid'] for v in workers]
   sd['latest_checkpoint_semantic_gpu_reload']='PASS actualHF435 and8Adam100/scheduler5/RNG equal'
   sd['actual_counter_scope']='source fullGPU restore verified; subsequent actual writes individually per rank'
  if driver:sd['current_candidate_source_consumed']=driver['actual_candidate_source_cursor']
  if all(type(v) is int for v in counts):sd['actual_adam_per_rank_counts']=counts;sd['actual_adam_per_rank']=min(counts)
  sd.update(remaining_normal_disk_bytes=d['storage']['ordinary_free_bytes'],physical_free_including_reserved_bytes=d['storage']['physical_free_including_reserved_bytes'])
  tmp=state.with_suffix('.watch3368.tmp');tmp.write_text(json.dumps(sd,indent=2,ensure_ascii=False)+'\n');os.replace(tmp,state)
  signature=(stage,len(workers),summary['individual_restored_ranks'],bool(driver),complete,tuple(counts),updates['count'],attempts['count'])
  if signature!=last: print(json.dumps(summary),flush=True);last=signature
  if summary['native_failed']:
   print('TRAINING_FAILURE_NEEDS_TERMINAL_CAPTURE_AND_REPAIR',flush=True);break
  if summary['acceptance'] and summary['acceptance']['accepted'] and all(type(v) is int and v>200 for v in counts):
   print('ACTUAL_FIVE_RESUMED_COMPLETE_AND_NEXT11_ACTUAL_ADAM_STARTED',flush=True);break
 except Exception as exc:print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),observation_error=repr(exc),not_training_exit=True)),flush=True)
 for _ in range(6):time.sleep(20)
