import pathlib,json,datetime,shlex,time,sys,re,os
sys.path.insert(0,'work');from search_g1_restore5_transport import remote
script=pathlib.Path('work/read_search_g1_reduced_gate_0553ade_actual_template.py').read_text()
statepath=pathlib.Path('/Users/shaoshitong/Documents/Search-G1/preparation/mixed-first-fresh200-state.json')
while True:
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
 evidence=pathlib.Path(f'outputs/gigpo-e1-instruct200/SEARCH-G1-SHARED3381-ACTUAL-{stamp}.json')
 try:
  x=json.loads(remote('/usr/bin/python3 -c '+shlex.quote(script),str(evidence),emit=False))
 except Exception as e:
  print(json.dumps(dict(utc=stamp,observation_failed=str(e),not_training_exit=True)),flush=True);time.sleep(50);continue
 status=x['native_status'];m=re.search(r'JobState=(\S+)',status);status=m.group(1) if m else 'UNKNOWN'
 up=x['actual-updates.jsonl'];attempt=x['actual-attempts.jsonl'];last=up['last'];rank=x['ranks'];pipeline=x.get('logs/pipeline-status.json')
 restored=(x.get('logs/full-state-restored-workers.json') or {}).get('ranks',[])
 counts=[r['actual_last_write_count'] if r['actual_last_write_count'] is not None else (restored[i]['actual_optimizer_writes'] if len(restored)==8 else None) for i,r in enumerate(rank)]
 summary=dict(utc=x['utc'],job=x['job_id'],native=status,pipeline=pipeline,updates=up['count'],last_complete=last,attempt=attempt,adam=counts,restored_workers=x.get('logs/full-state-restored-workers.json') is not None,restored_driver=x.get('logs/full-state-restored-driver.json') is not None,varlen_ranks=sum(v is not None for v in x['varlen_actual']),driver=x.get('actual_training_driver_process'),native_log_bytes=(x.get('logs/native-3381.log') or {}).get('bytes'),training_log_tail=(x.get('logs/search-g1-common8-training.log') or {}).get('tail','')[-1400:],evidence=str(evidence.resolve()))
 pathlib.Path('work/search-g1-shared3381-latest-compact.json').write_text(json.dumps(summary,indent=2))
 s=json.loads(statepath.read_text())
 if str(s['slurm_job_id'])!='3381':raise RuntimeError('New job state belongs to another execution; stop old watcher')
 s.update(updated_utc=x['utc'],last_observed_utc=x['utc'],native_job_state=status,training_started=status=='RUNNING',actual_progress_evidence=str(evidence.resolve()),actual_adam_per_rank_counts=summary['adam'],current_training_driver_pid=(summary['driver'] or {}).get('pid'),current_stage=pipeline,last_observed_stage=(pipeline or {}).get('stage'),actual_adam_per_rank=counts[0] if counts[0] is not None and len(set(counts))==1 else None,actual_new_adam_per_rank=(counts[0]-100) if counts[0] is not None and len(set(counts))==1 else None)
 if last:
  s['current_last_actual_update']=last
  s['completed_outer']=last.get('completed_outer',last.get('step',s['completed_outer']))
 statepath.write_text(json.dumps(s,indent=2,ensure_ascii=False))
 print(json.dumps({k:v for k,v in summary.items() if k not in ('last_complete','attempt','training_log_tail')},ensure_ascii=False),flush=True)
 if status in ('FAILED','CANCELLED','COMPLETED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL'):
  print('TERMINAL_OBSERVED '+json.dumps(summary,ensure_ascii=False),flush=True);break
 accepted=x.get('logs/five-resumed-updates-acceptance.json')
 if accepted and accepted.get('accepted') and up['count']>=5:
  tail=(x.get('logs/search-g1-common8-training.log') or {}).get('tail','')
  if any(isinstance(r['last_write_context'],dict) and r['last_write_context'].get('logical_update')==11 for r in rank) or re.search(r'step[: =]+11',tail):
   print('PERSONAL_FIVE_PLUS_NEXT_ACTUAL '+json.dumps(accepted),flush=True);break
 time.sleep(50)
