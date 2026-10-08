import collections,datetime,json,pathlib,subprocess,sys,time
base=pathlib.Path(__file__).resolve().parent
observations=sorted(base.glob('runtime-watch-3331-observation-*.json'))
number=int(observations[-1].stem.rsplit('-',1)[1])+1
last=datetime.datetime.fromisoformat(json.loads(observations[-1].read_text())['observed_at']).timestamp()
terminal={'FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
def latest(files,key):return [r.get('metadata',{}) for r in files.get(key,{}).get('recent',[])]
def summary(o,num):
 files=o.get('files',{});logs=o.get('anchored_log_metadata',{});stage=logs.get('stage.log',{})
 migrated=latest(files,'output/current-chain-migrated.json')
 migration=[]
 for m in migrated:
  driver=m.get('driver',{})
  migration.append({'status':m.get('status'),'protocol':m.get('protocol'),'driver':{k:driver.get(k) for k in ['actual_file_sha256_verified','actual_cold_driver_state_equal','completed_outer_updates','next_outer_update','sampler_cursor','sampler_epoch','training_attempts','completed_optimizer_calls_per_rank']},'worker_rank_count':len(m.get('workers',[])),'workers':[{k:w.get(k) for k in ['status','rank','world_size','parent_manifest_sha256','state_file_sha256','state_file_bytes','actual_file_sha256_verified','serialized_logical_state_sha256','actual_loaded_logical_state_sha256','loaded_model_section_sha256','actual_cold_native_state_equal','adam_steps','scheduler_step','actual_total_optimizer_calls','child_code_commit','executor_equivalence','automatic_resume_supported']} for w in m.get('workers',[])],'executor_equivalence':m.get('executor_equivalence'),'initial_evaluation_added':m.get('initial_evaluation_added'),'step5_approval_hold_added':m.get('step5_approval_hold_added')})
 attempts=[]
 for a in latest(files,'checkpoints/training_state/attempt-results/attempt-*.json'):
  rs=a.get('all_rank_result',{}).get('ranks',[])
  attempts.append({k:a.get(k) for k in ['status','attempt_id','accepted_outer','actual_completed_optimizer_calls_per_rank','all_skipped_attempts','completed_outer_before','completed_outer_after']}|{'decision':a.get('all_rank_result',{}).get('decision'),'ranks':[{k:r.get(k) for k in ['rank','accepted_outer','optimizer_calls','total_optimizer_calls','opportunities_processed','opportunities_skipped','scheduler_before','scheduler_after','learning_rates_before','learning_rates_after','first_outer_lr_zero','parameter_delta_verified_by_this_record']}|{'ledger_entries':len(r.get('opportunity_ledger',[])),'ledger_status_counts':dict(collections.Counter(e.get('status') for e in r.get('opportunity_ledger',[])))} for r in rs]})
 rollouts=[{k:r.get(k) for k in ['status','attempt_id','candidate_outer','wall_seconds','trainer_process_cpu_seconds','actual_optimizer_writes','error_type']} for r in latest(files,'output/training-rollout-cost/attempt-*.json')]
 policy=o.get('policy_safety_events',{})
 policy_latest={k:{kk:vv for kk,vv in v.items() if kk in {'rank','phase','attempt_id','global_step','committed_optimizer_calls','monitor_native_forward_calls'}} for k,v in policy.get('latest_by_rank',{}).items()}
 rolling=o.get('rolling_completed_native_metadata',{})
 rolling_json=[{'name':r['name'],'sha256':r.get('sha256'),'bytes':r.get('bytes'),'status':r.get('metadata',{}).get('status'),'completed_step':r.get('metadata',{}).get('identity',{}).get('completed_step'),'completed_outer_updates':r.get('metadata',{}).get('completed_outer_updates'),'next_outer_update':r.get('metadata',{}).get('next_outer_update')} for r in rolling.get('json_metadata',[])]
 resources=files.get('resource-samples.jsonl',{}).get('recent',[])[-1:]
 return {'observed_at':o['observed_at'],'observation':num,'slurm':{k:v for k,v in o.get('slurm_identity',{}).get('fields',{}).items() if k in {'JobState','ExitCode','RunTime','Reason'}},'progress_numeric':stage.get('recent_progress_numeric',{}),'training_log_numeric_metrics':stage.get('native_log_numeric_metrics',{}),'exception_classes':{k:v.get('exception_classes') for k,v in logs.items() if v.get('exception_classes')},'migration':migration,'latest_attempt':attempts,'latest_rollout':rollouts,'policy_statistics_scope':policy.get('statistics_scope'),'policy_selected_files':policy.get('selected_event_file_count'),'policy_latest_by_rank':policy_latest,'rolling_json':rolling_json,'rolling_file_count':len(rolling.get('files',[])),'native_run_result':latest(files,'run-result.json'),'background_all_preserved':all(r.get('starttime_preserved') for r in o.get('original_eight_pid_identities',[])),'guard_latest':[{k:r.get(k) for k in ['epoch','gpu_count','foreign_process_count','max_foreign_memory_fraction','max_foreign_sm_upper']} for r in resources]}
def signature(s):
 value={k:v for k,v in s.items() if k not in {'observed_at','observation','guard_latest'}}
 value['slurm']={k:v for k,v in value['slurm'].items() if k!='RunTime'}
 return json.dumps(value,sort_keys=True)
previous=signature(summary(json.loads(observations[-1].read_text()),number-1))
while True:
 interval=60 if number<=4 else 120
 remaining=last+interval-time.time()
 if remaining>0:time.sleep(remaining)
 response=subprocess.run([sys.executable,str(base/'runtime-watch-3331-poll.py'),str(number)],capture_output=True,text=True,timeout=70)
 (base/('runtime-watch-3331-poll-%02d.stdout'%number)).write_text(response.stdout)
 (base/('runtime-watch-3331-poll-%02d.stderr'%number)).write_text(response.stderr)
 path=base/('runtime-watch-3331-observation-%02d.json'%number)
 if response.returncode or not path.exists():
  last=time.time();print(json.dumps({'observation':number,'connection_or_observation_status':'UNKNOWN','returncode':response.returncode,'diagnostic_metadata_saved':True}),flush=True);number+=1;continue
 value=json.loads(path.read_text());last=datetime.datetime.fromisoformat(value['observed_at']).timestamp()
 current=summary(value,number);key=signature(current)
 if key!=previous:print(json.dumps(current,sort_keys=True),flush=True);previous=key
 status=current['slurm'].get('JobState')
 if status in terminal:
  fault=subprocess.run([sys.executable,'work/remote.py','--stdin',str(base/'runtime-watch-3331-fault-details.py'),'python3 -'],capture_output=True,text=True,timeout=50)
  (base/'runtime-watch-3331-fault-details.json').write_text(fault.stdout if fault.returncode==0 else json.dumps({'remote_exit_code':fault.returncode,'safe_diagnostic':fault.stderr[:200]}))
  print(json.dumps({'watch_completed':'ACTUAL_FAULT','observation':number,'positive_slurm_captured':bool(value.get('positive_terminal_slurm_raw_stdout')),'fault_metadata_returncode':fault.returncode}),flush=True);break
 if status=='COMPLETED':print(json.dumps({'watch_completed':'ACTUAL_SLURM_COMPLETED','observation':number}),flush=True);break
 latest_rolling=next((r.get('metadata',{}) for r in value.get('rolling_completed_native_metadata',{}).get('json_metadata',[]) if r['name'].endswith('/latest.json')),{})
 if latest_rolling.get('completed_outer_updates',0)>=3:
  print(json.dumps({'watch_completed':'THIRD_OUTER_AND_NATIVE_ROLLING_PUBLISHED','observation':number}),flush=True);break
 number+=1
