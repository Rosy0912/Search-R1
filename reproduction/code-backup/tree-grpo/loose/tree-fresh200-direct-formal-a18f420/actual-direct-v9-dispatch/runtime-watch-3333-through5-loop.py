import collections,datetime,json,pathlib,subprocess,sys,time
base=pathlib.Path(__file__).resolve().parent
observations=sorted(base.glob('runtime-watch-3333-observation-*.json'))
number=int(observations[-1].stem.rsplit('-',1)[1])+1 if observations else 2
last=datetime.datetime.fromisoformat(json.loads(observations[-1].read_text())['observed_at']).timestamp() if observations else time.time()
terminal={'FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
def latest(files,key):return [r.get('metadata',{}) for r in files.get(key,{}).get('recent',[])]
def summary(o,num):
 files=o.get('files',{});logs=o.get('anchored_log_metadata',{});stage=logs.get('stage.log',{})
 migrated=latest(files,'output/current-chain-migrated.json')
 migration=[]
 for m in migrated:
  driver=m.get('driver',{})
  migration.append({'status':m.get('status'),'driver':{k:driver.get(k) for k in ['completed_outer_updates','next_outer_update','sampler_cursor','sampler_epoch','completed_optimizer_calls_per_rank']},'worker_rank_count':len(m.get('workers',[])),'all_worker_native_states_equal':len(m.get('workers',[]))==8 and all(w.get('actual_cold_native_state_equal') is True for w in m.get('workers',[]))})
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
 return {'observed_at':o['observed_at'],'observation':num,'slurm':{k:v for k,v in o.get('slurm_identity',{}).get('fields',{}).items() if k in {'JobState','ExitCode','RunTime','Reason'}},'progress_numeric':stage.get('recent_progress_numeric',{}),'training_log_numeric_metrics':stage.get('native_log_numeric_metrics',{}),'exception_classes':{k:v.get('exception_classes') for k,v in logs.items() if v.get('exception_classes')},'native_launch_config_identity':[{k:r.get(k) for k in ['code_commit','config_sha256','job_id','run_id','stage','status']} for r in latest(files,'launch-manifest.json')], 'migration':migration,'latest_attempt':attempts,'latest_rollout':rollouts,'policy_statistics_scope':policy.get('statistics_scope'),'policy_selected_files':policy.get('selected_event_file_count'),'policy_latest_by_rank':policy_latest,'rolling_json':rolling_json,'rolling_file_count':len(rolling.get('files',[])),'native_run_result':latest(files,'run-result.json'),'permanent5':{'manifest_present':o.get('permanent_step5_metadata',{}).get('manifest_present'),'status':o.get('permanent_step5_metadata',{}).get('manifest_metadata',{}).get('status'),'manifest_sha256':o.get('permanent_step5_metadata',{}).get('manifest_sha256'),'file_count':len(o.get('permanent_step5_metadata',{}).get('files',[]))},'development_latest':latest(files,'output/validation-cost/validation-*.json'),'background_identity_disposition':'MISSING_AT_V9_STARTUP_CAUSE_UNKNOWN', 'background_exists_count':sum(r.get('exists',False) for r in o.get('original_eight_pid_identities',[])), 'background_preserved_count':sum(r.get('starttime_preserved') is True for r in o.get('original_eight_pid_identities',[])),'guard_latest':[{k:r.get(k) for k in ['epoch','gpu_count','foreign_process_count','max_foreign_memory_fraction','max_foreign_sm_upper']} for r in resources]}
def signature(s):
 value={k:v for k,v in s.items() if k not in {'observed_at','observation','guard_latest'}}
 value['slurm']={k:v for k,v in value['slurm'].items() if k!='RunTime'}
 return json.dumps(value,sort_keys=True)
previous=signature(summary(json.loads(observations[-1].read_text()),number-1)) if observations else None
milestone5_emitted=False
while True:
 interval=120
 remaining=last+interval-time.time()
 if remaining>0:time.sleep(remaining)
 response=subprocess.run([sys.executable,str(base/'runtime-watch-3333-poll.py'),str(number)],capture_output=True,text=True,timeout=70)
 (base/('runtime-watch-3333-poll-%02d.stdout'%number)).write_text(response.stdout)
 (base/('runtime-watch-3333-poll-%02d.stderr'%number)).write_text(response.stderr)
 path=base/('runtime-watch-3333-observation-%02d.json'%number)
 if response.returncode or not path.exists():
  last=time.time();print(json.dumps({'observation':number,'connection_or_observation_status':'UNKNOWN','returncode':response.returncode,'diagnostic_metadata_saved':True}),flush=True);number+=1;continue
 value=json.loads(path.read_text());last=datetime.datetime.fromisoformat(value['observed_at']).timestamp()
 current=summary(value,number);key=signature(current)
 if key!=previous:print(json.dumps(current,sort_keys=True),flush=True);previous=key
 status=current['slurm'].get('JobState')
 if status in terminal:
  fault=subprocess.run([sys.executable,'work/remote.py','--stdin',str(base/'runtime-watch-3333-fault-details.py'),'python3 -'],capture_output=True,text=True,timeout=50)
  (base/'runtime-watch-3333-fault-details.json').write_text(fault.stdout if fault.returncode==0 else json.dumps({'remote_exit_code':fault.returncode,'safe_diagnostic':fault.stderr[:200]}))
  print(json.dumps({'watch_completed':'ACTUAL_FAULT','observation':number,'positive_slurm_captured':bool(value.get('positive_terminal_slurm_raw_stdout')),'fault_metadata_returncode':fault.returncode}),flush=True);break
 if status=='COMPLETED':print(json.dumps({'watch_completed':'ACTUAL_SLURM_COMPLETED','observation':number}),flush=True);break
 latest_rolling=next((r.get('metadata',{}) for r in value.get('rolling_completed_native_metadata',{}).get('json_metadata',[]) if r['name'].endswith('/latest.json')),{})
 permanent=value.get('permanent_step5_metadata',{})
 dev=latest(value.get('files',{}),'output/validation-cost/validation-*.json')
 dev5=any(r.get('status')=='VALIDATION_COST_COMPLETE' and r.get('global_step')==5 and r.get('evaluated_rows')==512 for r in dev)
 if not milestone5_emitted and dev5 and permanent.get('manifest_present') and permanent.get('manifest_metadata',{}).get('status') in {'COMPLETE_SAME_WORLD_STATE_SAVED','COMPLETE_NATIVE_ROLLING_STATE_SAVED'}:
  print(json.dumps({'configured_milestone':'FIFTH_NATIVE_CHECKPOINT_AND_CONFIGURED_DEV_PUBLISHED_CONTINUING_SAME_JOB','observation':number}),flush=True);milestone5_emitted=True
 number+=1
