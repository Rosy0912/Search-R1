import json,subprocess,pathlib,hashlib,time
r=pathlib.Path('/dataset/shaoshitong/search-r1-rl/cosearch/author-native/runs/author-native-continue-300-x')
launch=json.loads((r/'launch.json').read_text()) if (r/'launch.json').exists() else {}
job=launch.get('job')
q=subprocess.run(['squeue','-h']+(['-j',str(job)] if job else ['--name=cosearch-author-continue'])+['-o','%i %T %N'],text=True,capture_output=True)
v=dict(run=r.name,job=job,checked_at=time.time(),queue=q.stdout,queue_query=dict(returncode=q.returncode,stderr=q.stderr),files={})
for name in ['driver-exit.json','startup-failure.json','launch.json','training.log','retriever.log','ray-head.log']:
 p=r/name
 if not p.exists():continue
 size=p.stat().st_size
 if name.endswith('.log'):
  with p.open('rb') as f:
   f.seek(max(0,size-2600));b=f.read()
  rec=dict(bytes=size,tail_sha256=hashlib.sha256(b).hexdigest(),full_file_sha_not_computed=True)
 else:
  b=p.read_bytes();rec=dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
 if name=='launch.json':
  d=json.loads(b);rec['metadata']={k:d.get(k) for k in ['job','node','leased_uuids','source_commit','startup_seconds','acceptance_passed','warm_initialization','external_tool_manifest'] if k in d}
  rec['metadata'].pop('warm_initialization',None)
 else:rec['text']=b.decode(errors='replace')[-2600:] if name.endswith('.log') else b.decode()
 v['files'][name]=rec
v['acks']=[];v['controller_events']=[];v['phase_install_pids']=[];v['ranker_contracts']=0;v['worker_updates']=[];v['outers']=[];v['loaded_states']=[]
for p in (r/'evidence/workers').glob('*.jsonl'):
 for line in p.open():
  if not line.endswith('\n'):break
  if 'ranker_inference_contract' in line:
   v['ranker_contracts']+=1;continue
  if not any(('\"kind\": \"'+kind+'\"') in line for kind in ['ack','phase_sampling_correction_installed','checkpoint_loaded_full_digest','update','all_rank_ack_gate','rollout_attempt_start','rollout_attempt_return','zero_support_whole_batch_discarded','generation_start','generation_end','outer','original_ranker_metric_support','global_update_completed','continuation_resume_verified','natural_horizon','context_batch_rejected_before_ppo','global_pre_update','ranker_update_skipped_no_rows','global_empty_ranker_batch','empty_global_ranker_output','optional_trajectory_metadata_missing_shards']):continue
  x=json.loads(line)
  if x.get('kind')=='ack':v['acks'].append({k:x.get(k) for k in ['pid','job','role','rank','uuid','process_start_ticks','process_cgroup','adam_state_initial','hook_sha256','source_manifest_sha256']})
  if x.get('kind')=='phase_sampling_correction_installed':v['phase_install_pids'].append(x['pid'])
  if x.get('kind')=='ranker_inference_contract':v['ranker_contracts']+=1
  if x.get('kind')=='checkpoint_loaded_full_digest':v['loaded_states'].append(x)
  if x.get('kind')=='update':v['worker_updates'].append({k:x.get(k) for k in ['role','outer_step','rank','pid','job','update_index','actual_adam_delta','lr_before','parameter_probe_delta_l1','original_gradient_norms','wall_seconds']})
for p in (r/'evidence').glob('controller-*.jsonl'):
 for line in p.open():
  if not line.endswith('\n'):break
  if 'ranker_inference_contract' in line:
   v['ranker_contracts']+=1;continue
  if not any(('\"kind\": \"'+kind+'\"') in line for kind in ['ack','phase_sampling_correction_installed','checkpoint_loaded_full_digest','update','all_rank_ack_gate','rollout_attempt_start','rollout_attempt_return','zero_support_whole_batch_discarded','generation_start','generation_end','outer','original_ranker_metric_support','global_update_completed','continuation_resume_verified','natural_horizon','context_batch_rejected_before_ppo','global_pre_update','ranker_update_skipped_no_rows','global_empty_ranker_batch','empty_global_ranker_output','optional_trajectory_metadata_missing_shards']):continue
  x=json.loads(line)
  if x.get('kind') in ['all_rank_ack_gate','rollout_attempt_start','rollout_attempt_return','zero_support_whole_batch_discarded','generation_start','generation_end','outer','original_ranker_metric_support','global_update_completed','continuation_resume_verified','natural_horizon','context_batch_rejected_before_ppo','global_pre_update','ranker_update_skipped_no_rows','global_empty_ranker_batch','empty_global_ranker_output','optional_trajectory_metadata_missing_shards']:
   if x['kind']=='outer':v['outers'].append(x)
   v['controller_events'].append({k:x[k] for k in ['attempt','max_attempts','input_sha256','next_attempt','elapsed_seconds','discarded_before_both_PPO','kind','pid','job','time_ns','step','validation','input_rows','passed','qualified_groups','role','monotonic_ns','source_run','source_step','next_training_step','target_total_step','dataloader_state_equal','dataloader_sha256','full_state_load_ranks','source_job','original_load_called_once','natural_scheduler_horizon_unchanged','total_steps','actor_horizon','ranker_horizon','informative_reward_groups','masked_nonzero_advantage_tokens','ranker_rows','Adam_steps','scheduler_steps','fields','training_rows_changed','semantics'] if k in x})
v['resource_snapshot']={'disk_free_bytes':__import__('shutil').disk_usage(r.parent.parent).free}
if job and q.stdout.strip():
 resource=subprocess.run(['nvidia-smi','--query-gpu=uuid,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],text=True,capture_output=True)
 v['resource_snapshot']['gpus']={'returncode':resource.returncode,'stdout':resource.stdout,'stderr':resource.stderr}
 v['resource_snapshot']['live_training_pids']=[]
 for ack in v['acks']:
  try:
   pp=pathlib.Path('/proc')/str(ack['pid']);ticks=int((pp/'stat').read_text().rsplit(')',1)[1].split()[19]);cgroup=(pp/'cgroup').read_text()
   v['resource_snapshot']['live_training_pids'].append({'pid':ack['pid'],'role':ack['role'],'rank':ack['rank'],'current_start_ticks_match':ticks==ack['process_start_ticks'],'current_cgroup_match':cgroup==ack['process_cgroup']})
  except FileNotFoundError:v['resource_snapshot']['live_training_pids'].append({'pid':ack['pid'],'missing':True})
v['raw_rollouts']=[dict(name=p.name,bytes=p.stat().st_size) for p in (r/'evidence/raw-rollouts').glob('*.pt')]
print(json.dumps(v))
