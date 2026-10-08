from pathlib import Path
import json,hashlib,math,re,itertools,struct
cap=Path('work/evidence/stabilityfix-r2-hardware3220-raw');raw=cap/'raw'
inv=json.loads((cap/'inventory.json').read_text());root=Path(inv['root']);run=raw/'runs/bapo-instruct300-hardware-3220'
source='39cdcdd1404213b897bb0de99cc52d72707682f0';repo=Path('work/bapo-e1-300-stabilityfix')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
assert inv['expected_source']==inv['actual_source']==source and inv['root'].endswith('stabilityfix-r2-20261005')
assert inv['source_clean'] and inv['source_matches'] and not inv['missing_required_files'] and not inv['frozen_tool_hash_mismatches']
files={x['path']:x for x in inv['files']};assert len(files)==281
for x in files.values():
 p=raw/x['archive_path'];assert p.is_file()and not p.is_symlink()and sha(p)==x['sha256']and p.stat().st_size==x['bytes']
for x in files.values():
 relative=Path(x['archive_path'])
 if relative.parts[0]=='repo':assert sha(raw/relative)==sha(repo/relative.relative_to('repo'))
manifest=read(raw/'logs/tools-manifest.json');assert len(manifest['files'])==22
assert all(sha(raw/'tools'/n)==v for n,v in manifest['files'].items())
assert sha(raw/'logs/tools-manifest.json')=='f74329d955df6dbf02927424e8e5c225e4c958c776c50f5aaea37f2cac3cd9c6'
for name in ('cpu-gate.json','cpu-gate-source.txt','launch-approval.json','tools-manifest.json'):assert sha(run/name)==sha(raw/'logs'/name)
cpu=read(raw/'logs/cpu-gate.json');assert (cpu['passed'],cpu['failed'],cpu['skipped'],cpu['exit_code'])==(305,0,0,0)
assert sha(raw/'logs/cpu-gate.json')=='c7a68a370632bcabd124c6c5cf7a820156e48fb8933fbef917c3c253854e3778'
assert cpu['sys_prefix']==str(root/'env')and cpu['source_commit']==source
assert cpu['log_sha256']==sha(raw/'logs/cpu-gate.log')and cpu['test_launcher_sha256']==sha(raw/'tools/cpu-tests.py')
origins=0
for log in (raw/'logs/cpu-gate.log',run/'runtime-import-origins.log'):
 proof={}
 for label,marker in [('before','CPU_IMPORT_ORIGINS_BEFORE_JSON='),('after','CPU_IMPORT_ORIGINS_AFTER_JSON=')]:
  ev=[json.loads(x.split(marker,1)[1])for x in log.read_text().splitlines()if marker in x];assert len(ev)==1
  p=ev[0];proof[label]=p;assert p['repo']==str(root/'repo')and p['source_commit']==source
  for item in list(p['modules'].values())+list(p['functions'].values()):
   relative=Path(item['path']).relative_to(root/'repo');assert sha(repo/relative)==item['sha256'];origins+=1
   if 'co_filename'in item:assert item['co_filename']==item['path']
 assert proof['before']['functions']==proof['after']['functions']
 for name,item in proof['before']['modules'].items():assert proof['after']['modules'][name]==item
assert sha(raw/'common-data/manifest.json')=='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
assert sha(raw/'assets/qwen2.5-3b-instruct-aa8e725/VERIFIED-MODEL-MANIFEST.json')=='7f53ce6c010ac076b1ec482031bea7c9e6b8d0a096a61b880fe8ad4971bc9e44'
hooks=[];guard_coords=0;state_profiles=0;numeric_records=0;flash_errors=[]
for rank in range(8):
 v=read(run/f'native-stability/rank-{rank}.json')
 assert v['rank']==rank and v['world']==8 and v['backend']=='nccl'and v['device']==f'cuda:{rank}'
 assert v['status']=='PASS_NATIVE_STABILITY_NCCL8' and v['source_commit']==source and v['sys_prefix']==str(root/'env')and v['repo']==str(root/'repo')
 assert v['fixture_sha256']==sha(raw/'tools/native-stability-fixture.py')
 for f,h in v['source_files_sha256'].items():assert sha(raw/'repo'/f)==h
 assert len(v['records'])==3;guard,unguarded,hook=v['records']
 for r in (guard,unguarded):
  assert r['delta']==30 and r['advantage']==-1 and r['pre_step_shard_numel']==2 and r['pre_step_gradients_present']is True
  assert r['post_return_gradients_cleaned']is True and r['parameters_unchanged_lr0']is True and r['optimizer_lr']==r['optimizer_weight_decay']==0
  assert r['optimizer_steps']==1 and r['forward_device']==f'cuda:{rank}'and r['input_device_before_forward']=='cpu'
 assert guard['pre_step_values']==[0.,0.]and guard['preclip_norm']==0 and guard['pre_step_all_zero']is True
 assert math.isclose(guard['pg_loss'],math.exp(20),rel_tol=1e-12)
 assert unguarded['fixture_baseline_method']=='grpo'and unguarded['pre_step_all_zero']is False
 assert all(math.isclose(x,math.exp(30),rel_tol=1e-12)for x in unguarded['pre_step_values'])
 assert math.isclose(unguarded['pg_loss'],math.exp(30),rel_tol=1e-12)and math.isclose(unguarded['preclip_norm'],4*math.exp(30),rel_tol=1e-12)
 guard_coords+=4;hooks.append(hook)
 s=read(run/f'native-state/rank-{rank}.json');assert s['source']==source and s['world_size']==8 and s['rank']==rank
 for r in s['records']:
  state_profiles+=1;assert r['num_warmup_steps']==114 and r['scheduler_last_epoch']==6
  assert r['same_model_adam_scheduler_objects']is True and r['next_update_full_state_exact_equal']is True
  assert r['initial_optimizer_calls']==20 and r['total_actual_optimizer_calls']==28 and r['initial_synthetic_outer_updates']==5
  assert r['oracle_sixth_optimizer_calls']==r['reloaded_sixth_optimizer_calls']==4
  assert r['adam_moment_dtypes']==r['persistent_parameter_dtypes']==['torch.float32']
  assert r['sharding']=='FULL_SHARD'and r['mesh_shape']==[8]and r['native_cpu_offload']is False
  if r['profile']=='production-nested-bf16-manual':assert r['nested_fsdp_wrappers']>=3 and r['manual_param_grad_optimizer_offload']is True and r['param_compute_dtype']=='bfloat16'
  assert len(r['outer_lrs'])==5
  for k,l in enumerate(r['outer_lrs'],1):assert l['outer']==k and math.isclose(l['before'],1e-6*(k-1)/114,abs_tol=1e-18)and math.isclose(l['after'],1e-6*k/114,abs_tol=1e-18)
  assert r['oracle_sixth_lr']==r['reloaded_sixth_lr']and math.isclose(r['oracle_sixth_lr']['after'],1e-6*6/114,abs_tol=1e-18)
 scalar=s['independent_scalar_warmup_oracle'];assert scalar['optimizer_calls']==scalar['scheduler_calls']==114 and scalar['first_lr']==0 and scalar['after114_lr']==1e-6 and math.isclose(scalar['last_before_lr'],1e-6*113/114,rel_tol=1e-15)
 num=read(run/f'numeric-fixture/rank-{rank}.json');assert num['status']=='PASS'and num['rank']==rank and num['world']==8 and num['source_commit']==source and num['sys_prefix']==str(root/'env')
 assert len(num['records'])==22
 for r in num['records']:
  numeric_records+=1;case=r['case']
  if case=='normal_native_and_fullvector_oracle':assert math.isclose(r['norm'],math.sqrt(5120),rel_tol=1e-12)
  elif case=='finite_gradient_fp32_square_overflow':assert math.isclose(r['norm'],math.sqrt(2048)*struct.unpack("f",struct.pack("f",1e20))[0],rel_tol=1e-12)
  elif case=='empty_one_rank_sharded_gradients':assert math.isclose(r['norm'],math.sqrt(480),rel_tol=1e-12)
  elif case=='pure_NO_SHARD_all_empty_gradients':assert r['norm']==0
  elif case=='gpu_low_var_kl_saturated_forward_backward_fp64_oracle':assert r['points']==408 and r['max_value_error']<=2e-5 and r['max_grad_error']<=2e-5
  elif case=='gpu_low_var_kl_nonfinite_inputs_refused':assert r['count']==3
  else:assert case in ('one_rank_nonsharded_nan_collective_rejection','pure_NO_SHARD_one_rank_inf_rejection')
 f=read(run/f'flash-native/rank-{rank}.json');assert f['rank']==rank and f['finite_forward_backward']is True and f['max_abs_reference_error']<=.02;flash_errors.append(f['max_abs_reference_error'])
locals_=[h['local_original_metrics']for h in hooks]
for h in hooks:
 assert h['all_returns_identical']is True and h['local_original_unchanged']is True and h['replicated_clock_mismatch_rejected']is True and h['scheduler_calls']==1
 g=h['global_metrics']
 for key in ('actor/pg_loss','actor/kl_loss','actor/entropy_loss','actor/ppo_kl','actor/pg_clipfrac'):
  assert len(g[key])==20
  for mini in range(20):assert math.isclose(g[key][mini],math.fsum(r[key][mini]for r in locals_)/8,abs_tol=1e-15)
 assert g['bapo/loss_tokens']==sum(r['bapo/loss_tokens']for r in locals_)==396
 assert g['actor/grad_norm']==[7.]*20 and g['actor/optimizer_steps']==20 and g['actor/scheduler_outer_steps']==5
 assert math.isclose(g['actor/lr_before_outer'],4e-6/114,rel_tol=1e-15)and math.isclose(g['actor/lr'],5e-6/114,rel_tol=1e-15)
cpu_ret=read(run/'retrieval-cpu.json');gpu_ret=read(run/'retrieval-gpu.json');comp=read(run/'retrieval-comparison.json')
assert cpu_ret['asset_identity']==gpu_ret['asset_identity']and cpu_ret['queries']==gpu_ret['queries']and cpu_ret['queries_sha256']==gpu_ret['queries_sha256']==comp['queries_sha256']
assert gpu_ret['source']['code_commit']==cpu_ret['source']['code_commit']==source
assert gpu_ret['source']['slurm_job_id']==cpu_ret['source']['slurm_job_id']=='3220'
assert gpu_ret['index_metadata']['d']==768 and gpu_ret['index_metadata']['ntotal']==21015324 and gpu_ret['index_metadata']['gpu_shard']is True and gpu_ret['index_metadata']['gpu_use_float16']is True
assert gpu_ret['source']['retrieval_server_sha256']==sha(repo/'search_r1/search/retrieval_server.py')
assert gpu_ret['recipe']['encoder_fp16']is True and gpu_ret['source']['faiss_import_path'].startswith(str(root/'retriever-gpu-lib'))
retrieval={}
for k in ('3','50'):
 exact=0;overlap=[];maxdelta=0.
 for a,b in zip(cpu_ret['results'],gpu_ret['results']):
  a=a[k];b=b[k];assert len(a['ids'])==len(b['ids'])==int(k)
  exact+=a['ids']==b['ids'];common=set(a['ids'])&set(b['ids']);overlap.append(len(common)/int(k))
  am={i:(s,h)for i,s,h in zip(a['ids'],a['scores'],a['contents_sha256'])};bm={i:(s,h)for i,s,h in zip(b['ids'],b['scores'],b['contents_sha256'])}
  for i in common:assert am[i][1]==bm[i][1];maxdelta=max(maxdelta,abs(am[i][0]-bm[i][0]))
 assert len(overlap)==14;summary=comp['summary'][k];assert exact==summary['exact_rank_queries']and math.isclose(sum(overlap)/14,summary['mean_overlap_fraction'])and math.isclose(maxdelta,summary['max_matched_id_score_abs_delta'])
 retrieval[k]={'exact_order_queries':exact,'queries':14,'mean_overlap':sum(overlap)/14,'max_shared_score_delta':maxdelta}
watch=[json.loads(x)for x in (run/'resource-watch.jsonl').read_text().splitlines()];assert len(watch)==40 and all(not x['pause']for x in watch)
uuids={x['uuid']for x in gpu_ret['source']['gpu_leases']};assert len(uuids)==8
foreign0={(x['gpu_uuid'],x['pid'],x['memory_mib'])for x in watch[0]['external_processes']}
for row in watch:
 assert row['run']==str(root/'runs/bapo-instruct300-hardware-3220')and row['slurm_job_id']=='3220'
 assert {x['uuid']for x in row['allocated_gpus'].values()}==uuids
 assert {(x['gpu_uuid'],x['pid'],x['memory_mib'])for x in row['external_processes']}==foreign0
cleanup=read(run/'own-process-cleanup.json');assert cleanup['remaining']==[]and cleanup['own_marker']==str(root/'runs/bapo-instruct300-hardware-3220')
assert not {x[1]for x in foreign0}&set(cleanup['stopped_pids'])
terminal=(run/'slurm-terminal.txt').read_text();assert 'JobId=3220 'in terminal and 'JobState=COMPLETED'in terminal and 'ExitCode=0:0'in terminal and 'Restarts=0'in terminal and 'RunTime=00:07:48'in terminal
model=read(run/'model-gpu.json');assert model['source_commit']==source and model['status']=='REAL_OFFICIAL_INSTRUCT_VLLM_CUDA_OK'and model['model_path']==str(root/'assets/qwen2.5-3b-instruct-aa8e725')and model['token_ids']
assert (run/'HARDWARE_MEASUREMENTS_RETURNED_SUCCESSFULLY').is_file()
out={'status':'PASS_INDEPENDENT_RAW_HARDWARE_SUPPLEMENT','job':3220,'source':source,'files':281,'origins_checks':origins,'guard_gradient_coordinates':guard_coords,'state_profiles':state_profiles,'numeric_records':numeric_records,'flash_max_error':max(flash_errors),'retrieval':retrieval,'watch_samples':len(watch),'pause_samples':0,'foreign_stable_sampled_pids':sorted(x[1]for x in foreign0),'cleanup_remaining':[],'allocation_elapsed_seconds':468,'allocation_gpu_hours':468*8/3600,'scope':'ACTUAL_HARDWARE_ONLY_NOT_REAL_RL_FIVE_OR300'}
Path('work/evidence/stabilityfix-r2-hardware3220-independent-supplement.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
