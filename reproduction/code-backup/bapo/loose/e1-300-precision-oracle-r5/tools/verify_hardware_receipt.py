import pathlib,json,hashlib,subprocess,re,math,itertools,collections,ast
ROOT=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-precision-r5-20261004')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
r=json.loads((ROOT/'logs/hardware-acceptance.json').read_text());source=(ROOT/'source-expected.txt').read_text().strip()
assert source=='733097feeae371b4efd8fd48940c46ff36da839b'
assert r['status']=='PASS_INDEPENDENT_E1_HARDWARE_REVIEW'and r['source_commit']==source and r['root']==str(ROOT)
assert subprocess.check_output(['git','-C',str(ROOT/'repo'),'rev-parse','HEAD'],text=True).strip()==source
assert not subprocess.check_output(['git','-C',str(ROOT/'repo'),'status','--porcelain'],text=True)
job=r['job_id'];assert type(job)is int and job>0
run=ROOT/'runs'/f'bapo-instruct300-hardware-{job}';assert r['run_root']==str(run)and not run.is_symlink()
items=r['evidence']+[r['independent_review']];bound={}
for item in items:
 p=pathlib.Path(item['path']);assert p.is_absolute()and ROOT in p.resolve().parents and p.is_file()and not p.is_symlink()
 assert str(p)not in bound and sha(p)==item['sha256'];bound[str(p)]=item['sha256']
required=[ROOT/'source-expected.txt',ROOT/'common-data/manifest.json',ROOT/'assets/qwen2.5-3b-instruct-aa8e725/VERIFIED-MODEL-MANIFEST.json',run/'source-commit.txt',run/'slurm-terminal.txt',run/'own-process-cleanup.json',run/'HARDWARE_MEASUREMENTS_RETURNED_SUCCESSFULLY',run/'hardware.log',run/'retrieval-comparison.json',run/'resource-watch.jsonl']
required.extend(ROOT/'repo'/name for name in ('search_r1/baselines/distillation.py','search_r1/baselines/objectives.py','verl/workers/actor/dp_actor.py'))
required.extend(p for p in (ROOT/'tools').iterdir()if p.is_file())
required.extend(run/n for n in ('model-gpu.json','retrieval-cpu.json','retrieval-gpu.json','gpu-embeddings.npz','parity-cpu.log','parity-gpu.log','parity-cpu-time.log','parity-gpu-time.log','launch-approval.json','tools-manifest.json','cpu-gate.json','cpu-gate-source.txt','tensordict-device-api.json','runtime-import-origins.log'))
required.extend(ROOT/'logs'/n for n in ('launch-approval.json','tools-manifest.json','cpu-gate.json','cpu-gate.log','cpu-gate-source.txt'))
for p in required:assert str(p)in bound,str(p)
assert (run/'source-commit.txt').read_text().strip()==source
slurm=(run/'slurm-terminal.txt').read_text();assert re.search(r'JobId='+str(job)+r'\b',slurm)and 'JobState=COMPLETED'in slurm and 'ExitCode=0:0'in slurm
assert json.loads((run/'own-process-cleanup.json').read_text())['remaining']==[]
for name in ('launch-approval.json','tools-manifest.json','cpu-gate.json','cpu-gate-source.txt'):
 assert sha(run/name)==sha(ROOT/'logs'/name)
def finite(value):
 if type(value)is float:assert math.isfinite(value)
 elif isinstance(value,dict):
  for v in value.values():finite(v)
 elif isinstance(value,list):
  for v in value:finite(v)
def validate_state_snapshots(x, rank, run, bound):
 profile=x['profile'];root=run/'native-state'/profile
 specs=(('initial_snapshot','initial-step5',5,'FULL_STATE_SERIALIZED_OK'),
        ('actual_worker_roundtrip','roundtrip-step5',5,'EXACT_FULL_STATE_RELOAD_OK'),
        ('oracle_snapshot','no-roundtrip-step6',6,'FULL_STATE_SERIALIZED_OK'),
        ('reloaded_snapshot','reloaded-step6',6,'FULL_STATE_SERIALIZED_OK'))
 pids=set()
 for key,directory,step,status in specs:
  report=x[key];path=root/directory/f'rank_{rank}.pt';record=path.with_suffix('.json')
  assert report['path']==str(path) and path.resolve()==path and record.resolve()==record
  assert path.is_file()and record.is_file()and not path.is_symlink()and not record.is_symlink()
  assert bound.get(str(path))==report['sha256']==sha(path)
  assert str(record)in bound and bound[str(record)]==sha(record) and json.loads(record.read_text())==report
  assert type(report['rank'])is int and report['rank']==rank and type(report['world_size'])is int and report['world_size']==8
  assert type(report['step'])is int and report['step']==step and report['status']==status
  assert report['model_state_dict_type']=='SHARDED_STATE_DICT' and type(report['model_dtensor_entries'])is int and report['model_dtensor_entries']>0
  assert type(report['scheduler_last_epoch'])is int and report['scheduler_last_epoch']==step
  assert type(report['optimizer_state_entries'])is int and report['optimizer_state_entries']>0
  assert re.fullmatch('[0-9a-f]{64}',report['state_digest'])
  assert type(report['pid'])is int and report['pid']>0;pids.add(report['pid'])
 assert len(pids)==1
 assert x['initial_snapshot']['state_digest']==x['actual_worker_roundtrip']['state_digest']
 assert x['oracle_snapshot']['state_digest']==x['reloaded_snapshot']['state_digest']
 components=x['next_update_components'];assert set(components)=={'model','optimizer','scheduler','rng'}
 for item in components.values():
  assert item['exact_equal']is True and re.fullmatch('[0-9a-f]{64}',item['no_roundtrip_sha256'])
  assert item['no_roundtrip_sha256']==item['reloaded_sha256']
 return next(iter(pids))
def validate_production_forward_probe(v, rank, run, ROOT, source, bound):
 probe=v['production_forward_precision']
 probe_path=run/'native-actor'/f'production-forward-rank-{rank}.json'
 assert probe==json.loads(probe_path.read_text())
 assert str(probe_path)in bound and bound[str(probe_path)]==sha(probe_path)
 assert probe['status']=='PASS_REAL_ACTOR_FORWARD_PRECISION_PROBE'and probe['rank']==rank and probe['world']==8
 assert probe['source_commit']==source and probe['actor_source_sha256']==sha(ROOT/'repo/verl/workers/actor/dp_actor.py')
 assert probe['fixture_sha256']==sha(ROOT/'tools/native-actor-fixture.py')and probe['input_device_before_forward']=='cpu'and probe['device']==f'cuda:{rank}'
 assert probe['config_path']==str(ROOT/'repo/verl/trainer/config/ppo_bapo_e1_300.yaml')and probe['config_sha256']==sha(pathlib.Path(probe['config_path']))
 assert probe['torch'].startswith('2.4.0')and probe['sys_prefix']==str(ROOT/'env')
 traversal=probe['actual_output_traversal']
 assert traversal['torch'].startswith('2.4.0')and traversal['namespace_tensor_visits']==0 and traversal['HF_tensor_visits']==1
 assert traversal['output_class']=='transformers.modeling_outputs.CausalLMOutputWithPast'
 traversal_file=pathlib.Path(traversal['utils_file']);assert traversal_file.is_file()and traversal_file.resolve().is_relative_to((ROOT/'env').resolve())
 assert sha(traversal_file)==traversal['utils_sha256']=='33b72b820b44a3b96d89a386737f753d54a96339368aa18efaf09115f207a4db'
 owned_traversal=run/'native-actor'/f'production-output-traversal-source-rank-{rank}.py'
 assert traversal['owned_utils_source_path']==str(owned_traversal)and str(owned_traversal)in bound and bound[str(owned_traversal)]==sha(owned_traversal)==traversal['utils_sha256']

 assert len(probe['branch_records'])==2 and {x['use_remove_padding']for x in probe['branch_records']}=={False,True}
 for b in probe['branch_records']:
  assert type(b['use_remove_padding'])is bool and b['production_branch']is b['use_remove_padding']
  assert b['use_ulysses_sp']is False and b['ulysses_sequence_parallel_size']==1
  assert b['input_device_before_forward']=='cpu'and b['actual_device']==f'cuda:{rank}'
  assert b['actual_parameter_devices']==b['actual_tensor_devices']==[f'cuda:{rank}']
  assert b['logits_dtype']=='torch.bfloat16'and b['logprob_dtype']=='torch.float32'
  assert b['entropy_dtype']in ('torch.bfloat16','torch.float32')and b['gradient_dtypes']==['torch.float32']
  assert b['finite_forward_backward']is True and b['nonzero_gradient']is True
  assert 0<=b['max_logprob_fp64_error']<=.03*4+2e-4
 owned_kernel=run/'native-actor'/f'production-logprob-kernel-source-rank-{rank}.py'
 assert probe['kernel_owned_source_path']==str(owned_kernel)and str(owned_kernel)in bound and bound[str(owned_kernel)]==sha(owned_kernel)==probe['kernel_source_sha256']
 kernel=pathlib.Path(probe['kernel_source_path']);assert kernel.is_file()and sha(kernel)==probe['kernel_source_sha256']
 if probe['kernel_branch']=='flash_cross_entropy':
  assert probe['kernel_module']=='flash_attn.ops.triton.cross_entropy'
  assert kernel.resolve().is_relative_to((ROOT/'env').resolve())and kernel.name=='cross_entropy.py'
 else:
  assert probe['kernel_branch']=='naive_log_softmax'and probe['kernel_module']=='verl.utils.torch_functional'and kernel==ROOT/'repo/verl/utils/torch_functional.py'

 assert set(v['source_files_sha256'])=={'verl/workers/actor/dp_actor.py','search_r1/baselines/objectives.py','search_r1/baselines/gradient_norm.py','verl/trainer/ppo/core_algos.py','verl/utils/fsdp_utils.py','verl/workers/fsdp_workers.py','verl/protocol.py','verl/utils/torch_functional.py','search_r1/baselines/distillation.py'}
 for name,digest in v['source_files_sha256'].items():
  assert not pathlib.PurePosixPath(name).is_absolute()and '..'not in pathlib.PurePosixPath(name).parts
  assert sha(ROOT/'repo'/name)==digest
 assert v['source_files_sha256']['verl/workers/actor/dp_actor.py']==sha(ROOT/'repo/verl/workers/actor/dp_actor.py')

def validate_native_counts(v, rank, source, root):
 assert v['status']=='PASS_NATIVE_NCCL_CPU_INPUT_COUNTS_ONLY'and v['source_commit']==source
 assert type(v['rank'])is int and v['rank']==rank and type(v['world'])is int and v['world']==8
 assert v['fixture_sha256']==sha(root/'tools/native-count-fixture.py')
 assert set(v['source_files_sha256'])=={'search_r1/baselines/distillation.py','search_r1/baselines/objectives.py'}
 for name,digest in v['source_files_sha256'].items():assert sha(root/'repo'/name)==digest
 cases={'one_empty_rank':([0,1,2,3,4,5,6,7],[3,4,5,6,7,8,9,10]),
        'global_empty_tokens':([0]*8,[3,4,5,6,7,8,9,10]),
        'unequal_rows_with_zero':([2,1,4,1,6,2,8,3],[0,2,3,4,5,6,7,8]),
        'global_empty_rows':([0]*8,[0]*8)}
 assert len(v['records'])==4 and {x['case']for x in v['records']}==set(cases)
 for x in v['records']:
  tokens,rows=cases[x['case']]
  assert all(type(x[key])is int for key in ('world','local_tokens','local_rows'))
  assert all(type(x[key])in (int,float)and math.isfinite(x[key])for key in ('global_tokens','token_denominator','global_row_denominator'))
  assert x['world']==8 and x['input_device']=='cpu' and x['collective_device']==f'cuda:{rank}'
  assert x['local_tokens']==tokens[rank]and x['local_rows']==rows[rank]
  assert x['global_tokens']==sum(tokens)and x['token_denominator']==max(sum(tokens),1)/8
  assert x['global_row_denominator']==max(sum(rows),1)/8
native_state_pids={}
for kind in ('native-count','native-actor','native-state','numeric-fixture','flash-native'):
 for rank in range(8):
  p=run/kind/f'rank-{rank}.json';assert str(p)in bound
  v=json.loads(p.read_text());assert v['rank']==rank;finite(v)
  if kind=='native-count':
   validate_native_counts(v,rank,source,ROOT)
  elif kind=='native-actor':
   assert v['status']=='PASS_NATIVE_ACTOR_FSDP8'and v['world']==8 and v['source_commit']==source and len(v['records'])==62
   assert v['repo']==str(ROOT/'repo')and v['actor_file']==str(ROOT/'repo/verl/workers/actor/dp_actor.py')
   assert v['fixture_sha256']==sha(ROOT/'tools/native-actor-fixture.py')
   validate_production_forward_probe(v, rank, run, ROOT, source, bound)
   component_path=run/'native-actor'/f'component-only-rank-{rank}.json'
   assert str(component_path)in bound and bound[str(component_path)]==sha(component_path)
   component=json.loads(component_path.read_text())
   assert component['status']=='PASS_COMPONENT_ACTOR_62_CASES_ONLY_NOT_FULL_HARDWARE'
   assert component['scope']=='COMPONENT_ONLY_PRODUCTION_FORWARD_PROBE_NOT_YET_RUN'
   for name in ('rank','world','torch','sys_prefix','source_commit','repo','actor_file','source_files_sha256','fixture_sha256','records'):assert component[name]==v[name]

   counts=collections.Counter(x['case']for x in v['records'])
   assert counts=={'actual_actor_micro_invariance':48,'actual_actor_global_empty_step':2,'actual_actor_2560_rows_20_updates':2,'actual_actor_production_bf16_fp32_micro_invariance':8,'actual_actor_production_bf16_20_updates':2}
   keys={(x['manual_outer_offload'],x['empty_rank'],x['pg_on'],x['kl_on'],x['micro_per_rank'])for x in v['records']if x['case']=='actual_actor_micro_invariance'}
   expected={(off,empty,pg,kl,micro)for off,empty,(pg,kl),micro in itertools.product((False,True),(False,True),((True,False),(False,True),(True,True)),(1,2,4,16))}
   assert keys==expected and len(keys)==48
   for case,precision,rows,micros in (
     ('actual_actor_global_empty_step',False,128,(2,)),
     ('actual_actor_2560_rows_20_updates',False,2560,(2,)),
     ('actual_actor_production_bf16_fp32_micro_invariance',True,128,(1,2,4,16)),
     ('actual_actor_production_bf16_20_updates',True,2560,(2,))):
    subset=[x for x in v['records']if x['case']==case]
    assert {(x['manual_outer_offload'],x['micro_per_rank'])for x in subset}==set(itertools.product((False,True),micros))
    assert len(subset)==2*len(micros)
    for x in subset:
     assert x['production_bf16_fp32']is precision and x['global_rows']==rows and x['pg_on']is True and x['kl_on']is True
   for x in v['records']:
    assert x['native_cpu_offload']is False and x['mini_per_rank']==16
    assert x['precision_oracle']==('ANALYTIC_BF16_TOY_VJP_FP32_REDUCTION'if x['production_bf16_fp32']else 'FP64_MATHEMATICAL')
    assert x['toy_output_dtype']==('torch.bfloat16'if x['production_bf16_fp32']else 'torch.float64')
    assert x['max_fp64_approximation_deviation']>=0
    assert x['input_device_before_update']=='cpu'and type(x['forward_device_checks'])is int and x['forward_device_checks']>=0
    assert x['actual_forward_devices']==([f'cuda:{rank}']if x['forward_device_checks']else [])
    if x['case']!='actual_actor_global_empty_step':assert x['forward_device_checks']>0
    if x['case']=='actual_actor_micro_invariance':assert x['production_bf16_fp32']is False and x['global_rows']==128
    assert x['optimizer_steps']==(20 if x['global_rows']==2560 else 1)
    assert x['global_rows']in (128,2560)
    assert len(x['gradients'])==x['optimizer_steps'] and all(len(g)==16 for g in x['gradients'])
    assert 0<=x['gradient_norm_min']<=x['gradient_norm_max']
    flat=[abs(z)for g in x['gradients']for z in g]
    assert 0<=x['max_gradient_error']<=1.1*((.03 if x['production_bf16_fp32']else 2e-5)*max(flat,default=0)+(2e-4 if x['production_bf16_fp32']else 2e-7))
    if x['case']=='actual_actor_global_empty_step':assert all(z==0 for z in flat)and x['gradient_norm_min']==x['gradient_norm_max']==0
  elif kind=='native-state':
   assert v['status']=='NATIVE_EIGHT_RANK_FULL_STATE_FIXTURE_PASS_NOT_RL_TRAINING'and v['world_size']==8 and v['source']==source
   assert v['optional_cpu_offload_requested']is False and len(v['records'])==2
   assert {x['profile']for x in v['records']}=={'baseline-fp32','production-nested-bf16-manual'}
   for x in v['records']:
    actual_pid=validate_state_snapshots(x,rank,run,bound)
    assert rank not in native_state_pids or native_state_pids[rank]==actual_pid
    native_state_pids[rank]=actual_pid
    assert x['adam_moment_dtypes']==x['persistent_parameter_dtypes']==['torch.float32']
    assert x['initial_synthetic_outer_updates']==5 and x['initial_optimizer_calls']==20 and x['total_actual_optimizer_calls']==28
    assert x['oracle_sixth_optimizer_calls']==x['reloaded_sixth_optimizer_calls']==4
    assert x['num_warmup_steps']==57 and x['scheduler_last_epoch']==6 and x['same_model_adam_scheduler_objects']is True and x['next_update_full_state_exact_equal']is True
    assert x['native_cpu_offload']is False and x['sharding']=='FULL_SHARD'and x['mesh_shape']==[8]
    assert len(x['gradient_norms'])==28 and all(type(n)in (int,float)and math.isfinite(n)and n>=0 for n in x['gradient_norms'])
    assert len(x['outer_lrs'])==5
    for outer,lr in enumerate(x['outer_lrs'],1):
     assert lr['outer']==outer and math.isclose(lr['before'],1e-6*(outer-1)/57,abs_tol=1e-18,rel_tol=1e-12)and math.isclose(lr['after'],1e-6*outer/57,abs_tol=1e-18,rel_tol=1e-12)
    assert x['oracle_sixth_lr']==x['reloaded_sixth_lr']and x['oracle_sixth_lr']['outer']==6
    assert math.isclose(x['oracle_sixth_lr']['before'],1e-6*5/57,abs_tol=1e-18,rel_tol=1e-12)and math.isclose(x['oracle_sixth_lr']['after'],1e-6*6/57,abs_tol=1e-18,rel_tol=1e-12)
    if x['profile']=='production-nested-bf16-manual':
     assert x['production_like']is True and x['nested_fsdp_wrappers']>=3 and x['manual_param_grad_optimizer_offload']is True
     assert x['param_compute_dtype']=='bfloat16'and x['reduction_dtype']==x['buffer_dtype']=='float32'
   assert v['independent_scalar_warmup_oracle']['optimizer_calls']==v['independent_scalar_warmup_oracle']['scheduler_calls']==57
   assert set(v['reviewed_sources'])=={str(ROOT/'repo'/p)for p in ('verl/workers/fsdp_workers.py','search_r1/baselines/full_state.py','search_r1/baselines/checkpoint_gate.py','search_r1/baselines/gradient_norm.py')}
   for name,digest in v['reviewed_sources'].items():
    q=pathlib.Path(name);assert ROOT/'repo' in q.resolve().parents and sha(q)==digest
  elif kind=='numeric-fixture':
   assert v['status']=='PASS'and v['world']==8 and v['source_commit']==source and len(v['records'])==22
   assert v['gradient_norm_file']==str(ROOT/'repo/search_r1/baselines/gradient_norm.py')and v['core_algos_file']==str(ROOT/'repo/verl/trainer/ppo/core_algos.py')
   ordinary={'normal_native_and_fullvector_oracle','finite_gradient_fp32_square_overflow','one_rank_nonsharded_nan_collective_rejection','empty_one_rank_sharded_gradients'}
   replica={'pure_NO_SHARD_all_empty_gradients','pure_NO_SHARD_one_rank_inf_rejection'}
   signatures=[(x['case'],x.get('orig'),x.get('offload'))for x in v['records']]
   expected={(case,orig,off)for case,orig,off in itertools.product(ordinary,(False,True),(False,True))}|{(case,orig,None)for case,orig in itertools.product(replica,(False,True))}|{('gpu_low_var_kl_saturated_forward_backward_fp64_oracle',None,None),('gpu_low_var_kl_nonfinite_inputs_refused',None,None)}
   assert len(set(signatures))==22 and set(signatures)==expected
   for x in v['records']:
    case=x['case']
    if case=='normal_native_and_fullvector_oracle':assert math.isclose(x['norm'],5120**.5,rel_tol=1e-12,abs_tol=1e-12)
    elif case=='finite_gradient_fp32_square_overflow':assert math.isclose(x['norm'],4.525483490288562e21,rel_tol=1e-12)
    elif case=='empty_one_rank_sharded_gradients':assert math.isclose(x['norm'],480**.5,rel_tol=1e-12,abs_tol=1e-12)
    elif case=='pure_NO_SHARD_all_empty_gradients':assert x['norm']==0
    elif case=='gpu_low_var_kl_saturated_forward_backward_fp64_oracle':
     assert x['points']==408 and 0<=x['max_value_error']<=2e-5 and 0<=x['max_grad_error']<=2e-5
    elif case=='gpu_low_var_kl_nonfinite_inputs_refused':assert x['count']==3
  else:assert v['finite_forward_backward']is True and 0<=v['max_abs_reference_error']<=.02
assert len(native_state_pids)==8 and len(set(native_state_pids.values()))==8
launch_ast=ast.parse((ROOT/'tools/verify_launch.py').read_text())
origin_fn=next(node for node in launch_ast.body if isinstance(node,ast.FunctionDef)and node.name=='validate_import_origins')
namespace={'pathlib':pathlib,'sha':sha};exec(compile(ast.Module(body=[origin_fn],type_ignores=[]),'frozen-origin-validator','exec'),namespace)
text=(run/'runtime-import-origins.log').read_text();proof={}
for label,prefix in [('before','CPU_IMPORT_ORIGINS_BEFORE_JSON='),('after','CPU_IMPORT_ORIGINS_AFTER_JSON=')]:
 events=[json.loads(line.split(prefix,1)[1])for line in text.splitlines()if prefix in line];assert len(events)==1;proof[label]=events[0]
namespace['validate_import_origins'](ROOT/'repo',source,proof)
api=json.loads((run/'tensordict-device-api.json').read_text())
assert api['status']=='ACTUAL_TARGET_TENSORDICT_API_SOURCE_ONLY_NOT_GPU_ACCEPTANCE'and api['source_commit']==source
assert api['torch_version'].startswith('2.4.0')and api['tensordict_version']
assert api['actor_path']==str(ROOT/'repo/verl/workers/actor/dp_actor.py')and api['actor_sha256']==sha(ROOT/'repo/verl/workers/actor/dp_actor.py')
assert set(api['cuda_methods'])=={'TensorDict.cuda','TensorDictBase.cuda'}
for v in api['cuda_methods'].values():
 assert v['signature']and v['source']and pathlib.Path(v['file']).is_file()and sha(pathlib.Path(v['file']))==v['file_sha256']
model=json.loads((run/'model-gpu.json').read_text())
assert model['source_commit']==source and model['status']=='REAL_OFFICIAL_INSTRUCT_VLLM_CUDA_OK'
assert model['model_path']==str(ROOT/'assets/qwen2.5-3b-instruct-aa8e725')
assert model['token_ids']and all(type(token)is int and token>=0 for token in model['token_ids'])
comparison=json.loads((run/'retrieval-comparison.json').read_text());finite(comparison)
assert comparison['status']=='DIFFERENCE_REPORT_ONLY'and comparison['same_asset_identity']is True and comparison['asset_identity_verified']is True and comparison['modes']==['cpu','gpu']
measurements=[]
for mode in ('cpu','gpu'):
 v=json.loads((run/f'retrieval-{mode}.json').read_text());finite(v)
 assert v['status']=='MEASUREMENT_ONLY'and v['mode']==mode and v['asset_identity_verified']is True
 assert v['source']['code_commit']==source and v['source']['slurm_job_id']==str(job)
 assert v['index_metadata']['d']==768 and v['index_metadata']['ntotal']==21015324
 assert v['index_metadata']['gpu_shard']is (mode=='gpu')and v['index_metadata']['gpu_use_float16']is (mode=='gpu')
 assert v['queries_sha256']==comparison['queries_sha256']and len(v['queries'])==14
 measurements.append(v)
assert measurements[0]['asset_identity']==measurements[1]['asset_identity']
assert r['common_manifest_sha256']==sha(ROOT/'common-data/manifest.json')=='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
assert r['native_actor_independent_review_passed']is True and r['native_state_independent_review_passed']is True and r['same_node_gpu_retrieval_verified']is True
subprocess.run([str(ROOT/'env/bin/python'),str(ROOT/'tools/verify_shared_inputs.py')],check=True)
subprocess.run([str(ROOT/'env/bin/python'),str(ROOT/'tools/verify_launch.py'),'formal300'],check=True)
print('NEW_INSTRUCT300_HARDWARE_RECEIPT_GATE_PASS',flush=True)
