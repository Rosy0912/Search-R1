"""Independent local review of real raw evidence; missing large .pt remain remote."""
from pathlib import Path
import json,hashlib,math,re,ast,collections,types,yaml,copy
B=Path(__file__).resolve().parent;A=(B.parent/'evidence/recovery3231-hardware3239-raw-evidence').resolve();ROOT=Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-recovery-20261005');RUN=ROOT/'runs/bapo-instruct300-hardware-3239';R=A/'runs'/RUN.name;SOURCE='7de6494bff08f35ffeefbf23ed9a9e1b73e5612e'
def h(p):
 if not p.exists()and A/'repo'in p.parents:p=B.parent/'bapo-e1-300-recovery'/p.relative_to(A/'repo')
 return hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
inv=read(R/'actual-fullhash-inventory.json');items=inv['evidence'];assert len(items)==326 and len({i['path']for i in items})==326
bound={x['path']:x for x in items};missing=[]
for x in items:
 p=A/Path(x['path']).relative_to(ROOT)
 assert type(x['bytes'])is int and x['bytes']>=0 and re.fullmatch('[0-9a-f]{64}',x['sha256'])
 if p.exists():assert not p.is_symlink()and p.stat().st_size==x['bytes']and h(p)==x['sha256']
 else:missing.append(x)
assert {x['path']for x in missing}=={str(RUN/'recovery-native-v3-roundtrip'/f'rank_{i}.pt')for i in range(8)}
assert sum(x['bytes']for x in missing)==38278182288
s=(R/'slurm-terminal.txt').read_text()
for k,v in [('JobId','3239'),('JobState','COMPLETED'),('ExitCode','0:0'),('DerivedExitCode','0:0'),('Restarts','0'),('NodeList','h100-5')]:assert re.search(r'(?:^|\s)'+k+'='+re.escape(v)+r'(?:\s|$)',s)
assert read(R/'own-process-cleanup.json')['remaining']==[]
for n in ('source-commit.txt','cpu-gate-source.txt'):assert(R/n).read_text().strip()==SOURCE
cpu=read(R/'cpu-gate.json');assert cpu['passed']==395 and cpu['failed']==cpu['skipped']==cpu['exit_code']==0 and cpu['sys_prefix']==str(ROOT/'env')and cpu['source_commit']==SOURCE
assert h(A/'logs/cpu-gate.log')==cpu['log_sha256'];tools=read(R/'tools-manifest.json');assert len(tools['files'])==27
for n,sha in tools['files'].items():assert h(A/'tools'/n)==sha
manifest=read(R/'recovery-manifest.json');assert manifest['target_job_id']==3239 and manifest['target_source_commit']==SOURCE and manifest['target_run_root']==str(RUN)
c=read(B.parent/'evidence/formal3231-checkpoint25-fullhash-cpu-candidate.json');oldreports={x['rank']:x for x in c['native_reports']}
# Candidate native_reports is a list of true raw producer reports.
rawold=B.parent/'evidence/formal3231-fault-raw-20261005T0918/runs/bapo-instruct300-formal300-3231'
driverold=read(rawold/'checkpoints/full_state/global_step_25/driver-reload.json');algo=read(rawold/'checkpoints/algorithm/global_step_25/state.json');oldcfg=yaml.safe_load((rawold/'resolved-config.yaml').read_text());cfg=yaml.safe_load((R/'resolved-config.yaml').read_text())
treecfg=ast.parse((B.parent/'bapo-e1-300-recovery/search_r1/baselines/recovery.py').read_text());fn=next(x for x in treecfg.body if isinstance(x,ast.FunctionDef)and x.name=='config_identity');identity={'copy':copy};exec(compile(ast.Module(body=[fn],type_ignores=[]),'actual-config-identity','exec'),identity);assert identity['config_identity'](cfg)==identity['config_identity'](oldcfg)
assert cfg['trainer']['resume_verify_only']is True and cfg['trainer']['resume_full_state']==str(RUN/'recovery-manifest.json')
for cfgx in (cfg,oldcfg):assert cfgx['actor_rollout_ref']['rollout']['tensor_model_parallel_size']==1 and cfgx['actor_rollout_ref']['actor']['optim']['num_warmup_steps']==114 and cfgx['trainer']['total_training_steps']==301 and cfgx['baselines']['bapo']['stage_validation_size']==64
ranks=[read(R/'recovery-witness'/f'rank_{i}.json')for i in range(8)];d=read(R/'recovery-witness/driver.json');v3s=[]
for i,w in enumerate(ranks):
 old=oldreports[i];e=next(x for x in manifest['ranks']if x['rank']==i)
 assert w['status']=='EXACT_CROSS_JOB_FULL_STATE_RELOAD_OK'and w['rank']==i and w['world_size']==8 and w['step']==w['scheduler_last_epoch']==25 and w['next_step']==26
 assert w['distributed_backend']=='nccl'and w['model_state_dict_type']=='SHARDED_STATE_DICT'and w['model_dtensor_entries']==old['model_dtensor_entries']==435 and w['optimizer_state_entries']==old['optimizer_state_entries']==37
 assert w['old_pid']==old['pid']and w['new_pid']!=old['pid']and w['state_digest']==old['state_digest']and w['old_state_sha256']==e['state']['sha256']
 assert w['model_class']=='Qwen2ForCausalLM'and w['model_config']==dict(model_type='qwen2',hidden_size=2048,num_hidden_layers=36,vocab_size=151936,architectures=['Qwen2ForCausalLM'])
 assert all(math.isclose(lr,1e-6*25/114,rel_tol=1e-12)for lr in w['lr'])
 v=read(R/'recovery-native-v3-roundtrip'/f'rank_{i}.json');v3s.append(v)
 assert v['status']=='EXACT_FULL_STATE_RELOAD_OK'and v['native_state_version']==3 and v['roundtrip']is True and v['step']==v['scheduler_last_epoch']==25 and v['rank']==i and v['world_size']==8 and v['pid']==w['new_pid']
 assert v['path']==str(RUN/'recovery-native-v3-roundtrip'/f'rank_{i}.pt')and v['sha256']==bound[v['path']]['sha256']
 assert v['state_digest']==v['reloaded_state_digest']and v['generation_rng_before_digest']==v['generation_rng_after_digest']
 assert v['model_dtensor_entries']==435 and v['optimizer_state_entries']==37
 top=v['generation_rng_topology'];assert top['version']==1 and top['device_mesh']==dict(shape=[8,1],names=['dp','infer_tp'],ranks=[[i]for i in range(8)])
 for name in ('torch_random_states','gen_random_states'):assert type(top[name+'_numel'])is int and top[name+'_numel']>0 and top[name+'_dtype']=='torch.uint8'and top[name+'_shape']==[top[name+'_numel']]
assert d['status']=='EXACT_CROSS_JOB_DRIVER_RELOAD_OK'and d['verify_only']is True and d['no_generation_or_optimizer_update']is True and d['old_pid']==driverold['pid']and d['new_pid']!=driverold['pid']and d['loaded_state_digest']==driverold['state_digest']and d['bapo']==algo['bapo']and d['rank_witnesses']==ranks
assert d['step']==d['actual_outer_updates']==25 and d['next_step']==26 and d['epoch']==0 and d['sampler_cursor']==12800
for n in ('rng','sampler'):assert d['saved_'+n+'_digest']==d['restored_'+n+'_digest']
assert len({x['new_pid']for x in ranks+[d]})==9
for w in ranks+[d]:
 assert w['target_source_commit']==SOURCE and w['target_run_root']==str(RUN)and w['sampling_continuity']=='LEGACY25_GENERATION_RNG_RECONSTRUCTED_NOT_BITWISE_EXACT'
 p=w['process'];assert p['pid']==w['new_pid']and type(p['start_ticks'])is int and p['start_ticks']>0 and p['actual_env']['SLURM_JOB_ID']=='3239'and p['actual_env']['BAPO_RUN_ID']==str(RUN)
 assert w['runtime_module_sha256']==h(A/Path(w['runtime_module_path']).relative_to(ROOT))
train=(R/'train.log').read_text();assert 'BAPO_TRAIN_METRICS_JSON='not in train and 'E1_STEP5_PAUSED_JSON='not in train
for marker,expected,sort in [('BAPO_RECOVERY_RANK_JSON=',ranks,True),('BAPO_RECOVERY_DRIVER_JSON=',[d],False),('BAPO_RECOVERY_V3_RANK_JSON=',v3s,True)]:
 rows=[json.JSONDecoder().raw_decode(l.split(marker,1)[1].lstrip())[0]for l in train.splitlines()if marker in l];assert (sorted(rows,key=lambda x:x['rank'])if sort else rows)==expected
# Execute actual frozen math validation functions with only SHA I/O relocated.
source=(A/'tools/verify_hardware_receipt.py').read_text();tree=ast.parse(source);funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in('finite','validate_native_counts','validate_native_stability')]
ns={'math':math,'sha':lambda p:h(A/Path(p).relative_to(ROOT))};exec(compile(ast.Module(body=funcs,type_ignores=[]),'frozen-fixture-oracles','exec'),ns)
for i in range(8):
 ns['validate_native_counts'](read(R/'native-count'/f'rank-{i}.json'),i,SOURCE,ROOT);ns['validate_native_stability'](read(R/'native-stability'/f'rank-{i}.json'),i,SOURCE,ROOT)
 for kind in ('native-actor','native-state','numeric-fixture','flash-native'):
  x=read(R/kind/f'rank-{i}.json');ns['finite'](x);assert x['rank']==i
  if kind=='native-actor':
   assert x['source_commit']==SOURCE and len(x['records'])==62 and collections.Counter(z['case']for z in x['records'])=={'actual_actor_micro_invariance':48,'actual_actor_global_empty_step':2,'actual_actor_2560_rows_20_updates':2,'actual_actor_production_bf16_fp32_micro_invariance':8,'actual_actor_production_bf16_20_updates':2}
   for z in x['records']:
    flat=[abs(t)for g in z['gradients']for t in g];tol=(.03 if z['production_bf16_fp32']else 2e-5)*max(flat,default=0)+(2e-4 if z['production_bf16_fp32']else 2e-7);assert 0<=z['max_gradient_error']<=1.1*tol and z['optimizer_steps']==(20 if z['global_rows']==2560 else 1)
   probe=x['production_forward_precision'];assert probe['status']=='PASS_REAL_ACTOR_FORWARD_PRECISION_PROBE'and len(probe['branch_records'])==2
   for z in probe['branch_records']:assert z['finite_forward_backward']is True and z['nonzero_gradient']is True and z['logits_dtype']=='torch.bfloat16'and z['logprob_dtype']=='torch.float32'and z['gradient_dtypes']==['torch.float32']and z['actual_device']==f'cuda:{i}'
  elif kind=='native-state':
   assert x['world_size']==8 and len(x['records'])==2
   for z in x['records']:
    assert z['native_state_version']==3 and z['next_update_full_state_exact_equal']is True and z['generation_rng_saved_and_restored']is True and z['num_warmup_steps']==114 and z['total_actual_optimizer_calls']==28 and z['oracle_sixth_optimizer_calls']==z['reloaded_sixth_optimizer_calls']==4
    for component in z['next_update_components'].values():assert component['exact_equal']is True and component['no_roundtrip_sha256']==component['reloaded_sha256']
    for first,last in [('initial_snapshot','actual_worker_roundtrip'),('oracle_snapshot','reloaded_snapshot')]:assert z[first]['state_digest']==z[last]['state_digest']
  elif kind=='numeric-fixture':
   assert x['source_commit']==SOURCE and len(x['records'])==22
   for z in x['records']:
    if z['case']=='normal_native_and_fullvector_oracle':assert math.isclose(z['norm'],math.sqrt(5120),rel_tol=1e-12)
    elif z['case']=='empty_one_rank_sharded_gradients':assert math.isclose(z['norm'],math.sqrt(480),rel_tol=1e-12)
    elif z['case']=='finite_gradient_fp32_square_overflow':assert math.isclose(z['norm'],4.525483490288562e21,rel_tol=1e-12)
    elif z['case']=='gpu_low_var_kl_saturated_forward_backward_fp64_oracle':assert z['points']==408 and z['max_value_error']<=2e-5 and z['max_grad_error']<=2e-5
  else:assert x['finite_forward_backward']is True and x['max_abs_reference_error']<=.02
model=read(R/'model-gpu.json');assert model['status']=='REAL_OFFICIAL_INSTRUCT_VLLM_CUDA_OK'and model['source_commit']==SOURCE and model['token_ids']
comp=read(R/'retrieval-comparison.json');assert comp['same_asset_identity']is True and comp['asset_identity_verified']is True
for mode in ('cpu','gpu'):
 x=read(R/f'retrieval-{mode}.json');assert x['asset_identity_verified']is True and x['source']['slurm_job_id']=='3239'and x['source']['code_commit']==SOURCE and x['index_metadata']['d']==768 and x['index_metadata']['ntotal']==21015324 and x['index_metadata']['gpu_shard']is(mode=='gpu')and len(x['queries'])==14
assert inv['slurm_reserved_wall_seconds']==1236 and math.isclose(inv['slurm_reserved_gpu_hours'],1236*8/3600)
summary=read(R/'guard-summary.json');assert summary['samples']==70 and summary['abort_events']==0 and summary['recovery_samples']==summary['freeze_events']==0
watches=[json.loads(l)for l in (R/'resource-watch.jsonl').read_text().splitlines()];foreign=set();maxsm=0.;maxmem=0.
for x in watches:
 assert x['pause']is False
 for totals in x['external_totals'].values():
  value=totals['external_sm_percent']if totals['external_sm_percent']is not None else totals['external_sm_upper_bound_percent'];assert value is not None;maxsm=max(maxsm,value);maxmem=max(maxmem,totals['external_memory_percent'])
 for p in x['external_processes']:foreign.add(p['pid'])
assert foreign==set(range(2210497,2210505))and maxsm<=15 and maxmem<=15
assert not foreign.intersection(read(R/'own-process-cleanup.json')['stopped_pids'])
epochs={p:set()for p in foreign}
for row in watches:
 for pid in foreign:
  tick=row['pid_start_ticks'][str(pid)];assert tick['before']==tick['after'];epochs[pid].add(tick['before'])
assert all(len(x)==1 for x in epochs.values())
print(json.dumps(dict(status='PASS_INDEPENDENT_ACTUAL_HARDWARE_RAW_REVIEW_NOT_TARGET_GATE',job_id=3239,small_files_rehashed=318,large_files_remote_bound=8,large_bytes=38278182288,restore_witnesses=9,v3_reports=8,math_fixture_ranks=8,guard_samples=70,max_observed_external_sm_percent=maxsm,max_observed_external_memory_percent=maxmem,foreign_pids=sorted(foreign),wall_seconds=1236,reserved_gpu_hours=1236*8/3600)))
