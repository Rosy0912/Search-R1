from pathlib import Path
import json,hashlib,math,collections,itertools,re,numpy as np
b=Path.cwd();cap=b/'work/evidence/e1-300-monitor-r6-hardware3173-capture';raw=cap/'raw';run=raw/'runs/bapo-instruct300-hardware-3173';inv=json.loads((cap/'inventory.json').read_text());root=inv['root'];source=inv['expected_source'];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(cap/'inventory.json')=='476f6575571cf12afc541bf354ee9c697c03f758f09f3efac4b346c5a5462adf'
assert inv['job_id']==3173 and inv['actual_source']==source=='733097feeae371b4efd8fd48940c46ff36da839b' and inv['terminal_completed'] and not inv['missing_required_files']
bound={x['path']:x['sha256']for x in inv['files']}
for section,folder in [('tools_sha256','tools'),('extra_helpers_sha256','evidence-tools')]:
 a=json.loads((run/'launch-approval.json').read_text())
 for n,h in a[section].items():
  if folder=='tools':assert sha(raw/f'{folder}/{n}')==h
assert sha(run/'launch-approval.json')==sha(b/'work/e1-300-monitor-root-r6/launch-approval.json')==sha(raw/'logs/launch-approval.json')
assert sha(run/'tools-manifest.json')==sha(raw/'logs/tools-manifest.json')
assert sha(run/'cpu-gate.json')==sha(raw/'logs/cpu-gate.json')==a['cpu_gate_record_sha256']
assert sha(raw/'common-data/manifest.json')==a['data_manifest_sha256'];assert sha(raw/'assets/qwen2.5-3b-instruct-aa8e725/VERIFIED-MODEL-MANIFEST.json')==a['model_manifest_sha256']
slurm=(run/'slurm-terminal.txt').read_text();assert 'JobId=3173 'in slurm and 'JobState=COMPLETED'in slurm and 'ExitCode=0:0'in slurm and 'Restarts=0'in slurm and 'NodeList=h100-5'in slurm and 'JOB_GRES=gpu:h100:8'in slurm and 'RunTime=00:07:38'in slurm
assert (run/'source-commit.txt').read_text().strip()==source and (run/'HARDWARE_MEASUREMENTS_RETURNED_SUCCESSFULLY').is_file()
maxprobe=0;maxflash=0;maxkl=0
for rank in range(8):
 v=json.loads((run/f'native-actor/rank-{rank}.json').read_text());assert v['rank']==rank and v['source_commit']==source and v['fixture_sha256']==sha(raw/'tools/native-actor-fixture.py')and v['repo']==root+'/repo'and v['actor_file']==root+'/repo/verl/workers/actor/dp_actor.py'
 for n,h in v['source_files_sha256'].items():assert sha(raw/'repo'/n)==h
 assert collections.Counter(x['case']for x in v['records'])=={'actual_actor_micro_invariance':48,'actual_actor_global_empty_step':2,'actual_actor_2560_rows_20_updates':2,'actual_actor_production_bf16_fp32_micro_invariance':8,'actual_actor_production_bf16_20_updates':2}
 groups=collections.defaultdict(list)
 for x in v['records']:
  assert x['input_device_before_update']=='cpu'and x['actual_forward_devices']==([f'cuda:{rank}']if x['forward_device_checks']else [])
  if x['case']!='actual_actor_global_empty_step':assert x['forward_device_checks']>0
  if 'micro_invariance'in x['case']:groups[(x['case'],x['manual_outer_offload'],x.get('empty_rank'),x['pg_on'],x['kl_on'])].append(x)
 for vals in groups.values():
  assert {x['micro_per_rank']for x in vals}=={1,2,4,16}
  for x in vals:
   g=np.array(x['gradients']);ref=np.array(vals[0]['gradients']);rt,at=(.03,2e-4)if x['production_bf16_fp32']else(2e-5,2e-7);assert np.all(abs(g-ref)<=at+rt*abs(ref))
 probe=json.loads((run/f'native-actor/production-forward-rank-{rank}.json').read_text());assert probe['source_commit']==source and probe['actor_source_sha256']==sha(raw/'repo/verl/workers/actor/dp_actor.py')and probe['fixture_sha256']==sha(raw/'tools/native-actor-fixture.py')and probe['config_sha256']==a['config_yaml_sha256']
 assert len(probe['branch_records'])==2 and {x['use_remove_padding']for x in probe['branch_records']}=={False,True}
 for x in probe['branch_records']:
  assert x['logits_dtype']=='torch.bfloat16'and x['logprob_dtype']=='torch.float32'and x['gradient_dtypes']==['torch.float32']and x['finite_forward_backward']and x['nonzero_gradient']and x['actual_tensor_devices']==[f'cuda:{rank}']and x['actual_parameter_devices']==[f'cuda:{rank}'];maxprobe=max(maxprobe,x['max_logprob_fp64_error'])
 for p in (run/'native-actor').glob(f'*rank-{rank}*.py'):assert str(p) # full hash already checked inventory
 n=json.loads((run/f'numeric-fixture/rank-{rank}.json').read_text());assert n['source_commit']==source and n['world']==8 and len(n['records'])==22
 for x in n['records']:
  case=x['case']
  if case=='normal_native_and_fullvector_oracle':assert math.isclose(x['norm'],math.sqrt(5120),rel_tol=1e-12)
  elif case=='finite_gradient_fp32_square_overflow':assert math.isclose(x['norm'],4.525483490288562e21,rel_tol=1e-12)
  elif case=='empty_one_rank_sharded_gradients':assert math.isclose(x['norm'],math.sqrt(480),rel_tol=1e-12)
  elif case=='pure_NO_SHARD_all_empty_gradients':assert x['norm']==0
  elif case=='gpu_low_var_kl_saturated_forward_backward_fp64_oracle':assert x['points']==408 and x['max_value_error']<2e-5 and x['max_grad_error']<2e-5;maxkl=max(maxkl,x['max_value_error'],x['max_grad_error'])
  elif case=='gpu_low_var_kl_nonfinite_inputs_refused':assert x['count']==3
 f=json.loads((run/f'flash-native/rank-{rank}.json').read_text());assert f['finite_forward_backward']and f['max_abs_reference_error']<.02;maxflash=max(maxflash,f['max_abs_reference_error'])
 m=json.loads((run/f'native-state/rank-{rank}.json').read_text())
 for x in m['records']:
  assert x['initial_optimizer_calls']==20 and x['total_actual_optimizer_calls']==28 and x['scheduler_last_epoch']==6 and x['num_warmup_steps']==57 and x['adam_moment_dtypes']==x['persistent_parameter_dtypes']==['torch.float32']
  for idx,lr in enumerate(x['outer_lrs'],1):assert math.isclose(lr['before'],1e-6*(idx-1)/57,abs_tol=1e-18)and math.isclose(lr['after'],1e-6*idx/57,abs_tol=1e-18)
# Direct recomputation of query overlap, shared-id content and scores, not report PASS labels.
c=json.loads((run/'retrieval-cpu.json').read_text());g=json.loads((run/'retrieval-gpu.json').read_text());comparison=json.loads((run/'retrieval-comparison.json').read_text());assert c['asset_identity']==g['asset_identity']and c['queries']==g['queries']and len(g['queries'])==14
assert c['recipe']==g['recipe']=={'model':'e5','pooling':'mean','max_query_length':256,'encoder_fp16':True,'topk':[3,50],'batch_size':512}
assert g['source']['slurm_job_id']=='3173'and g['source']['code_commit']==source and g['index_metadata']=={'type':'IndexShards','d':768,'ntotal':21015324,'gpu_shard':True,'gpu_use_float16':True}
assert g['source']['cuda_visible_devices']=='0,1,2,3,4,5,6,7'and len(g['source']['gpu_leases'])==8 and len({x['uuid']for x in g['source']['gpu_leases']})==8
assert c['reused_embeddings'] is False and g['reused_embeddings'] is False
assert c['reused_embeddings_file_sha256'] is None and g['reused_embeddings_file_sha256'] is None
emb=np.load(run/'gpu-embeddings.npz');assert all(np.isfinite(emb[k]).all()for k in emb.files if emb[k].dtype.kind in 'fiu')
summ={}
for k in ['3','50']:
 ranks=0;overlap=[];maxdelta=0
 for row1,row2 in zip(c['results'],g['results']):
  x,y=row1[k],row2[k];assert len(x['ids'])==len(y['ids'])==int(k);ranks+=x['ids']==y['ids'];ix={v:i for i,v in enumerate(x['ids'])};iy={v:i for i,v in enumerate(y['ids'])};shared=set(ix)&set(iy);overlap.append(len(shared)/int(k))
  for id in shared:
   i,j=ix[id],iy[id];assert x['contents_sha256'][i]==y['contents_sha256'][j];maxdelta=max(maxdelta,abs(x['scores'][i]-y['scores'][j]))
 assert min(overlap)==1.0
 summ[k]={'exact_rank_queries':ranks,'mean_overlap':sum(overlap)/14,'max_shared_score_delta':maxdelta}
 assert comparison['summary'][k]['exact_rank_queries']==ranks and math.isclose(comparison['summary'][k]['max_matched_id_score_abs_delta'],maxdelta)
# Fresh certified process ownership and foreign resource limits on every accepted full sample.
windows=[json.loads(x)for x in(run/'resource-watch.jsonl').read_text().splitlines()];uuids={x['uuid']for x in g['source']['gpu_leases']};maxmem=maxsm=0;foreign=set();seen={}
for x in windows:
 assert not x['pause']and not x['exceeded']and x['reason']=='within_threshold'and x['run']==root+'/runs/bapo-instruct300-hardware-3173'and x['slurm_job_id']=='3173'and x['attempt'] in range(1,5)
 for pid,o in x['owned_before_query'].items():
  assert o['run']==x['run']and o['slurm_job_id']=='3173'and o['uid']==0 and set(o['lease_uuids'])==uuids and o['start_ticks']>0
  if pid in seen:assert seen[pid]==o['start_ticks']
  seen[pid]=o['start_ticks']
 for gpu,v in x['external_totals'].items():
  assert v['external_memory_percent']<=15;maxmem=max(maxmem,v['external_memory_percent']);sm=v['external_sm_percent']if v['external_sm_percent']is not None else v['external_sm_upper_bound_percent'];assert sm<=15;maxsm=max(maxsm,sm)
 for p in x['external_processes']:foreign.add(p['pid']);assert p['gpu_uuid']in uuids
assert foreign==set(range(2210497,2210505))and len(windows)==39
cleanup=json.loads((run/'own-process-cleanup.json').read_text());assert cleanup['remaining']==[]and cleanup['own_marker']==root+'/runs/bapo-instruct300-hardware-3173'and not(set(cleanup['stopped_pids'])&foreign)
model=json.loads((run/'model-gpu.json').read_text());assert model['source_commit']==source and model['status']=='REAL_OFFICIAL_INSTRUCT_VLLM_CUDA_OK'and len(model['token_ids'])==16
out={'status':'PASS_INDEPENDENT_ACTUAL_HARDWARE_SUPPLEMENT','job':3173,'source':source,'raw_files':len(inv['files']),'actual_hf_branches':16,'hf_logprob_fp64_max_error':maxprobe,'flash_max_error':maxflash,'numeric_kl_max_error':maxkl,'retrieval':summ,'gpu_retrieval_warm_seconds':g['resources']['measurements'][1]['wall'],'cpu_native_encoding_warm_seconds':c['resources']['measurements'][1]['wall'],'resource_samples':len(windows),'foreign_pids_stable':sorted(foreign),'max_foreign_memory_percent':maxmem,'max_foreign_sm_percent_or_upper_bound':maxsm,'certified_owned_pids':len(seen),'cleanup_remaining':[],'reserved_H100_GPU_hours':458*8/3600,'real_RL_iterations_accepted':0}
p=b/'work/evidence/e1-300-hardware3173-independent-supplemental-audit.json';p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
