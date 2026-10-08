"""Detached actual evidence contracts; never manufactures approvals."""
import hashlib,json,math,os,pathlib
ROOT=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd')
METHOD='e23b1cdb73736723ff422a9c041f2678b5e8f04cb9a496bd219ad87827b5ef7d'
D23='d23c56c78fdb09f40b7859e02b14eeddbe83a863'
R2='9b382b777c3a22b497639eb7bee37c8ae1c1461dc978e2ea95765f480bec7b30'
def require(x,message):
 if not x:raise RuntimeError(message)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(x):return hashlib.sha256(canonical(x)).hexdigest()
def file_ref(ref):
 p=pathlib.Path(ref['path']);require(p.is_file() and not p.is_symlink() and sha(p)==ref['sha256'],'missing/drifted actual raw reference');return p
def artifact(ref):return json.loads(file_ref(ref).read_text())
def own(p):
 p=pathlib.Path(p).resolve();require(p.is_relative_to(ROOT),'foreign path');return p

def numerical(raw,parameter_sha,uuids):
 require(raw['status']=='SCORER_NUMERICAL_RELEASE_COMPLETE_UNREVIEWED_NOT_TRAINING_READY' and max(raw['actual_tokens_per_row'])==6144,'new32B complete longest-context numerical proof required')
 phases=raw['phase_records'];pairs=[p for p in phases if p['kind']=='PIPELINE'];require(len(pairs)==2 and [p['repeat'] for p in pairs]==[0,1],'two actual originalGPU parity rounds')
 require(all(p['allclose'] is True and p['atol']==p['rtol']==1e-4 for p in pairs),'strict numerical gate failed')
 for p in phases:
  r=p['record'];a,b=r['initial_cpu'],r['released_cpu'];alloc=r['allocators_released']
  require(r['status']=='RELEASED' and r['original_method_sha256']==METHOD and a['parameter_count']==32763876352 and a['parameter_sha256']==parameter_sha,'wrong teacher/math')
  require(all(a[k]==b[k] for k in ('parameter_sha256','buffer_sha256','parameter_count','parameter_bytes')) and b['parameter_devices']==['cpu'] and all(d=='cpu' for d in b['buffer_devices']),'frozen CPU boundary drift')
  require(len(alloc)==8 and {x['uuid'] for x in alloc}==set(uuids) and all(x['allocated_bytes']==x['reserved_bytes']==0 for x in alloc),'all8 actual release not zero')
 return raw

def actual_worker_ready(receipts,uuids,job,read_identity):
 require(len(receipts)==8 and {r['rank'] for r in receipts}==set(range(8)) and {r['gpu_uuid'] for r in receipts}==set(uuids),'actual8 actor rank/UUID coverage')
 for r in receipts:
  actual=read_identity(r['pid']);require(actual['starttime']==r['identity']['starttime'] and actual['cgroups']==r['identity']['cgroups'] and str(r['job'])==str(job) and any(('/job_'+str(job)+'/') in (v+'/') for v in actual['cgroups']),'actor identity drift')
  require(r['initialized'] is True and r['vllm_model_parameter_count']>0 and r['gpu_uuid'] in r['actual_gpu_uuids'],'worker/vLLM not actually initialized on lease')
  require(type(r['allocated_bytes']) is int and type(r['reserved_bytes']) is int,'missing actual CUDA allocator observation')
 return receipts

def release_and_ack(directory,step,binding):
 root=pathlib.Path(directory);release=json.loads((root/f'global-release-step{step}.json').read_text());ack=json.loads((root/f'global-ack-step{step}.json').read_text())
 for r in (release,ack):
  require(r['outer_step']==step and all(r[k]==binding[k] for k in ('run_id','code_commit','config_sha256','data_identity','teacher_receipt_sha256','source_sha256')),'outer/run identity drift')
 require(release['status']=='GLOBAL_ALL8_RELEASED' and ack['status']=='ALL8_ACKED_ACTOR_RELEASE_BARRIER' and ack['ranks']==list(range(8)) and ack['nonce']==release['nonce'],'all8 ACK before actor missing')
 require(ack['release_receipt']['sha256']==sha(root/ack['release_receipt']['name']),'released raw SHA drift')
 require(json.loads((root/ack['release_receipt']['name']).read_text())==release,'ACK raw release mismatch')
 r=release['original_scorer_receipt'];a,b=r['initial_cpu'],r['released_cpu'];alloc=r['allocators_released']
 require(r['status']=='RELEASED' and r['original_method_sha256']==METHOD and r['all_threads_joined'] is True and all(a[k]==b[k] for k in ('parameter_sha256','buffer_sha256','parameter_count','parameter_bytes')),'frozen scorer boundary failed')
 require(b['parameter_devices']==['cpu'] and all(d=='cpu' for d in b['buffer_devices']) and b['requires_grad_parameters']==b['grad_parameters']==b['training_modules']==0,'teacher not frozen CPU')
 require(a['parameter_sha256']==binding['fp32_parameter_sha256'] and len(alloc)==8 and {x['uuid'] for x in alloc}==set(binding['leased_uuids']) and all(x['allocated_bytes']==x['reserved_bytes']==0 for x in alloc),'actual leased8 allocator0 required')
 return release,ack

def read_complete_jsonl(path):
 raw=pathlib.Path(path).read_bytes();return [json.loads(line) for line in raw.splitlines() if line.strip()] if not raw or raw.endswith(b'\n') else [json.loads(line) for line in raw.rsplit(b'\n',1)[0].splitlines() if line.strip()]
def actual_acceptance(run,steps,binding):
 root=pathlib.Path(run);steps=list(range(1,steps+1));rawfiles={}
 for rank in range(8):
  path=root/'audit'/f'{rank}.jsonl';rows=[r for r in read_complete_jsonl(path) if r['kind']=='actor_outer' and r['outer_step'] in steps]
  require(sorted(r['outer_step'] for r in rows)==steps,'missing/duplicate actual actor update rank'+str(rank))
  for r in rows:
   require(all(r[k]==binding[k] for k in ('run_id','config_sha256','data_identity')) and r['commit']==binding['code_commit'],'actor raw identity mismatch')
   require(r['rank']==rank and r['rows']==320 and r['epochs']==2 and r['optimizer_steps']==20 and r['grad_finite']==r['grad_nonzero']==20 and r['skipped_empty_minibatches']==0 and r['fixed_unchanged'] and r['shard_layout_unchanged'],'actual rank micro64/mini256/PPO2 update failed')
   require(len(r['grad_norms'])==20 and all(v is not None and math.isfinite(v) and v>0 for v in r['grad_norms']) and len(r['micro_stats'])==80,'actual backward/optimizer evidence absent')
   events=read_complete_jsonl(root/'workers'/f'{rank}.jsonl')
   before=[e for e in events if e['event']=='ALL8_RELEASED_ACKED_BEFORE_ACTOR_UPDATE' and e['outer_step']==r['outer_step']]
   after=[e for e in events if e['event']=='ORIGINAL_ACTOR_UPDATE_RETURNED' and e['outer_step']==r['outer_step']]
   ack=json.loads((pathlib.Path(binding['ipc_directory'])/f"global-ack-step{r['outer_step']}.json").read_text())
   require(len(before)==len(after)==1 and before[0]['pid']==after[0]['pid']==r['pid'] and ack['ack_epoch']<=before[0]['epoch']<=after[0]['epoch'],'actual release/ACK-before-optimizer ordering absent')
   for micro in r['micro_stats']:
    for key in ('current_log_probs','ratios','policy_loss'):
     stats=micro[key];require(stats['count']>0 and stats['finite']==stats['count'],'missing/nonfinite actual micro forward')
   require({'old_log_probs','advantages','policy_weights'}<=set(r['tensors']),'Missing actual actor audit tensors')
   for stats in r['tensors'].values():require(stats['count']>0 and stats['finite']==stats['count'],'missing/nonfinite actor tensor')
  require(any(r['outer_step']>1 and r['shard_delta']['nonzero']>0 for r in rows),'no actual shard change after source zero-LR outer1')
  snapshot=root/'evidence'/('first'+str(steps[-1]))/f'actor-{rank}.jsonl';snapshot.parent.mkdir(parents=True,exist_ok=True)
  actual_bytes=path.read_bytes();complete=actual_bytes if actual_bytes.endswith(b'\n') else actual_bytes.rsplit(b'\n',1)[0]+b'\n'
  content=b''.join(line+b'\n' for line in complete.splitlines() if line.strip() and (lambda r:r.get('kind')=='actor_outer' and r.get('outer_step') in steps)(json.loads(line)))
  if snapshot.exists():require(snapshot.read_bytes()==content,'detached acceptance snapshot drift')
  else:
   with snapshot.open('xb') as stream:stream.write(content);stream.flush();os.fsync(stream.fileno())
  rawfiles[str(snapshot)]=sha(snapshot)
 driver=root/'audit/driver.jsonl';rows=[r for r in read_complete_jsonl(driver) if r['kind']=='driver_outer' and r['outer_step'] in steps]
 require(sorted(r['outer_step'] for r in rows)==steps,'missing actual driver outer')
 for r in rows:
  require(all(r[k]==binding[k] for k in ('run_id','config_sha256','data_identity')) and r['commit']==binding['code_commit'],'driver identity drift')
  require(r['teacher_targets_none'] and r['teacher_scored_tokens_delta']>0 and r['tensors']['advantages']['nonzero']>0,'noGold/effective advantage missing')
  require({'teacher_log_probs','old_log_probs','token_level_rewards','returns','advantages','policy_weights'}<=set(r['tensors']),'Missing actual driver audit tensors')
  for stats in r['tensors'].values():require(stats['count']>0 and stats['finite']==stats['count'],'missing/nonfinite driver tensor')
  release_and_ack(binding['ipc_directory'],r['outer_step'],binding)
 require(any(r['observation_tokens']>0 for r in rows),'no actual search observations')
 checkpoint=root/'checkpoints/actor'/f'global_step_{steps[-1]}';state=root/'checkpoints/algorithm'/f'global_step_{steps[-1]}/state.json'
 require((checkpoint/'config.json').is_file(),'actual acceptance checkpoint missing');s=json.loads(state.read_text());require(s['method']=='topd' and s['global_step']==steps[-1] and s['resume_supported'] is False,'algorithm checkpoint state missing')
 require(pathlib.Path(s['teacher_path']).resolve()==pathlib.Path(binding['teacher_path']).resolve(),'checkpoint teacher provenance drift')
 from hf_inventory import complete_hf
 hf=complete_hf(checkpoint,binding['student_model_directory'])
 effect=validation_effect(root,binding) if steps[-1]>=5 else None
 return {'scope':'ACTUAL_FIRST5_UPDATES_UNREVIEWED_CONTINUING_FORMAL500' if steps[-1]==5 else ('ACTUAL_FORMAL500_UNREVIEWED' if steps[-1]==500 else 'ACTUAL_RESOURCE_PROFILE_UPDATES_UNREVIEWED_NOT_ALGORITHM_ACCEPTANCE'),'completed_actual_outer_steps':steps,'raw_rank_files':rawfiles,'driver_selected_records_sha256':digest(rows),'driver_observed_file_sha256':sha(driver),'observed_file_hash_scope':'MUTABLE_SOURCE_FILE_AT_OBSERVATION_SELECTED_RECORDS_BOUND_SEPARATELY','checkpoint':str(checkpoint),'checkpoint_full_hf_inventory':hf,'initial_accept5_actual_effect64':effect,'algorithm_state_sha256':sha(state),'status':'ACTUAL_EVIDENCE_COMPLETE_UNREVIEWED'}

def resolved_recipe(config):
 expected={'baselines.method':'topd','baselines.topd.alpha':.1,'baselines.distillation.privileged_answers':False,
 'baselines.topd.teacher_backend':'local_gpu_fp32_shared_pipeline32b','baselines.distillation.chunk_size':16,
 'data.train_batch_size':512,'data.val_data_num':64,'data.val_batch_size':64,
 'data.max_start_length':2048,'data.max_prompt_length':4096,'data.max_response_length':500,
 'actor_rollout_ref.actor.ppo_mini_batch_size':256,'actor_rollout_ref.actor.ppo_micro_batch_size':64,
 'actor_rollout_ref.actor.ppo_epochs':2,'actor_rollout_ref.rollout.n_agent':5,
 'actor_rollout_ref.rollout.tensor_model_parallel_size':1,'actor_rollout_ref.rollout.gpu_memory_utilization':.6,
 'trainer.n_gpus_per_node':8,'trainer.nnodes':1,'trainer.total_training_steps':501,'trainer.acceptance_step':5,
 'trainer.smoke_only':False,'trainer.save_freq':100,'trainer.test_freq':50,'baselines.bapo.stage_validation_num':64,
 'retriever.url':'http://127.0.0.1:18085/retrieve'}
 for key,value in expected.items():
  current=config
  for part in key.split('.'):current=current[part]
  require(current==value,'Actual resolved recipe drift '+key)
 return expected


def verify_tree(base,files,manifest_path):
 base=pathlib.Path(base);require(base.is_dir() and not base.is_symlink(),'source tree missing/symlink')
 actual=set()
 for directory,dirs,names in os.walk(base,followlinks=False):
  for name in dirs:require(not pathlib.Path(directory,name).is_symlink(),'source directory symlink')
  dirs[:]=[d for d in dirs if d!='.git']
  for name in names:
   path=pathlib.Path(directory,name)
   require(not path.is_symlink() and path.is_file(),'source nonregular member')
   if path.resolve()==pathlib.Path(manifest_path).resolve():continue
   actual.add(str(path.relative_to(base)))
 require(actual==set(files),'source exact inventory mismatch')
 for name,value in files.items():
  rel=pathlib.Path(name);require(not rel.is_absolute() and '..' not in rel.parts,'source alias escape')
  path=base/rel
  expected=value['sha256'] if isinstance(value,dict) else value
  require(sha(path)==expected and (not isinstance(value,dict) or path.stat().st_size==value['size']),'source full bytes drift')
 return files


def next_rollout_rebuilt(records,workers):
 require(len(records)==2 and records[0]['step']==1 and records[1]['step']==2,'Actual two profile rounds required')
 left=records[0]['scoring_completed_epoch'];right=records[1]['scoring_started_epoch']
 require(left<right,'Teacher scoring phase order invalid')
 result=[]
 require(set(workers)==set(range(8)),'Missing actual rank KV observer')
 for rank,rows in workers.items():
  actual=[r for r in rows if r['event']=='ACTUAL_REBUILT_KV_BEFORE_VLLM_GENERATE' and left<r['epoch']<right and all(not k['cache_engine_is_none'] and not k['gpu_cache_is_none'] for k in r['kv'])]
  require(actual,'No actual rebuilt next-rollout KV between scoring phases rank'+str(rank))
  result.extend(actual)
 return result


def telemetry_pair(review,guard,runtime):
 require(review['status']=='PASS' and review['guard_source_sha256']==guard['sha256'] and review['runtime_source_sha256']==runtime['sha256'] and review['actual_native_dual_owner_coverage_passed'] is True,'Exact full telemetry runtime/guard native reviewed pair required')
 return review


def explicit_accept_validation(step,acceptance_step,period):
 require(acceptance_step==5 and period==50,'Reviewed original acceptance/periodic budget changed')
 return step==acceptance_step and step%period!=0

def validation_effect(root,binding):
 rows=read_complete_jsonl(pathlib.Path(root)/'validation64.jsonl')
 initial=[r for r in rows if r['step']==0];accept=[r for r in rows if r['step']==5]
 require(len(initial)==len(accept)==1,'Exactly one actual initial64/accept5 effect check required')
 for r in initial+accept:
  require(int(r['metrics']['val/qa_count/all'])==64 and all(r[k]==binding[k] for k in ('run_id','config_sha256','data_identity')) and r['commit']==binding['code_commit'],'Actual effect64 validation identity/count mismatch')
 require(initial[0]['question_identity_sha256']==accept[0]['question_identity_sha256'],'Initial/accept5 questions differ')
 return dict(initial=initial[0],accept5=accept[0],status='MEASURED_ORIGINAL_VALIDATOR64_UNREVIEWED_NO_IMPROVEMENT_CLAIM')
