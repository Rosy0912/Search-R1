import pathlib,json,hashlib,subprocess,re,math,itertools,collections,ast
ROOT=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-freeze-scalability-20261006')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
r=json.loads((ROOT/'logs/hardware-acceptance.json').read_text());source=(ROOT/'source-expected.txt').read_text().strip()
assert source=='b7119791b86951727cea36a1881e8a525c932494'
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
required.extend(ROOT/'repo'/name for name in ('search_r1/baselines/distillation.py','search_r1/baselines/objectives.py','verl/workers/actor/dp_actor.py','verl/workers/fsdp_workers.py','search_r1/baselines/actor_metrics.py'))
required.extend(p for p in (ROOT/'tools').iterdir()if p.is_file())
required.extend(run/n for n in ('model-gpu.json','retrieval-cpu.json','retrieval-gpu.json','gpu-embeddings.npz','parity-cpu.log','parity-gpu.log','parity-cpu-time.log','parity-gpu-time.log','launch-approval.json','tools-manifest.json','cpu-gate.json','cpu-gate-source.txt','tensordict-device-api.json','runtime-import-origins.log'))
required.extend(ROOT/'logs'/n for n in ('launch-approval.json','tools-manifest.json','cpu-gate.json','cpu-gate.log','cpu-gate-source.txt'))
for p in required:assert str(p)in bound,str(p)
assert (run/'source-commit.txt').read_text().strip()==source
slurm=(run/'slurm-terminal.txt').read_text()
for field,value in [('JobId',str(job)),('JobState','COMPLETED'),('ExitCode','0:0'),('DerivedExitCode','0:0'),('Restarts','0'),('NodeList','h100-5')]:
 assert re.search(r'(?:^|\s)'+field+'='+re.escape(value)+r'(?:\s|$)',slurm),field
assert json.loads((run/'own-process-cleanup.json').read_text())['remaining']==[]
for name in ('launch-approval.json','tools-manifest.json','cpu-gate.json','cpu-gate-source.txt'):
 assert sha(run/name)==sha(ROOT/'logs'/name)
def finite(value):
 if type(value)is float:assert math.isfinite(value)
 elif isinstance(value,dict):
  for v in value.values():finite(v)
 elif isinstance(value,list):
  for v in value:finite(v)
def validate_freeze_scalability(run, root, source, job, bound, allocated_uuids):
 """Recompute the forced fixture from bound raw files; never grant resource permission."""
 fixture_sha='2de29718f0a30fb90d0e1b814f146ee78a50e7680cc87640807fc89e9394ab80'
 watch_sha='e56781fefcd42b3d97f06af824312ac69fd318eb480093a66c4194d1f0fcaf74'
 scope='FORCED_OWN_FREEZE_FIXTURE_NOT_PRODUCTION_UNSAFE_OBSERVATION'
 directory=run/'freeze-scalability-gpu'
 def file(path):
  assert path.is_file()and not path.is_symlink()and path.resolve()==path
  assert bound.get(str(path))==sha(path),str(path)
 def parse(text):
  def pairs(items):
   result={}
   for k,v in items:
    assert k not in result,'Duplicate raw JSON key';result[k]=v
   return result
  return json.loads(text,object_pairs_hook=pairs,parse_constant=lambda value:(_ for _ in ()).throw(AssertionError('Nonfinite JSON')))
 def load(name):
  path=directory/name;file(path);value=parse(path.read_text());finite(value);return value
 def exactint(v, minimum=0):assert type(v)is int and v>=minimum
 def identity(v):
  assert v['source_commit']==source and type(v['job_id'])is int and v['job_id']==job
  assert v['run_root']==str(run)and v['fixture_sha256']==fixture_sha and v['watch_sha256']==watch_sha
  assert v['scope']==scope
 file(root/'tools/freeze_scalability_gpu_fixture.py');assert sha(root/'tools/freeze_scalability_gpu_fixture.py')==fixture_sha
 file(root/'tools/resource_watch.py');assert sha(root/'tools/resource_watch.py')==watch_sha
 result=load('result.json');identity(result)
 assert result['status']=='PASS_ACTUAL_NCCL8_301CPU_FORCED_OWN_FREEZE_ONLY'
 assert result['actual_unsafe_gpu_observation']is False
 for k in ('all_own_stopped','all_rank_counters_stable','all_rank_counters_resumed'):assert result[k]is True
 uuids=set(allocated_uuids);assert len(uuids)==8 and result['allocated_uuids']==sorted(uuids)
 exactint(result['foreign_signals']);assert result['foreign_signals']==0
 exactint(result['actual_ranks']);assert result['actual_ranks']==8
 exactint(result['cpu_processes'],301)
 assert type(result['freeze_seconds'])in (int,float)and type(result['wall_seconds'])in (int,float)and type(result['reserved_gpu_hours'])in (int,float)
 assert 0<result['freeze_seconds']<=5 and result['freeze_seconds']<result['wall_seconds'] and math.isclose(result['reserved_gpu_hours'],result['wall_seconds']*8/3600,rel_tol=1e-12)
 # All files produced before result, including every heartbeat/raw/finished log,
 # must be bound exactly. Extra/missing files cannot be silently discarded.
 actual={p.name for p in directory.iterdir()};assert actual==set(result['raw_files'])|{'result.json','forced-own-freeze-journal.jsonl'}
 for name,item in list(result['raw_files'].items())+[('forced-own-freeze-journal.jsonl',result['journal'])]:
  assert pathlib.Path(name).name==name
  path=directory/name;file(path);assert item['path']==str(path)and type(item['bytes'])is int and item['bytes']==path.stat().st_size and item['sha256']==sha(path)
 topology=load('topology.json');created={int(p):v for p,v in topology['created'].items()}
 phase=topology['phase_pid'];exactint(phase,1);cpus=topology['cpu_pids'];ranks=topology['rank_pids']
 assert len(cpus)==result['cpu_processes']and len(ranks)==8
 assert len(set(cpus+ranks+[phase]))==len(cpus)+9 and set(created)==set(cpus+ranks+[phase])and len(created)>=310
 for pid,v in created.items():
  for k in ('pid','ppid','start_ticks','uid'):exactint(v[k],1 if k in ('pid','start_ticks')else 0)
  assert v['pid']==pid and v['state']not in ('T','t','Z','X')
  if pid!=phase:assert v['ppid']==phase
 assert len({x['uid']for x in created.values()})==1
 creation=[parse(line)for line in (directory/'created-so-far.jsonl').read_text().splitlines()]
 assert len(creation)==len(created)and {x['process']['pid']for x in creation}==set(created)
 for x in creation:
  assert x['source_commit']==source and type(x['job_id'])is int and x['job_id']==job and x['run_root']==str(run)and x['creator_pid']==phase and x['process']==created[x['process']['pid']]
 selection=load('outer-sample-selection.json');assert selection['scope']=='SCHEDULING_ONLY_NOT_RESOURCE_PERMISSION'
 original=[line for line in (run/'resource-watch.jsonl').read_bytes().splitlines()if hashlib.sha256(line).hexdigest()==selection['raw_line_sha256']]
 assert len(original)==1 and parse(original[0])==selection['raw_row']
 slot=selection['raw_row'];assert slot['run']==str(run)and slot['slurm_job_id']==str(job)and slot['pause']is False and slot['reason']=='within_threshold'and 'event'not in slot
 assert type(selection['selection_time'])in (float,int)and type(slot['time'])in (float,int)and 0<=selection['selection_time']-slot['time']<=1.5
 paused=load('paused-proof.json');resumed=load('resumed-proof.json')
 assert paused['freeze_seconds']==result['freeze_seconds']
 states={int(p):v for p,v in paused['states'].items()};certs={int(p):v for p,v in paused['held_certificates'].items()}
 assert set(states)==set(certs)==set(created)
 for pid,v in states.items():
  assert v['state']in ('T','t')and all(v[k]==created[pid][k]for k in ('pid','ppid','start_ticks','uid'))
  cert=certs[pid]
  for k in ('pid','ppid','start_ticks','uid'):exactint(cert[k],1 if k in ('pid','start_ticks')else 0)
  assert cert['state']in ('R','S','D','T','t','I')
  assert all(cert[k]==created[pid][k]for k in ('pid','ppid','start_ticks','uid'))
  assert cert['run']==str(run)and cert['slurm_job_id']==str(job)
  assert cert['role']==('phase'if pid==phase else 'phase_descendant')
  assert cert['allocated_uuid_set']==sorted(uuids)and cert['gpu_leases_inherited']is True and cert['inherited_gpu_lease_uuids']==sorted(uuids)
  assert type(cert['parent_start_ticks'])is int and cert['parent_start_ticks']>0
  if pid!=phase:assert cert['parent_start_ticks']==created[phase]['start_ticks']
  assert isinstance(cert['command'],list)and len(cert['command'])>=2 and cert['command'][1]==str(root/'tools/freeze_scalability_gpu_fixture.py')
 foreign=paused['foreign_state'];exactint(result['foreign_pid'],1)
 assert foreign['pid']==paused['foreign_pid']==result['foreign_pid']and foreign['pid']not in created and foreign['state']not in ('T','t','Z','X')
 for k in ('start_ticks','uid'):exactint(foreign[k],1 if k=='start_ticks'else 0)
 assert resumed['finite_allreduce']==36. and resumed['all_eight_progress']is True
 keys={str(i)for i in range(8)}
 for counts in (paused['before'],paused['paused_first'],paused['paused_second'],resumed['before'],resumed['after']):
  assert set(counts)==keys
  for count in counts.values():exactint(count,1)
 assert paused['paused_first']==paused['paused_second']==resumed['before']
 observed_uuids=set();visibility=set()
 for rank,pid in enumerate(ranks):
  before=paused['before'][str(rank)];stopped=paused['paused_first'][str(rank)];after=resumed['after'][str(rank)]
  assert before>=3 and stopped>=before and after>=stopped+3
  raw=[parse(line)for line in (directory/f'rank-{rank}-raw.jsonl').read_text().splitlines()];assert len(raw)>=after
  for count,row in enumerate(raw,1):
   finite(row);identity(row)
   assert type(row['rank'])is int and row['rank']==rank and type(row['world_size'])is int and row['world_size']==8 and row['backend']=='nccl'
   exactint(row['start_ticks'],1);exactint(row['uid'])
   assert type(row['pid'])is int and row['pid']==pid and row['start_ticks']==created[pid]['start_ticks']and row['uid']==created[pid]['uid']
   assert type(row['cuda_device'])is int and row['cuda_device']==rank
   assert row['nccl_socket_ifname']=='lo'and row['nccl_ib_disable']=='1'
   assert type(row['completed_collectives'])is int and row['completed_collectives']==count and type(row['allreduce_value'])in (int,float)and row['allreduce_value']==36.
   assert type(row['time_unix'])in (int,float)and row['time_unix']>0
   if count>1:assert row['time_unix']>=raw[count-2]['time_unix']
   assert row['cuda_uuid']==raw[0]['cuda_uuid']and row['cuda_visible_devices']==raw[0]['cuda_visible_devices']
  observed_uuids.add(raw[0]['cuda_uuid']);visibility.add(raw[0]['cuda_visible_devices'])
  finished=load(f'rank-{rank}-finished.json');assert finished['status']=='ACTUAL_NCCL_RANK_FINISHED'and {k:v for k,v in finished.items()if k!='status'}==raw[-1]
  assert load(f'heartbeat-{rank}.json')==raw[-1]
 assert observed_uuids==uuids and len(visibility)==1 and len(next(iter(visibility)).split(','))==8
 journal=[parse(line)for line in (directory/'forced-own-freeze-journal.jsonl').read_text().splitlines()]
 for row in journal:finite(row);identity(row)
 stops=[x for x in journal if x['event']=='owned_freeze_signal'];continues=[x for x in journal if x['event']=='owned_resume_signal'];stable=[x for x in journal if x['event']=='owned_freeze_stable']
 assert len(stops)==len(continues)==len(created)and len(stable)==1
 assert {x['pid']for x in stops}=={x['pid']for x in continues}==set(created)
 assert stable[0]['pids']==sorted(created)and stable[0]['epochs']=={str(p):v['start_ticks']for p,v in created.items()}
 assert all(x['event']in ('owned_freeze_signal','owned_freeze_stable','owned_resume_signal','fixture_finished')for x in journal)
 for row in stops:
  exactint(row['pid'],1);exactint(row['start_ticks'],1);exactint(row['uid'])
  assert row['certificate']==certs[row['pid']]and row['start_ticks']==created[row['pid']]['start_ticks']and row['uid']==created[row['pid']]['uid']and row['lease_uuids']==sorted(uuids)
 for row in continues:
  exactint(row['pid'],1);exactint(row['start_ticks'],1);exactint(row['signal'],1)
  assert row['signal']==18 and row['start_ticks']==created[row['pid']]['start_ticks']
 assert max(journal.index(x)for x in stops)<journal.index(stable[0])<min(journal.index(x)for x in continues)
 assert all(x.get('pid')!=result['foreign_pid']for x in journal)
 last=journal[-1];assert last['event']=='fixture_finished'and last['forced_own_freeze']is True and last['actual_unsafe_gpu_observation']is False
 assert last['wall_seconds']==result['wall_seconds']and last['reserved_gpu_hours']==result['reserved_gpu_hours']
 markers=[parse(line.split('ACTUAL_FORCED_OWN_NCCL8_CPU301_FREEZE_JSON=',1)[1])for line in (run/'hardware.log').read_text().splitlines()if 'ACTUAL_FORCED_OWN_NCCL8_CPU301_FREEZE_JSON='in line]
 assert markers==[result]
 return dict(status='RAW_FORCED_OWN_FREEZE_RECOMPUTED',own_certificates=len(created),ranks=8,foreign_signals=0,actual_unsafe_gpu_observation=False)

def validate_state_snapshots(x, rank, run, bound):
 assert x['native_state_version']==3 and x['generation_rng_saved_and_restored']is True
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
 components=x['next_update_components'];assert set(components)=={'model','optimizer','scheduler','rng','generation_rng'}
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

def validate_native_stability(v, rank, source, root):
 finite(v)
 assert type(v['rank'])is int and type(v['world'])is int
 assert v['status']=='PASS_NATIVE_STABILITY_NCCL8'and v['rank']==rank and v['world']==8
 assert v['source_commit']==source and v['repo']==str(root/'repo')and v['sys_prefix']==str(root/'env')
 assert v['backend']=='nccl'and v['device']==f'cuda:{rank}'
 assert v['fixture_sha256']==sha(root/'tools/native-stability-fixture.py')
 expected_sources=('verl/workers/actor/dp_actor.py','verl/workers/fsdp_workers.py','search_r1/baselines/objectives.py','search_r1/baselines/actor_metrics.py')
 assert set(v['source_files_sha256'])==set(expected_sources)
 for name,digest in v['source_files_sha256'].items():assert sha(root/'repo'/name)==digest
 assert len(v['records'])==3
 guard,unguarded,hook=v['records']
 assert guard['case']=='actual_FSDP_actor_delta30_negativeA_guard'and guard['delta']==30 and guard['advantage']==-1
 assert math.isclose(guard['pg_loss'],math.exp(20),rel_tol=1e-12)and guard['preclip_norm']==0
 assert guard['optimizer_steps']==1 and guard['clamp_zero_backward']is True and guard['pre_step_gradients_present']is True and guard['pre_step_all_zero']is True
 assert guard['input_device_before_forward']=='cpu'and guard['forward_device']==f'cuda:{rank}'
 assert type(guard['pre_step_shard_numel'])is int and guard['pre_step_shard_numel']==2
 assert guard['pre_step_max_abs']==guard['pre_step_min']==guard['pre_step_max']==0
 assert guard['post_return_gradients_cleaned']is True and guard['parameters_unchanged_lr0']is True
 assert guard['pre_step_values']==[0.,0.]and guard['optimizer_lr']==guard['optimizer_weight_decay']==0
 assert unguarded['case']=='actual_FSDP_actor_delta30_negativeA_without_guard_counterexample'
 assert unguarded['fixture_baseline_method']=='grpo'and unguarded['delta']==30 and unguarded['advantage']==-1
 assert unguarded['optimizer_steps']==1 and unguarded['pre_step_gradients_present']is True and unguarded['pre_step_all_zero']is False
 assert type(unguarded['pre_step_shard_numel'])is int and unguarded['pre_step_shard_numel']==2
 assert math.isclose(unguarded['pg_loss'],math.exp(30),rel_tol=1e-12)and math.isclose(unguarded['preclip_norm'],4*math.exp(30),rel_tol=1e-12)
 for name in ('pre_step_max_abs','pre_step_min','pre_step_max'):assert math.isclose(unguarded[name],math.exp(30),rel_tol=1e-12)
 assert unguarded['post_return_gradients_cleaned']is True and unguarded['parameters_unchanged_lr0']is True
 assert len(unguarded['pre_step_values'])==2 and all(math.isclose(x,math.exp(30),rel_tol=1e-12)for x in unguarded['pre_step_values'])
 assert unguarded['optimizer_lr']==unguarded['optimizer_weight_decay']==0
 assert unguarded['input_device_before_forward']=='cpu'and unguarded['forward_device']==f'cuda:{rank}'

 assert hook['case']=='actual_worker_NCCL8_DataProto_return_concat'and hook['scheduler_calls']==1
 assert hook['local_original_unchanged']is True and hook['all_returns_identical']is True and hook['replicated_clock_mismatch_rejected']is True
 m=hook['global_metrics'];local=hook['local_original_metrics']
 for key,value in [('actor/pg_loss',4.5),('actor/kl_loss',5.5),('actor/entropy_loss',6.5),('actor/ppo_kl',7.5)]:assert m[key]==[value]*20
 assert all(math.isclose(x,.35,abs_tol=1e-15)for x in m['actor/pg_clipfrac'])and len(m['actor/pg_clipfrac'])==20
 assert m['actor/optimizer_steps']==20 and m['actor/scheduler_outer_steps']==5 and m['bapo/loss_tokens']==396
 assert m['actor/grad_norm']==[7.]*20
 assert local['actor/pg_loss']==[rank+1.]*20 and local['bapo/loss_tokens']==(rank+1)*11
 expected_local={'actor/pg_loss':[float(rank+1)]*20,'actor/kl_loss':[float(rank+2)]*20,'actor/entropy_loss':[float(rank+3)]*20,'actor/ppo_kl':[float(rank+4)]*20,'actor/pg_clipfrac':[rank*.1]*20,'actor/grad_norm':[7.]*20,'bapo/grad_norm_min':7.,'bapo/grad_norm_max':7.,'actor/optimizer_steps':20,'actor/skipped_empty_minibatches':0,'actor/kl_coef':.001,'bapo/loss_tokens':(rank+1)*11,'bapo/old_logprob_nonfinite':0,'bapo/new_logprob_nonfinite':0}
 assert local==expected_local
 assert set(m)==set(local)|{'mfu/actor','actor/lr','actor/lr_before_outer','actor/scheduler_outer_steps'}
 assert m['bapo/old_logprob_nonfinite']==m['bapo/new_logprob_nonfinite']==0 and m['actor/kl_coef']==.001
 assert m['bapo/grad_norm_min']==m['bapo/grad_norm_max']==7. and m['actor/skipped_empty_minibatches']==0
 assert type(m['actor/optimizer_steps'])is int and type(m['actor/scheduler_outer_steps'])is int
 assert math.isclose(m['mfu/actor'],.0125,rel_tol=1e-15)
 assert math.isclose(m['actor/lr_before_outer'],4e-6/114,rel_tol=1e-15)and math.isclose(m['actor/lr'],5e-6/114,rel_tol=1e-15)

for rank in range(8):
 p=run/'native-stability'/f'rank-{rank}.json';assert str(p)in bound,str(p)
 v=json.loads(p.read_text());finite(v);validate_native_stability(v,rank,source,ROOT)

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
    assert x['num_warmup_steps']==114 and x['scheduler_last_epoch']==6 and x['same_model_adam_scheduler_objects']is True and x['next_update_full_state_exact_equal']is True
    assert x['native_cpu_offload']is False and x['sharding']=='FULL_SHARD'and x['mesh_shape']==[8]
    assert len(x['gradient_norms'])==28 and all(type(n)in (int,float)and math.isfinite(n)and n>=0 for n in x['gradient_norms'])
    assert len(x['outer_lrs'])==5
    for outer,lr in enumerate(x['outer_lrs'],1):
     assert lr['outer']==outer and math.isclose(lr['before'],1e-6*(outer-1)/114,abs_tol=1e-18,rel_tol=1e-12)and math.isclose(lr['after'],1e-6*outer/114,abs_tol=1e-18,rel_tol=1e-12)
    assert x['oracle_sixth_lr']==x['reloaded_sixth_lr']and x['oracle_sixth_lr']['outer']==6
    assert math.isclose(x['oracle_sixth_lr']['before'],1e-6*5/114,abs_tol=1e-18,rel_tol=1e-12)and math.isclose(x['oracle_sixth_lr']['after'],1e-6*6/114,abs_tol=1e-18,rel_tol=1e-12)
    if x['profile']=='production-nested-bf16-manual':
     assert x['production_like']is True and x['nested_fsdp_wrappers']>=3 and x['manual_param_grad_optimizer_offload']is True
     assert x['param_compute_dtype']=='bfloat16'and x['reduction_dtype']==x['buffer_dtype']=='float32'
   assert v['independent_scalar_warmup_oracle']['optimizer_calls']==v['independent_scalar_warmup_oracle']['scheduler_calls']==114
   scalar=v['independent_scalar_warmup_oracle'];assert scalar['first_lr']==0 and math.isclose(scalar['last_before_lr'],1e-6*113/114,rel_tol=1e-15)and scalar['after114_lr']==1e-6
   assert set(v['reviewed_sources'])=={str(ROOT/'repo'/p)for p in ('verl/workers/fsdp_workers.py','search_r1/baselines/full_state.py','search_r1/baselines/checkpoint_gate.py','search_r1/baselines/gradient_norm.py','verl/workers/sharding_manager/fsdp_vllm.py')}
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
assert model['model_path']=='/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-stabilityfix-r2-20261005/assets/qwen2.5-3b-instruct-aa8e725'
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
# The hardware receipt also binds the real production eight-rank restore and
# its read-only driver. The allocated training run performs its own fresh
# permission validation; this gate never manufactures a training permission.
from omegaconf import OmegaConf
from search_r1.baselines.recovery import validate_gpu_proof
import importlib.util
spec=importlib.util.spec_from_file_location('bound_hardware_guard_journal',ROOT/'tools/guard_journal.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
summary=helper.replay(run/'resource-watch.jsonl',job,str(run),source,ROOT/'tools/resource_watch.py')
assert summary==json.loads((run/'guard-summary.json').read_text())
validate_freeze_scalability(run,ROOT,source,job,bound,summary['allocated_uuids'])
for p in [run/'guard-summary.json',run/'recovery-manifest.json',run/'recovery-permission.json',run/'resolved-config.yaml',run/'train.log']+[run/'recovery-witness'/f'rank_{i}.json'for i in range(8)]+[run/'recovery-witness/driver.json']:
 assert bound.get(str(p))==sha(p),str(p)
manifest=json.loads((run/'recovery-manifest.json').read_text())
assert manifest['target_job_id']==job and manifest['target_run_root']==str(run)and manifest['target_source_commit']==source
assert manifest['step']==25 and manifest['next_step']==26
config=OmegaConf.to_container(OmegaConf.load(run/'resolved-config.yaml'),resolve=True)
assert config['trainer']['resume_verify_only']is True
assert config['trainer']['resume_full_state']==str(run/'recovery-manifest.json')
train=(run/'train.log').read_text()
assert 'BAPO_TRAIN_METRICS_JSON='not in train
rank_records=[json.JSONDecoder().raw_decode(line.split('BAPO_RECOVERY_RANK_JSON=',1)[1].lstrip())[0]for line in train.splitlines()if 'BAPO_RECOVERY_RANK_JSON='in line]
driver_records=[json.JSONDecoder().raw_decode(line.split('BAPO_RECOVERY_DRIVER_JSON=',1)[1].lstrip())[0]for line in train.splitlines()if 'BAPO_RECOVERY_DRIVER_JSON='in line]
assert len(rank_records)==8 and sorted(rank_records,key=lambda x:x['rank'])==[json.loads((run/'recovery-witness'/f'rank_{i}.json').read_text())for i in range(8)]
assert driver_records==[json.loads((run/'recovery-witness/driver.json').read_text())]
assert r['real_production_restore_independent_review_passed']is True
old_config=OmegaConf.to_container(OmegaConf.load(manifest['resolved_config']['path']),resolve=True)
validate_gpu_proof(r,manifest,old_config,ROOT,hardware_self_review=True)
# Real production native v3 save/reload must retain generation RNG as well.
v3_records=[json.JSONDecoder().raw_decode(line.split('BAPO_RECOVERY_V3_RANK_JSON=',1)[1].lstrip())[0]for line in train.splitlines()if 'BAPO_RECOVERY_V3_RANK_JSON='in line]
assert len(v3_records)==8 and {x['rank']for x in v3_records}==set(range(8))
tp=old_config['actor_rollout_ref']['rollout']['tensor_model_parallel_size']
assert type(tp)is int and tp>0 and 8%tp==0
for v in v3_records:
 assert type(v['rank'])is int and type(v['pid'])is int and v['pid']>0
 rank=v['rank'];p=run/'recovery-native-v3-roundtrip'/f'rank_{rank}.pt';rp=p.with_suffix('.json')
 assert bound.get(str(p))==sha(p)==v['sha256']and bound.get(str(rp))==sha(rp)
 assert json.loads(rp.read_text())==v and v['path']==str(p)
 assert v['status']=='EXACT_FULL_STATE_RELOAD_OK'and v['native_state_version']==3 and v['roundtrip']is True
 assert v['world_size']==8 and v['step']==v['scheduler_last_epoch']==25
 assert v['model_state_dict_type']=='SHARDED_STATE_DICT'and v['model_dtensor_entries']==435 and v['optimizer_state_entries']==37
 assert v['pid']==next(x['new_pid']for x in rank_records if x['rank']==rank)
 assert re.fullmatch('[0-9a-f]{64}',v['state_digest'])and v['state_digest']==v['reloaded_state_digest']
 assert re.fullmatch('[0-9a-f]{64}',v['generation_rng_before_digest'])and v['generation_rng_before_digest']==v['generation_rng_after_digest']
 topology=v['generation_rng_topology'];assert topology['version']==1 and type(topology['torch_random_states_numel'])is int and topology['torch_random_states_numel']>0
 assert type(topology['gen_random_states_numel'])is int and topology['gen_random_states_numel']>0
 assert topology['device_mesh']['shape']==[8//tp,tp]and topology['device_mesh']['names']==['dp','infer_tp']
 assert topology['device_mesh']['ranks']==[list(range(i*tp,(i+1)*tp))for i in range(8//tp)]
assert r['production_native_v3_roundtrip_independent_review_passed']is True


subprocess.run([str(ROOT/'env/bin/python'),str(ROOT/'tools/verify_shared_inputs.py')],check=True)
subprocess.run([str(ROOT/'env/bin/python'),str(ROOT/'tools/verify_launch.py'),'formal300'],check=True)
print('NEW_INSTRUCT300_HARDWARE_RECEIPT_GATE_PASS',flush=True)
