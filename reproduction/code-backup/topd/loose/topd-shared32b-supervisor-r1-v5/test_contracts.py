import copy,json,os,sys,types
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parent))
from contracts import *
from lifecycle import StartupPhases,publish,wait
from dependencies import recipe

class Guard:
 def __init__(self):self.closed=True
 def close_startup(self):self.closed=True

def test_teacher_ready_not_global_startup():
 guard=Guard();phase=StartupPhases(guard);phase.arm();phase.teacher()
 assert not guard.closed and not phase.ready_check()
 phase.workers([1],lambda r:require(r==[1],'actual'))
 assert guard.closed and phase.ready_check()

def test_bad_workers_cannot_close_startup():
 guard=Guard();phase=StartupPhases(guard);phase.arm();phase.teacher()
 with pytest.raises(RuntimeError):phase.workers([],lambda r:require(bool(r),'missing actual'))
 assert not guard.closed

def test_early_global_ready_rejected():
 phase=StartupPhases(Guard());phase.arm()
 with pytest.raises(RuntimeError):phase.workers([1],lambda r:None)

def test_private_atomic_actual_barrier(tmp_path):
 path=tmp_path/'ready';publish(path,dict(status='ACTUAL'))
 assert path.stat().st_mode&0o077==0 and wait(path,[],[],.1)['status']=='ACTUAL'
 assert not list(tmp_path.glob('*.tmp'))

def test_exited_owned_child_blocks_barrier(tmp_path):
 class Child:
  def poll(self):return 0
 with pytest.raises(RuntimeError,match='exited'):wait(tmp_path/'absent',[Child()],[],.1)

def test_failed_monitor_blocks_barrier(tmp_path):
 with pytest.raises(RuntimeError,match='monitor'):wait(tmp_path/'absent',[],['strict foreign failure'],.1)

def test_partial_jsonl_retained_not_false_complete(tmp_path):
 p=tmp_path/'audit';p.write_bytes(b'{"step":1}\n{"step":')
 assert read_complete_jsonl(p)==[dict(step=1)]

def worker_rows():
 return [dict(rank=i,gpu_uuid='GPU-'+str(i),pid=100+i,identity=dict(starttime=55,cgroups=['/job_8/step']),job='8',initialized=True,vllm_model_parameter_count=3085938688,actual_gpu_uuids=['GPU-'+str(i)],allocated_bytes=100,reserved_bytes=200) for i in range(8)]
def identity(pid):return dict(starttime=55,cgroups=['/job_8/step'])
def test_actual_eight_worker_coverage():
 rows=worker_rows();assert actual_worker_ready(rows,['GPU-'+str(i) for i in range(8)],'8',identity)==rows
@pytest.mark.parametrize('field,value',[('rank',0),('gpu_uuid','GPU-0'),('initialized',False),('vllm_model_parameter_count',0),('allocated_bytes',None),('job','other')])
def test_worker_counterexamples(field,value):
 rows=worker_rows();rows[7][field]=value
 with pytest.raises(RuntimeError):actual_worker_ready(rows,['GPU-'+str(i) for i in range(8)],'8',identity)

def test_worker_pid_starttime_drift():
 rows=worker_rows();rows[4]['identity']['starttime']=999
 with pytest.raises(RuntimeError):actual_worker_ready(rows,['GPU-'+str(i) for i in range(8)],'8',identity)

def test_formal_recipe_is_one_500_run_not_smoke():
 overrides=recipe('formal500','/run','/teacher','/ipc')
 assert 'trainer.total_training_steps=501' in overrides and 'trainer.acceptance_step=5' in overrides and 'trainer.smoke_only=false' in overrides
 assert 'actor_rollout_ref.actor.ppo_micro_batch_size=64' in overrides

def test_old_cpu_or_hardware_cannot_replace_actual32b():
 with pytest.raises((KeyError,RuntimeError)):numerical(dict(status='CPU_PASS'),'',[])

def test_release_unknown_ack_no_advancement(tmp_path):
 binding=dict(run_id='x',code_commit=D23,config_sha256='c',data_identity='d',teacher_receipt_sha256='t',source_sha256='s')
 release=dict(binding,outer_step=1,status='GLOBAL_ALL8_RELEASED',nonce='n');ack=dict(binding,outer_step=1,status='UNKNOWN',nonce='n',ranks=list(range(8)))
 (tmp_path/'global-release-step1.json').write_text(json.dumps(release));(tmp_path/'global-ack-step1.json').write_text(json.dumps(ack))
 with pytest.raises(RuntimeError,match='ACK'):release_and_ack(tmp_path,1,binding)

def test_no_detached_actor_audit_no_acceptance(tmp_path):
 with pytest.raises(FileNotFoundError):actual_acceptance(tmp_path,5,{})

def test_external_tree_extra_executable_rejected(tmp_path):
 p=tmp_path/'a.py';p.write_text('canonical');manifest=tmp_path/'files.json';manifest.write_text('{}')
 files={'a.py':dict(sha256=sha(p),size=p.stat().st_size)}
 assert verify_tree(tmp_path,files,manifest)==files
 (tmp_path/'shadow.py').write_text('extra executable')
 with pytest.raises(RuntimeError,match='inventory'):verify_tree(tmp_path,files,manifest)

def test_external_tree_directory_alias_rejected(tmp_path):
 (tmp_path/'alias').symlink_to(tmp_path,target_is_directory=True)
 with pytest.raises(RuntimeError,match='symlink'):verify_tree(tmp_path,{},tmp_path/'absent.json')

def test_observer_follows_actual_pinned_spmd_worker_schema(monkeypatch):
 import importlib.util
 base=ROOT/'repo-shared32b' if os.environ.get('AGENT_ROOT')==str(ROOT) else Path(__file__).resolve().parents[1]/'topd-shared32b-integration-candidate'
 ast=__import__('ast');tree=ast.parse((base/'verl/third_party/vllm/vllm_v_0_6_3/spmd_gpu_executor.py').read_text())
 fields={n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute) and isinstance(n.ctx,ast.Store) and isinstance(n.value,ast.Name) and n.value.id=='self'}
 assert 'worker' in fields
 # Load real external observer helpers while refusing all CUDA/import side effects.
 worker=types.ModuleType('verl.workers.fsdp_workers');worker.ActorRolloutRefWorker=object
 decorator=types.ModuleType('verl.single_controller.base.decorator');decorator.register=lambda **kw:lambda f:f;decorator.Dispatch=types.SimpleNamespace(ONE_TO_ALL=1,DP_COMPUTE_PROTO=2)
 monkeypatch.setitem(sys.modules,'verl.workers.fsdp_workers',worker);monkeypatch.setitem(sys.modules,'verl.single_controller.base.decorator',decorator)
 spec=importlib.util.spec_from_file_location('observed_fixture',Path(__file__).with_name('shared_supervision_hooks.py'));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 actual=types.SimpleNamespace(cache_engine=None,gpu_cache=None)
 engine=types.SimpleNamespace(llm_engine=types.SimpleNamespace(model_executor=types.SimpleNamespace(worker=actual)))
 assert module.kv_state(engine)==[dict(cache_engine_is_none=True,gpu_cache_is_none=True)]
 actual.cache_engine=object();actual.gpu_cache=object()
 assert module.kv_state(engine)==[dict(cache_engine_is_none=False,gpu_cache_is_none=False)]


def test_profile_calls_actual_source_public_methods():
 import ast
 base=ROOT/'repo-shared32b' if os.environ.get('AGENT_ROOT')==str(ROOT) else Path(__file__).resolve().parents[1]/'topd-shared32b-integration-candidate'
 tree=ast.parse((base/'search_r1/baselines/controller.py').read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Controller');names={n.name for n in cls.body if isinstance(n,ast.FunctionDef)}
 profile=ast.parse(Path(__file__).with_name('hardware_profile.py').read_text())
 calls={n.func.attr for n in ast.walk(profile) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Attribute) and n.func.value.attr=='baseline_controller'}
 assert calls<=names and {'compute_advantages','before_actor_update','after_actor'}<=calls
 assert not any(isinstance(n,ast.Constant) and n.value=='advantages' for n in ast.walk(profile))

def acceptance_fixture(tmp_path):
 binding=dict(run_id='cpu_fixture',code_commit=D23,config_sha256='c',data_identity='d',teacher_receipt_sha256='t',source_sha256='s',fp32_parameter_sha256='p',leased_uuids=['GPU-'+str(i) for i in range(8)],teacher_path='/cpu/fixture',ipc_directory=str(tmp_path/'ipc'))
 root=tmp_path;root.joinpath('audit').mkdir();root.joinpath('ipc').mkdir();root.joinpath('workers').mkdir()
 state=dict(parameter_sha256='p',buffer_sha256='b',parameter_count=32763876352,parameter_bytes=131055505408,parameter_devices=['cpu'],buffer_devices=['cpu'],requires_grad_parameters=0,grad_parameters=0,training_modules=0)
 for step in range(1,6):
  raw=dict(status='RELEASED',original_method_sha256=METHOD,all_threads_joined=True,initial_cpu=state,released_cpu=state,allocators_released=[dict(uuid=u,allocated_bytes=0,reserved_bytes=0) for u in binding['leased_uuids']])
  release={k:binding[k] for k in ('run_id','code_commit','config_sha256','data_identity','teacher_receipt_sha256','source_sha256')};release.update(outer_step=step,status='GLOBAL_ALL8_RELEASED',nonce=str(step),original_scorer_receipt=raw)
  path=root/'ipc'/f'global-release-step{step}.json';path.write_text(json.dumps(release))
  ack={k:binding[k] for k in ('run_id','code_commit','config_sha256','data_identity','teacher_receipt_sha256','source_sha256')};ack.update(outer_step=step,status='ALL8_ACKED_ACTOR_RELEASE_BARRIER',ranks=list(range(8)),nonce=str(step),release_receipt=dict(name=path.name,sha256=sha(path)),ack_epoch=step*100)
  (root/'ipc'/f'global-ack-step{step}.json').write_text(json.dumps(ack))
 for rank in range(8):
  with (root/'audit'/f'{rank}.jsonl').open('w') as f:
   for step in range(1,6):
    row=dict(kind='actor_outer',outer_step=step,rank=rank,pid=100+rank,rows=320,epochs=2,optimizer_steps=20,grad_finite=20,grad_nonzero=20,skipped_empty_minibatches=0,fixed_unchanged=True,shard_layout_unchanged=True,grad_norms=[1.]*20,micro_stats=[{k:dict(count=1,finite=1) for k in ('current_log_probs','ratios','policy_loss')}]*80,tensors={k:dict(finite=10,count=10) for k in ('old_log_probs','advantages','policy_weights')},shard_delta=dict(nonzero=int(step>1)),run_id='cpu_fixture',config_sha256='c',data_identity='d',commit=D23)
    f.write(json.dumps(row)+'\n')
    with (root/'workers'/f'{rank}.jsonl').open('a') as e:
     for event,epoch in [('ALL8_RELEASED_ACKED_BEFORE_ACTOR_UPDATE',step*100+1),('ORIGINAL_ACTOR_UPDATE_RETURNED',step*100+2)]:e.write(json.dumps(dict(event=event,epoch=epoch,outer_step=step,pid=100+rank))+'\n')
 with (root/'audit/driver.jsonl').open('w') as f:
  for step in range(1,6):f.write(json.dumps(dict(kind='driver_outer',outer_step=step,teacher_targets_none=True,teacher_scored_tokens_delta=2560,tensors={k:dict(nonzero=10,finite=10,count=10) for k in ('teacher_log_probs','old_log_probs','token_level_rewards','returns','advantages','policy_weights')},observation_tokens=1,run_id='cpu_fixture',config_sha256='c',data_identity='d',commit=D23))+'\n')
 checkpoint=root/'checkpoints/actor/global_step_5';checkpoint.mkdir(parents=True);checkpoint_fixture(checkpoint);binding['student_model_directory']=str(checkpoint)
 statepath=root/'checkpoints/algorithm/global_step_5';statepath.mkdir(parents=True);(statepath/'state.json').write_text(json.dumps(dict(method='topd',global_step=5,resume_supported=False,teacher_path='/cpu/fixture')))
 with (root/'validation64.jsonl').open('w') as f:
  for step in (0,5):f.write(json.dumps(dict(step=step,metrics={'val/qa_count/all':64,'val/em/all':0.},question_identity_sha256='same64',run_id='cpu_fixture',config_sha256='c',data_identity='d',commit=D23))+'\n')
 return binding

def test_detached_fixture_accepts_updates_not_owner_counter(tmp_path):
 binding=acceptance_fixture(tmp_path);report=actual_acceptance(tmp_path,5,binding)
 assert report['completed_actual_outer_steps']==list(range(1,6)) and report['scope']=='ACTUAL_FIRST5_UPDATES_UNREVIEWED_CONTINUING_FORMAL500'
 assert len(report['raw_rank_files'])==8
 assert all(Path(p).exists() for p in report['raw_rank_files'])
@pytest.mark.parametrize('mutation',['zero_update','unchanged_shard','wrong_run','missing_rank','wrong_teacher_math','release_alloc_nonzero','nonfinite_micro'])
def test_detached_actual_counterexamples(tmp_path,mutation):
 binding=acceptance_fixture(tmp_path)
 if mutation in ('wrong_teacher_math','release_alloc_nonzero'):
  path=tmp_path/'ipc/global-release-step3.json';r=json.loads(path.read_text())
  if mutation=='wrong_teacher_math':r['original_scorer_receipt']['original_method_sha256']='foreign'
  else:r['original_scorer_receipt']['allocators_released'][0]['allocated_bytes']=32*1024**2
  path.write_text(json.dumps(r));ackpath=tmp_path/'ipc/global-ack-step3.json';a=json.loads(ackpath.read_text());a['release_receipt']['sha256']=sha(path);ackpath.write_text(json.dumps(a))
 else:
  path=tmp_path/'audit/7.jsonl';rows=read_complete_jsonl(path)
  if mutation=='zero_update':rows[2]['optimizer_steps']=0
  elif mutation=='unchanged_shard':
   for r in rows:r['shard_delta']['nonzero']=0
  elif mutation=='nonfinite_micro':rows[2]['micro_stats'][4]['policy_loss']['finite']=0
  elif mutation=='wrong_run':rows[3]['run_id']='other'
  else:rows.pop()
  path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
 with pytest.raises(RuntimeError):actual_acceptance(tmp_path,5,binding)

def test_ack_after_optimizer_counterexample(tmp_path):
 binding=acceptance_fixture(tmp_path);path=tmp_path/'ipc/global-ack-step2.json';ack=json.loads(path.read_text());ack['ack_epoch']=999999;path.write_text(json.dumps(ack))
 with pytest.raises(RuntimeError,match='ordering'):actual_acceptance(tmp_path,5,binding)


def test_actual_next_rollout_rebuild_interval():
 records=[dict(step=1,scoring_completed_epoch=100),dict(step=2,scoring_started_epoch=200)]
 workers={i:[dict(event='ACTUAL_REBUILT_KV_BEFORE_VLLM_GENERATE',epoch=150,kv=[dict(cache_engine_is_none=False,gpu_cache_is_none=False)])] for i in range(8)}
 assert len(next_rollout_rebuilt(records,workers))==8

def test_first_rollout_many_turns_cannot_prove_next_rebuild():
 records=[dict(step=1,scoring_completed_epoch=100),dict(step=2,scoring_started_epoch=200)]
 workers={i:[dict(event='ACTUAL_REBUILT_KV_BEFORE_VLLM_GENERATE',epoch=50+n,kv=[dict(cache_engine_is_none=False,gpu_cache_is_none=False)]) for n in range(4)] for i in range(8)}
 with pytest.raises(RuntimeError,match='between scoring phases'):next_rollout_rebuilt(records,workers)


def source_base():return ROOT/'repo-shared32b' if os.environ.get('AGENT_ROOT')==str(ROOT) else Path(__file__).resolve().parents[1]/'topd-shared32b-integration-candidate'

def original_loss_mask_and_select():
 import ast,copy
 source=source_base();tree=ast.parse((source/'verl/trainer/ppo/ray_trainer.py').read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='RayPPOTrainer');method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_create_loss_mask');ns={};exec(compile(ast.Module(body=[copy.deepcopy(method)],type_ignores=[]),'original_ray_loss_mask','exec'),ns)
 actor=ast.parse((source/'verl/workers/actor/dp_actor.py').read_text());cls=next(n for n in actor.body if isinstance(n,ast.ClassDef) and n.name=='DataParallelPPOActor');update=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='update_policy')
 start=next(i for i,n in enumerate(update.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='select_keys' for t in n.targets))
 stop=next(i for i,n in enumerate(update.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='batch' for t in n.targets))
 boundary=copy.deepcopy(method);boundary.name='select_original_actor_boundary';boundary.args=ast.parse('def f(self,data): pass').body[0].args;boundary.body=copy.deepcopy(update.body[start:stop+1])+[ast.Return(value=ast.Name(id='batch',ctx=ast.Load()))]
 exec(compile(ast.fix_missing_locations(ast.Module(body=[boundary],type_ignores=[])),'original_DP_actor_select','exec'),ns)
 return ns['_create_loss_mask'],ns['select_original_actor_boundary']

def actual_data_proto(monkeypatch):
 import torch
 monkeypatch.syspath_prepend(str(source_base()))
 from verl import DataProto
 batch=DataProto.from_dict(tensors={k:torch.ones((2,4)) for k in ('responses','input_ids','attention_mask','position_ids','old_log_probs','advantages','policy_weights','info_mask')})
 batch.batch['info_mask'][0,2]=0
 return batch

def test_actual_DataProto_actor_select_rejects_missing_loss_mask(monkeypatch):
 _,select=original_loss_mask_and_select();batch=actual_data_proto(monkeypatch)
 actor=types.SimpleNamespace(config=types.SimpleNamespace(state_masking=True,use_kl_loss=False,get=lambda k: 'topd'))
 with pytest.raises(KeyError):select(actor,batch)

def test_original_loss_mask_then_actual_actor_select(monkeypatch):
 import torch
 create,select=original_loss_mask_and_select();batch=actual_data_proto(monkeypatch);batch,metrics=create(None,batch,{})
 actor=types.SimpleNamespace(config=types.SimpleNamespace(state_masking=True,use_kl_loss=False,get=lambda k:'topd'))
 selected=select(actor,batch)
 assert torch.equal(selected['loss_mask'],batch.batch['info_mask']) and selected['loss_mask'][0,2]==0
 assert metrics['state_tokens/total']==7

def checkpoint_fixture(directory):
 import torch
 from safetensors.torch import save_file
 from hf_inventory import shapes
 cfg=dict(model_type='qwen2',vocab_size=3,hidden_size=2,intermediate_size=4,num_attention_heads=2,num_key_value_heads=1,num_hidden_layers=1,tie_word_embeddings=True)
 (directory/'config.json').write_text(json.dumps(cfg));save_file({k:torch.zeros(v) for k,v in shapes(cfg).items()},str(directory/'model.safetensors'))
 for name in ('tokenizer.json','tokenizer_config.json','special_tokens_map.json'):(directory/name).write_text('{}')

def test_config_only_checkpoint_cannot_publish_evidence(tmp_path):
 binding=acceptance_fixture(tmp_path);(tmp_path/'checkpoints/actor/global_step_5/model.safetensors').unlink()
 with pytest.raises(RuntimeError,match='Incomplete HF'):actual_acceptance(tmp_path,5,binding)

def test_empty_actor_tensors_cannot_publish_evidence(tmp_path):
 binding=acceptance_fixture(tmp_path);p=tmp_path/'audit/3.jsonl';rows=read_complete_jsonl(p);rows[2]['tensors']={};p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
 with pytest.raises(RuntimeError,match='actor audit tensors'):actual_acceptance(tmp_path,5,binding)

def test_empty_driver_tensors_cannot_publish_evidence(tmp_path):
 binding=acceptance_fixture(tmp_path);p=tmp_path/'audit/driver.jsonl';rows=read_complete_jsonl(p);rows[2]['tensors']={'advantages':dict(nonzero=10,count=10,finite=10)};p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
 with pytest.raises(RuntimeError,match='driver audit tensors'):actual_acceptance(tmp_path,5,binding)


def test_terminal_join_cannot_drop_late_strict_monitor_error():
 import threading
 from lifecycle import join_monitor
 stop=threading.Event();errors=[]
 def actual_cpu_thread():
  stop.wait();errors.append('foreign SM15 boundary failure detected during closure')
 monitor=threading.Thread(target=actual_cpu_thread);monitor.start()
 with pytest.raises(RuntimeError,match='during terminal closure'):join_monitor(stop,monitor,errors,1)
 assert not monitor.is_alive() and errors


def test_raw_resource_JSONL_reference_verified_without_fake_JSON_parse(tmp_path):
 p=tmp_path/'actual.jsonl';p.write_text('{"sample":1}\n{"sample":2}\n')
 assert file_ref(dict(path=str(p),sha256=sha(p)))==p

def test_raw_resource_log_byte_drift_rejected(tmp_path):
 p=tmp_path/'native.log';p.write_text('actual native CPU log');ref=dict(path=str(p),sha256=sha(p));p.write_text('modified after review')
 with pytest.raises(RuntimeError):file_ref(ref)


def test_guard_only_or_mixed_runtime_native_proof_rejected():
 review=dict(status='PASS',guard_source_sha256='newguard',runtime_source_sha256='newwindow3316',actual_native_dual_owner_coverage_passed=True)
 assert telemetry_pair(review,dict(sha256='newguard'),dict(sha256='newwindow3316'))==review
 with pytest.raises(RuntimeError):telemetry_pair(review,dict(sha256='newguard'),dict(sha256='oldc7'))
 with pytest.raises(RuntimeError):telemetry_pair(review,dict(sha256='failedR2'),dict(sha256='newwindow3316'))


def test_accept5_single_validation_no500_extra_stage_calls():
 assert [s for s in range(1,501) if explicit_accept_validation(s,5,50)]==[5]
 assert not explicit_accept_validation(50,5,50)

def test_initial_and_accept5_require_same_actual64_questions(tmp_path):
 binding=acceptance_fixture(tmp_path);assert validation_effect(tmp_path,binding)['status'].startswith('MEASURED')
 p=tmp_path/'validation64.jsonl';rows=read_complete_jsonl(p);rows[1]['question_identity_sha256']='different64';p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
 with pytest.raises(RuntimeError,match='questions differ'):validation_effect(tmp_path,binding)


def test_terminal_final_name_invisible_until_full_fsync(tmp_path,monkeypatch):
 import threading,lifecycle
 entered=threading.Event();resume=threading.Event();original=json.dump
 def paused(*args,**kwargs):entered.set();assert resume.wait(2);return original(*args,**kwargs)
 monkeypatch.setattr(lifecycle.json,'dump',paused)
 path=tmp_path/'terminal-held.json';errors=[]
 def writer():
  try:publish(path,{'full':'receipt'})
  except BaseException as exc:errors.append(exc)
 t=threading.Thread(target=writer);t.start();assert entered.wait(2)
 assert not path.exists()
 resume.set();t.join(2);assert not t.is_alive() and not errors and json.loads(path.read_text())=={'full':'receipt'}

def test_existing_terminal_permission_never_overwritten(tmp_path):
 p=tmp_path/'authorized';publish(p,{'original':'permission'});before=p.read_bytes()
 with pytest.raises(FileExistsError):publish(p,{'replacement':'forbidden'})
 assert p.read_bytes()==before and not list(tmp_path.glob('*.tmp'))

def test_real_held_frame_waits_for_bound_parent_authorization(tmp_path,monkeypatch):
 import threading,lifecycle
 monkeypatch.setenv('TOPD32B_RUN',str(tmp_path));monkeypatch.setenv('SLURM_JOB_ID','78')
 trainer=types.SimpleNamespace(actual_worker_refs=object());results=[];errors=[]
 def worker():
  try:results.append(lifecycle.hold_algorithm_frame(trainer,'profile'))
  except BaseException as exc:errors.append(exc)
 t=threading.Thread(target=worker);t.start()
 from time import monotonic,sleep
 until=monotonic()+2
 while not (tmp_path/'driver-algorithm-returned.json').exists() and monotonic()<until:sleep(.01)
 assert t.is_alive() and not results
 held=tmp_path/'driver-algorithm-returned.json';assert json.loads(held.read_text())['status'].startswith('ORIGINAL_FIT_COMPLETED_TRAINER_HELD')
 publish(tmp_path/'authorized-terminal-exit.json',dict(job='78',parent_pid=os.getppid(),actual_driver_return_sha256=sha(held),phase='NO_MORE_GPU_WORK_OWNED_EXIT'))
 t.join(2);assert not t.is_alive() and not errors and results==[trainer]

def test_wrong_held_receipt_authorization_rejected(tmp_path,monkeypatch):
 import lifecycle
 monkeypatch.setenv('TOPD32B_RUN',str(tmp_path));monkeypatch.setenv('SLURM_JOB_ID','78')
 publish(tmp_path/'authorized-terminal-exit.json',dict(job='78',parent_pid=os.getppid(),actual_driver_return_sha256='not-original',phase='NO_MORE_GPU_WORK_OWNED_EXIT'))
 with pytest.raises(RuntimeError,match='terminal permission mismatch'):lifecycle.hold_algorithm_frame(object(),'formal500')

def test_terminal_hold_is_inside_real_fit_before_function_return():
 import ast
 tree=ast.parse((Path(__file__).parent/'shared_supervision_hooks.py').read_text())
 fit=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='fit')
 calls=[n for n in ast.walk(fit) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
 hold=next(n for n in calls if n.func.id=='hold_algorithm_frame')
 original=next(n for n in calls if n.func.id=='original_fit')
 assert original.lineno<hold.lineno<fit.body[-1].lineno
 train=ast.parse((Path(__file__).parent/'train_entry.py').read_text())
 assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='hold_algorithm_frame' for n in ast.walk(train))


def test_replace_only_explicit_mutable_progress(tmp_path):
 p=tmp_path/'progress';publish(p,{'phase':1},replace=True);publish(p,{'phase':2},replace=True)
 assert json.loads(p.read_text())=={'phase':2}


def startup_fixture(tmp_path,monkeypatch,all8=True):
 import importlib.util,sys,dependencies,retriever_lifecycle as r
 monkeypatch.setattr(dependencies,'own',lambda p:Path(p).resolve())
 monkeypatch.setitem(sys.modules,'eval64',types.SimpleNamespace(require=require))
 root=Path(__file__).parents[1]/'qwen25-32b-window-safety-candidate-v4/runtime_safety.py'
 spec=importlib.util.spec_from_file_location('test_current_window_runtime',root);safety=importlib.util.module_from_spec(spec);spec.loader.exec_module(safety)
 uuids=[str(i) for i in range(8)];pid=os.getpid();foreign_id={'pid':9,'starttime_ticks':1,'cgroup':['0::/foreign']};owner_id={'pid':pid,'starttime_ticks':2,'cgroup':['0::/job8']}
 baseline=dict(gpus=[dict(index=str(i),uuid=str(i),sm=2.,used_mib=256.,total_mib=80000.) for i in range(8)],applications=[dict(uuid=u,pid=9,used_mib=128.) for u in uuids],identities={'9':foreign_id})
 current=copy.deepcopy(baseline);current['applications'] += [dict(uuid=u,pid=pid,used_mib=128.) for u in (uuids if all8 else uuids[:3])];current['identities'][str(pid)]=owner_id
 safety.nvml_snapshot=lambda:copy.deepcopy(current);safety.pid_identity=lambda p:owner_id if p==pid else foreign_id
 safety.collect_process_windows=lambda snap,foreign:({},int(__import__('time').time()*1e6))
 monkeypatch.setenv('CUDA_VISIBLE_DEVICES',','.join(uuids))
 calls=[];guard=types.SimpleNamespace(lease_reader=lambda:set(uuids),active_gate=lambda runtime,pre_model:calls.append(runtime) or {'full_gate':'actual_call_contract'})
 stage=r.RetrieverLifecycle(safety,guard,baseline,uuids,'/job8',tmp_path);stage.attach(pid)
 # CPU fixture supplies synthetic UID sidecar explicitly; production uses real /proc.
 original_sample=stage.partial.sample
 monkeypatch.setattr(stage.partial,'sample',lambda *a,**k:original_sample(*a,**k,uid_read=lambda pid:os.getuid()))
 return stage,calls,current

def test_real_reused_partial_source_then_actual_full_gate_switch(tmp_path,monkeypatch):
 stage,calls,_=startup_fixture(tmp_path,monkeypatch);record=stage.sample()
 assert record['scope']=='STARTING_RETRIEVER_PARTIAL_RESIDENCY_NOT_ALL8_READY' and not calls
 assert stage.full(record)=={'full_gate':'actual_call_contract'} and not stage.starting and calls==[stage.runtime]
 stage.sample();assert len(calls)==2
 rows=read_complete_jsonl(tmp_path/'retriever-startup-raw.jsonl');assert any(r['stage']=='ALL8_RESIDENT_COMPLETE_WINDOW_V4_ACCEPTED' for r in rows)

def test_partial3_cards_cannot_claim_all8_ready(tmp_path,monkeypatch):
 stage,calls,_=startup_fixture(tmp_path,monkeypatch,all8=False);record=stage.sample()
 with pytest.raises(RuntimeError,match='all8'):stage.full(record)
 assert stage.starting and not calls

def test_full_gate_failure_preserves_partial_mode(tmp_path,monkeypatch):
 stage,calls,_=startup_fixture(tmp_path,monkeypatch);record=stage.sample()
 def reject(*a,**k):raise RuntimeError('protected foreign actual gate error')
 stage.guard.active_gate=reject
 with pytest.raises(RuntimeError,match='protected foreign'):stage.full(record)
 assert stage.starting

def test_reused_partial_gate_actual_foreign_memory_breach_rejected(tmp_path,monkeypatch):
 stage,calls,current=startup_fixture(tmp_path,monkeypatch);current['applications'][0]['used_mib']=16000.
 with pytest.raises(RuntimeError,match='Foreign GPU memory'):stage.sample()
 assert not calls and stage.starting
 rows=read_complete_jsonl(tmp_path/'retriever-startup-raw.jsonl');assert rows[-1]['stage']=='FAILED_STARTUP_NOT_ACCEPTED'

def test_docs_health_actual_HTML_without_JSON_decoder():
 from retriever_lifecycle import docs_health
 class Reply:
  status_code=200;headers={'content-type':'text/html; charset=utf-8'};content=b'<html>Swagger UI</html>'
  def json(self):raise AssertionError('HTML readiness must not decode JSON')
 class Session:
  def get(self,url,timeout):assert url.endswith(':18085/docs') and timeout==5;return Reply()
 assert docs_health(Session())['status']==200

def test_wrong_docs404_not_health_success():
 from retriever_lifecycle import docs_health
 assert docs_health(types.SimpleNamespace(get=lambda *a,**k:types.SimpleNamespace(status_code=404))) is None

def test_source_monitor_before_spawn_and_attach_under_same_lock():
 import ast
 tree=ast.parse((Path(__file__).parent/'supervise.py').read_text())
 spawn=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='retriever' for t in n.targets))
 start=next(n for n in ast.walk(tree) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and isinstance(n.value.func.value,ast.Name) and n.value.func.value.id=='monitor' and n.value.func.attr=='start')
 lock=next(n for n in ast.walk(tree) if isinstance(n,ast.With) and any(isinstance(it.context_expr,ast.Attribute) and it.context_expr.attr=='lock' for it in n.items))
 assert start.lineno<lock.lineno<spawn.lineno and any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='attach' for n in ast.walk(lock))

def test_wait_ready_stops_immediately_on_actual_monitor_failure(tmp_path,monkeypatch):
 from retriever_lifecycle import wait_ready
 stage,calls,_=startup_fixture(tmp_path,monkeypatch)
 with pytest.raises(RuntimeError,match='startup guardian failed'):wait_ready(stage,types.SimpleNamespace(poll=lambda:None),None,['foreign failure'],timeout=.1)

def test_fresh_partial_source_pin_rejects_changed_bytes(tmp_path,monkeypatch):
 import retriever_lifecycle as r,dependencies
 monkeypatch.setattr(r,'STARTUP_SHA','0'*64)
 with pytest.raises(RuntimeError,match='source drift'):r.RetrieverLifecycle(None,None,{},[],None,tmp_path)


def test_ready_file_cannot_hide_existing_monitor_error(tmp_path):
 p=tmp_path/'ready';publish(p,{'full':'json'})
 with pytest.raises(RuntimeError,match='monitor failed'):wait(p,[],['actual protected error'],.1)

def test_actual_wait_ready_docs_all8_idle_switch_closes_no_proxy_session(tmp_path,monkeypatch):
 import requests,retriever_lifecycle as r
 stage,calls,_=startup_fixture(tmp_path,monkeypatch);sessions=[]
 class Session:
  trust_env=True;closed=False
  def __init__(self):sessions.append(self)
  def get(self,url,timeout):
   assert self.trust_env is False and url=='http://127.0.0.1:18085/docs'
   return types.SimpleNamespace(status_code=200,headers={'content-type':'text/html'},content=b'<html>Swagger UI</html>')
  def close(self):self.closed=True
 monkeypatch.setattr(requests,'Session',Session)
 assert r.wait_ready(stage,types.SimpleNamespace(poll=lambda:None),types.SimpleNamespace(inherited_lease_fds=lambda:tuple(range(8))),[])=={'full_gate':'actual_call_contract'}
 assert not stage.starting and len(calls)==1 and sessions[0].closed
