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
    row=dict(kind='actor_outer',outer_step=step,rank=rank,pid=100+rank,rows=320,epochs=2,optimizer_steps=20,grad_finite=20,grad_nonzero=20,skipped_empty_minibatches=0,fixed_unchanged=True,shard_layout_unchanged=True,grad_norms=[1.]*20,micro_stats=[{k:dict(count=1,finite=1) for k in ('current_log_probs','ratios','policy_loss')}]*80,tensors={'advantages':dict(finite=10,count=10)},shard_delta=dict(nonzero=int(step>1)),run_id='cpu_fixture',config_sha256='c',data_identity='d',commit=D23)
    f.write(json.dumps(row)+'\n')
    with (root/'workers'/f'{rank}.jsonl').open('a') as e:
     for event,epoch in [('ALL8_RELEASED_ACKED_BEFORE_ACTOR_UPDATE',step*100+1),('ORIGINAL_ACTOR_UPDATE_RETURNED',step*100+2)]:e.write(json.dumps(dict(event=event,epoch=epoch,outer_step=step,pid=100+rank))+'\n')
 with (root/'audit/driver.jsonl').open('w') as f:
  for step in range(1,6):f.write(json.dumps(dict(kind='driver_outer',outer_step=step,teacher_targets_none=True,teacher_scored_tokens_delta=2560,tensors={'advantages':dict(nonzero=10,finite=10,count=10)},observation_tokens=1,run_id='cpu_fixture',config_sha256='c',data_identity='d',commit=D23))+'\n')
 checkpoint=root/'checkpoints/actor/global_step_5';checkpoint.mkdir(parents=True);(checkpoint/'config.json').write_text('{}')
 statepath=root/'checkpoints/algorithm/global_step_5';statepath.mkdir(parents=True);(statepath/'state.json').write_text(json.dumps(dict(method='topd',global_step=5,resume_supported=False,teacher_path='/cpu/fixture')))
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
