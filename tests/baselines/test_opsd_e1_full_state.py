"""Real CPU model/Adam/RNG same-object roundtrip; FSDP/CUDA routes mocked."""
import ast
import copy
import importlib.util
import random
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from search_r1.baselines.full_state import (StatefulSampler, state_digest, atomic_save,
    rng_state, restore_rng, resolve_warmup_steps, teacher_state, restore_teacher_state)

ROOT=Path(__file__).resolve().parents[2]

def test_explicit57_not_ratio_sentinel():
 assert resolve_warmup_steps({'lr_warmup_steps':57,'lr_warmup_steps_ratio':.2,'total_training_steps':301})==57
 assert resolve_warmup_steps({'lr_warmup_steps_ratio':.2,'total_training_steps':300})==60
 for bad in [True,57.,-1,'57']:
  with pytest.raises(ValueError):resolve_warmup_steps({'lr_warmup_steps':bad})

def test_sampler_consumption_and_same_live_iterator():
 sampler=StatefulSampler(100,seed=42);iterator=iter(sampler)
 first=[next(iterator)for _ in range(10)]
 with pytest.raises(ValueError):sampler.state_dict()
 sampler.commit_consumed(10);state=copy.deepcopy(sampler.state_dict());expected=[next(iterator)for _ in range(8)]
 sampler.load_state_dict(state);assert [next(iterator)for _ in range(8)]==expected
 assert len(set(first+expected))==18
 with pytest.raises(ValueError):sampler.commit_consumed(99)

def test_randomsampler_none_seed_order():
 torch.manual_seed(42);expected=list(torch.utils.data.RandomSampler(range(32)))
 torch.manual_seed(42);sampler=StatefulSampler(32);assert list(sampler)==expected

def test_rng_all_cpu_roundtrip():
 saved=rng_state();expected=(random.random(),np.random.rand(),torch.rand(4))
 restore_rng(saved);assert expected[0]==random.random() and expected[1]==np.random.rand() and torch.equal(expected[2],torch.rand(4))

def test_immutable_atomic_save_and_foreign_temp(tmp_path):
 path=tmp_path/'state.pt';atomic_save({'x':torch.tensor([1])},path);before=path.read_bytes()
 with pytest.raises(FileExistsError):atomic_save({'x':torch.tensor([2])},path)
 assert path.read_bytes()==before
 temp=tmp_path/'foreign.pt.tmp';temp.write_bytes(b'foreign')
 with pytest.raises(FileExistsError):atomic_save({},tmp_path/'foreign.pt')
 assert temp.read_bytes()==b'foreign'

class Teacher:
 def __init__(self):
  self.model=torch.nn.Linear(2,2).float().eval().requires_grad_(False)
  self.model.config=SimpleNamespace(to_dict=lambda:{'fixed':'CPU_TOY_NOT_OFFICIAL'})
  self.device=self.execution_device=torch.device('cpu');self.parameter_count=sum(p.numel()for p in self.model.parameters());self.model_bytes=self.parameter_count*4
  for name in ('cpu_seconds','wall_seconds','transfer_wall_seconds','cuda_update_entries','cuda_seconds','scored_tokens','load_cpu_seconds','load_wall_seconds'):setattr(self,name,0)
 def _flush_cuda_events(self):pass
 def _validate_execution(self,device):
  assert all(p.device.type=='cpu' and p.dtype==torch.float32 for p in self.model.parameters())

@pytest.fixture
def worker(tmp_path,monkeypatch):
 from torch.distributed.fsdp import FullyShardedDataParallel as FSDP,StateDictType
 names={'_training_state_payload','_load_training_state_payload','save_training_state','load_training_state'}
 source=ast.parse((ROOT/'verl/workers/fsdp_workers.py').read_text());cls=next(n for n in source.body if isinstance(n,ast.ClassDef)and n.name=='ActorRolloutRefWorker');methods=[n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name in names]
 for n in methods:n.decorator_list=[]
 namespace={'__file__':str(ROOT/'verl/workers/fsdp_workers.py'),'torch':torch,'os':__import__('os'),'load_fsdp_param_and_grad':lambda *a,**k:None,'offload_fsdp_param_and_grad':lambda *a,**k:None,'offload_fsdp_optimizer':lambda *a,**k:None}
 exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(name='ActualWorkerMethods',bases=[],keywords=[],body=methods,decorator_list=[])],type_ignores=[])),str(ROOT/'verl/workers/fsdp_workers.py'),'exec'),namespace)
 def context(*args):
  assert args[1]==StateDictType.SHARDED_STATE_DICT
  assert args[2].offload_to_cpu is True
  return nullcontext()
 monkeypatch.setattr(FSDP,'state_dict_type',context)
 monkeypatch.setattr(torch.cuda,'current_device',lambda:0);monkeypatch.setattr(torch.cuda,'get_rng_state',lambda device:torch.get_rng_state().clone());monkeypatch.setattr(torch.cuda,'set_rng_state',lambda state,device:None);monkeypatch.setattr(torch.distributed,'barrier',lambda:None)
 asset=tmp_path/'assets';asset.mkdir();(asset/'fixed-identity').write_bytes(b'toy-fixed-teacher-assets')
 model=torch.nn.Linear(2,2);optim=torch.optim.AdamW(model.parameters(),lr=.001);scheduler=torch.optim.lr_scheduler.LambdaLR(optim,lambda n:min(1,n/57))
 result=namespace['ActualWorkerMethods']();result.actor_module_fsdp=model;result.actor_optimizer=optim;result.actor_lr_scheduler=scheduler
 result.actor=SimpleNamespace(teacher=Teacher());result.rank=0;result.world_size=1;result._is_actor=True;result._is_offload_param=False;result._is_offload_optimizer=False
 class Config(SimpleNamespace):
  def get(self,name,default=None):return getattr(self,name,default)
 from omegaconf import OmegaConf
 result.config=OmegaConf.create({'actor':{'baseline_method':'opsd','distillation':{'teacher_path':str(asset)}}})
 return result,tmp_path

def update(w):
 # Real Adam and real randomness; four minis stand for one synthetic outer.
 for _ in range(4):
  x=torch.rand(3,2)+random.random()+float(np.random.rand());loss=w.actor_module_fsdp(x).square().mean();loss.backward();w.actor_optimizer.step();w.actor_optimizer.zero_grad()
 w.actor_lr_scheduler.step();w.actor.teacher.scored_tokens+=12

def test_actual_worker_methods_full_adam_teacher_next_update_oracle(worker):
 w,path=worker
 for _ in range(5):update(w)
 identities=(id(w.actor_module_fsdp),id(w.actor_optimizer),id(w.actor_lr_scheduler),id(w.actor.teacher))
 report=w.save_training_state(str(path/'rankstate'),5,roundtrip=True)
 assert report['status']=='EXACT_FULL_STATE_RELOAD_OK' and report['optimizer_state_entries']==2 and report['version']==3
 update(w);oracle=state_digest(w._training_state_payload(6))
 restored=w.load_training_state(str(path/'rankstate'),5)
 assert restored['status']=='EXACT_FULL_STATE_RELOAD_OK' and w.actor.teacher.scored_tokens==60
 update(w);assert state_digest(w._training_state_payload(6))==oracle
 assert identities==(id(w.actor_module_fsdp),id(w.actor_optimizer),id(w.actor_lr_scheduler),id(w.actor.teacher))

@pytest.mark.parametrize('fault',['pid','rank','step','object','teacher','content'])
def test_coherent_rehashed_wrong_identity_refused(worker,fault):
 w,path=worker;update(w);saved=w._training_state_payload(5)
 if fault in ('pid','rank','step'):saved[fault]+=1
 elif fault=='object':saved['object_ids']['optimizer']+=1
 elif fault=='teacher':saved['teacher']['asset_files']['fixed-identity']='0'*64
 else:saved['model']['weight'][0,0]+=1
 if fault=='content':
  # Content can be legitimate saved state, but a digest not matching it cannot pass.
  expected='0'*64
 else:expected=state_digest(saved)
 with pytest.raises(ValueError):w._load_training_state_payload(saved,5,expected)

def test_frozen_teacher_mutation_refused(worker):
 w,path=worker;saved=teacher_state(w.actor.teacher,w.config.actor.distillation.teacher_path)
 with torch.no_grad():w.actor.teacher.model.weight.add_(1)
 with pytest.raises(ValueError):restore_teacher_state(w.actor.teacher,w.config.actor.distillation.teacher_path,saved)

@pytest.mark.parametrize('index',[0,3,7])
@pytest.mark.parametrize('method',['update_actor','compute_log_prob','generate_sequences','compute_ref_log_prob'])
def test_actual_worker_entry_explicit_logical_device(index,method,monkeypatch):
 source=ast.parse((ROOT/'verl/workers/fsdp_workers.py').read_text());cls=next(n for n in source.body if isinstance(n,ast.ClassDef)and n.name=='ActorRolloutRefWorker');function=next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name==method);function.decorator_list=[]
 namespace={'__file__':str(ROOT/'verl/workers/fsdp_workers.py'),'torch':torch,'DataProto':object};exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),'actual-worker-entry','exec'),namespace)
 calls=[]
 class Stopped(Exception):pass
 class CPUData:
  def to(self,device):calls.append(device);raise Stopped()
 monkeypatch.setattr(torch.cuda,'current_device',lambda:index)
 with pytest.raises(Stopped):namespace[method](SimpleNamespace(_is_ref=True),CPUData())
 assert calls==[torch.device('cuda',index)]

def test_native_fixture_compiles_without_running_cuda():
 path=ROOT/'reproduction/baselines/e1-300/native_opsd_state_fixture.py'
 compile(path.read_text(),str(path),'exec')
 text=path.read_text();assert 'ShardedStateDictConfig' in text and 'component_clip=None' in text
 assert 'load_training_state(worker' in text and 'CUDA RNG' in text

def test_real_world_one_cpu_dtensor_disk_and_layout(tmp_path):
 import torch.distributed as dist
 from torch.distributed.device_mesh import DeviceMesh
 from torch.distributed.tensor import DTensor,Shard,Replicate
 dist.init_process_group('gloo',init_method='file://'+str(tmp_path/'group'),rank=0,world_size=1)
 try:
  mesh=DeviceMesh('cpu',[0]);local=torch.arange(8,dtype=torch.float32)
  value=DTensor.from_local(local,device_mesh=mesh,placements=[Shard(0)],run_check=False)
  path=tmp_path/'real-dtensor.pt';atomic_save({'tensor':value},path)
  loaded=torch.load(path,weights_only=False,map_location='cpu')['tensor']
  assert state_digest(value)==state_digest(loaded)
  changed=DTensor.from_local(local+1,device_mesh=mesh,placements=[Shard(0)],run_check=False)
  replicated=DTensor.from_local(local,device_mesh=mesh,placements=[Replicate()],run_check=False)
  assert state_digest(value)!=state_digest(changed) and state_digest(value)!=state_digest(replicated)
 finally:dist.destroy_process_group()

def test_actual_teacher_buffers_and_cache_roundtrip(worker):
 w,path=worker;teacher=w.actor.teacher
 teacher.model.register_buffer('dynamic',torch.tensor([1.,2.]),persistent=False)
 teacher.model.max_seq_len_cached=2
 saved=teacher_state(teacher,w.config.actor.distillation.teacher_path)
 teacher.model.dynamic=torch.tensor([7.,8.,9.]);teacher.model.max_seq_len_cached=3;teacher.scored_tokens=999
 restore_teacher_state(teacher,w.config.actor.distillation.teacher_path,saved)
 assert torch.equal(teacher.model.dynamic,torch.tensor([1.,2.])) and teacher.model.max_seq_len_cached==2 and teacher.scored_tokens==0

@pytest.mark.parametrize('state',['floatcursor','booleanepoch','nopermutation'])
def test_sampler_invalid_cursor_metadata_reject(state):
 sampler=StatefulSampler(32,seed=42);iterator=iter(sampler);next(iterator);sampler.commit_consumed(1);saved=sampler.state_dict()
 if state=='floatcursor':saved['cursor']=saved['consumed_cursor']=1.
 elif state=='booleanepoch':saved['epoch']=True
 else:saved['order']=None
 with pytest.raises(ValueError):sampler.load_state_dict(saved)

def test_teacher_optimizer_overlap_refused(worker):
 w,path=worker
 bad=torch.optim.Adam(w.actor.teacher.model.parameters())
 with pytest.raises(ValueError):teacher_state(w.actor.teacher,w.config.actor.distillation.teacher_path,bad)


def test_explicit_new_objects_recovery_next_update_oracle(worker):
 import copy,json,os
 from search_r1.baselines.full_state import recovery_core,file_digest
 w,path=worker
 for _ in range(5):update(w)
 report=w.save_training_state(str(path/'recovery'),5)
 saved=torch.load(path/'recovery/rank_0.pt',weights_only=False)
 update(w);oracle=state_digest(recovery_core(w._training_state_payload(6)))
 # New actual model/Adam/scheduler/teacher objects, representing a new worker.
 model=torch.nn.Linear(2,2);optim=torch.optim.AdamW(model.parameters(),lr=.001)
 w.actor_module_fsdp=model;w.actor_optimizer=optim;w.actor_lr_scheduler=torch.optim.lr_scheduler.LambdaLR(optim,lambda n:min(1,n/57))
 teacher=Teacher()
 # Fixed teacher parameters must be reconstructed from identical initial assets.
 oldteacher=w.actor.teacher;teacher.model.load_state_dict(oldteacher.model.state_dict());w.actor.teacher=teacher
 with pytest.raises(ValueError):w.load_training_state(str(path/'recovery'),5)
 rank_binding=dict(version=1,authorization='EXPLICIT_NEW_PROCESS_RECOVERY',rank=0,world_size=1,step=5,
  source_state_digest=report['state_digest'],source_pid=saved['pid'],source_object_ids=saved['object_ids'],
  destination_pid=os.getpid(),destination_object_ids=dict(model=id(model),optimizer=id(optim),scheduler=id(w.actor_lr_scheduler),teacher=id(teacher)),
  recovery_identity=saved['recovery_identity'],teacher_asset_files=saved['teacher']['asset_files'])
 binding=path/'recovery-binding.json';binding.write_text(json.dumps(dict(version=1,checkpoint_sha256=report['sha256'],rank_binding=rank_binding)))
 result=w.load_training_state(str(path/'recovery'),5,recovery_mode='authorized_new_process',recovery_binding_path=str(binding),recovery_binding_sha256=file_digest(binding))
 assert result['status']=='EXACT_NEW_PROCESS_FULL_STATE_RELOAD_OK'
 update(w);assert state_digest(recovery_core(w._training_state_payload(6)))==oracle
 rank_binding['recovery_identity']=dict(saved['recovery_identity'],actor_config_digest='0'*64)
 with pytest.raises(ValueError):w._load_training_state_payload(saved,5,report['state_digest'],rank_binding)



def test_cpu_child_process_authorized_recovery(worker):
 import subprocess,json,os,time,sys
 from search_r1.baselines.full_state import recovery_core,file_digest
 w,path=worker
 for _ in range(5):update(w)
 report=w.save_training_state(str(path/'child-state'),5)
 torch.save(w.actor.teacher.model.state_dict(),path/'teacher-initial.pt')
 update(w);oracle=state_digest(recovery_core(w._training_state_payload(6)))
 script="""
import importlib.util,json,sys,time,os
from pathlib import Path
import torch,pytest
from search_r1.baselines.full_state import recovery_core,state_digest,file_digest
root=Path(sys.argv[1]);spec=importlib.util.spec_from_file_location('state_tests',sys.argv[2]);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
(root/'new-worker').mkdir();w,_=module.worker.__wrapped__(root/'new-worker',pytest.MonkeyPatch())
w.config.actor.distillation.teacher_path=str(root/'assets')
w.actor.teacher.model.load_state_dict(torch.load(root/'teacher-initial.pt',weights_only=False))
try:w.load_training_state(str(root/'child-state'),5)
except ValueError:pass
else:raise AssertionError('Default allowed new process')
(root/'child-ids.json').write_text(json.dumps(dict(pid=os.getpid(),ids=dict(model=id(w.actor_module_fsdp),optimizer=id(w.actor_optimizer),scheduler=id(w.actor_lr_scheduler),teacher=id(w.actor.teacher)))))
limit=time.monotonic()+20
while not (root/'child-authorization.json').exists():
 if time.monotonic()>limit:raise TimeoutError('External parent binding missing')
 time.sleep(.01)
binding=root/'child-authorization.json'
result=w.load_training_state(str(root/'child-state'),5,recovery_mode='authorized_new_process',recovery_binding_path=str(binding),recovery_binding_sha256=file_digest(binding))
module.update(w)
(root/'child-result.json').write_text(json.dumps(dict(status=result['status'],digest=state_digest(recovery_core(w._training_state_payload(6))))))
"""
 process=subprocess.Popen([sys.executable,'-c',script,str(path),str(Path(__file__).resolve())],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 try:
  deadline=time.monotonic()+20
  while not (path/'child-ids.json').exists():
   assert process.poll() is None,process.communicate()
   assert time.monotonic()<deadline
   time.sleep(.01)
  ids=json.loads((path/'child-ids.json').read_text());assert ids['pid']==process.pid and ids['pid']!=os.getpid()
  saved=torch.load(path/'child-state/rank_0.pt',weights_only=False)
  rank_binding=dict(version=1,authorization='EXPLICIT_NEW_PROCESS_RECOVERY',rank=0,world_size=1,step=5,
   source_state_digest=report['state_digest'],source_pid=saved['pid'],source_object_ids=saved['object_ids'],
   destination_pid=ids['pid'],destination_object_ids=ids['ids'],recovery_identity=saved['recovery_identity'],teacher_asset_files=saved['teacher']['asset_files'])
  binding=path/'child-authorization.json';temp=path/'authorization.tmp'
  temp.write_text(json.dumps(dict(version=1,checkpoint_sha256=report['sha256'],rank_binding=rank_binding)));temp.rename(binding)
  stdout,stderr=process.communicate(timeout=20);assert process.returncode==0,(stdout,stderr)
  assert json.loads((path/'child-result.json').read_text())==dict(status='EXACT_NEW_PROCESS_FULL_STATE_RELOAD_OK',digest=oracle)
 finally:
  if process.poll() is None:process.kill();process.communicate(timeout=5)


@pytest.mark.parametrize('arguments',[{'recovery_mode':'unsafe'},{'recovery_binding_path':'foreign'}, {'recovery_mode':'authorized_new_process'}])
def test_recovery_mode_cannot_silently_bypass_default(worker,arguments):
 w,path=worker;update(w);w.save_training_state(str(path/'strict-recovery'),5)
 with pytest.raises(ValueError):w.load_training_state(str(path/'strict-recovery'),5,**arguments)


@pytest.mark.parametrize('fault',['shadow_origin','source_drift','git_commit_drift'])
def test_actual_recovery_source_closure_refuses_drift(worker,monkeypatch,fault):
 import importlib,types
 import search_r1.baselines.full_state as full
 w,path=worker;update(w);saved=w._training_state_payload(5)
 if fault=='shadow_origin':
  actual=importlib.import_module('search_r1.baselines.distillation')
  monkeypatch.setattr(actual,'__file__',str(path/'foreign-distillation.py'))
 elif fault=='source_drift':
  original=full.file_digest
  monkeypatch.setattr(full,'file_digest',lambda p:'0'*64 if str(p).endswith('distillation.py') else original(p))
 else:
  import subprocess
  monkeypatch.setattr(subprocess,'check_output',lambda *a,**k:'0'*40+'\n')
 with pytest.raises(ValueError):w._load_training_state_payload(saved,5,state_digest(saved))



def test_actual_rank_checker_pinned_bytes_and_loaded_runtime(tmp_path):
 import importlib,json,hashlib
 from search_r1.baselines.full_state import verify_rank_import_checker,file_digest
 checker=importlib.import_module('reproduction.baselines.e1_import_bootstrap')
 closure=checker.collect(ROOT,required=('search_r1.baselines.full_state','verl.workers.fsdp_workers'))
 path=tmp_path/'rank-closure.json';path.write_text(json.dumps(closure))
 record=dict(checker_path=str(ROOT/'reproduction/baselines/e1_import_bootstrap.py'),checker_sha256=file_digest(ROOT/'reproduction/baselines/e1_import_bootstrap.py'),closure_path=str(path.resolve()),closure_sha256=file_digest(path))
 proof=verify_rank_import_checker(ROOT,record,0)
 assert proof['pid']==__import__('os').getpid() and proof['rank']==0
 assert proof['verified_function_count']>0
 assert proof['actual_modules']['verl.workers.fsdp_workers']['path']==str(ROOT/'verl/workers/fsdp_workers.py')
 for key in ('checker_sha256','closure_sha256'):
  with pytest.raises(ValueError):verify_rank_import_checker(ROOT,dict(record,**{key:'0'*64}),0)
 closure['modules'].pop('reproduction.baselines.e1_import_bootstrap')
 path.write_text(json.dumps(closure));record['closure_sha256']=file_digest(path)
 with pytest.raises(ValueError):verify_rank_import_checker(ROOT,record,0)
