"""Execute pinned colocation factory and Ray packaging without starting a GPU/Ray job."""
import ast,copy,os,pathlib,sys,types
from unittest.mock import patch
import pytest
import ray
import ray.cloudpickle as cloudpickle
import observed_worker as observer
HERE=pathlib.Path(__file__).resolve().parent
SOURCE=pathlib.Path(os.environ.get('R1_SOURCE',str(HERE.parent/'search-r1-32b-grpo100-source-v2')))
MAGIC='attrs_3141562937'

def actual_contract_module():
 module=types.ModuleType('r1_factory_cpu_contract');sys.modules[module.__name__]=module
 ns=module.__dict__;ns.update(os=os,socket=__import__('socket'),ray=ray,patch=patch,MAGIC_ATTR=MAGIC,Dict=dict,Any=object,register=lambda *a,**kw:lambda f:f)
 # Real canonical Worker constructor, not Actor's config/role signature.
 worker_tree=ast.parse((SOURCE/'verl/single_controller/base/worker.py').read_text())
 classes=[]
 for c in worker_tree.body:
  if isinstance(c,ast.ClassDef) and c.name in ('WorkerHelper','WorkerMeta','Worker'):
   c=copy.deepcopy(c)
   if c.name=='Worker':c.body=[n for n in c.body if isinstance(n,ast.FunctionDef) and n.name in ('__new__','__init__','_configure_with_meta')]
   classes.append(c)
 exec(compile(ast.Module(body=classes,type_ignores=[]),'original-worker-contract','exec'),ns)
 # Exact original generic argument carrier and real Ray-specific carrier ctor.
 tree=ast.parse((SOURCE/'verl/single_controller/base/worker_group.py').read_text());cl=next(c for c in tree.body if isinstance(c,ast.ClassDef) and c.name=='ClassWithInitArgs');exec(compile(ast.Module(body=[cl],type_ignores=[]),'original-carrier','exec'),ns)
 tree=ast.parse((SOURCE/'verl/single_controller/ray/base.py').read_text());nodes=[]
 for n in tree.body:
  if isinstance(n,ast.ClassDef) and n.name=='RayClassWithInitArgs':
   n=copy.deepcopy(n);n.body=[f for f in n.body if isinstance(f,ast.FunctionDef) and f.name=='__init__'];nodes.append(n)
  elif isinstance(n,ast.FunctionDef) and n.name in ('_bind_workers_method_to_parent','_unwrap_ray_remote','create_colocated_worker_cls'):nodes.append(n)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),'original-colocation-factory','exec'),ns)
 exec('''
class Actor(Worker):
 def __init__(self,config,role):
  super().__init__();self.config=config;self.role=role;self._is_actor=role=='actor';self.actor=ActorPolicy();self.actor_lr_scheduler=Scheduler();self.actor_optimizer=Optimizer()
 def init_model(self):return 'original_init:'+self.role
 def update_actor(self,data):return self.actor.update_policy(data)
class ActorPolicy:
 def update_policy(self,data):return 'original_update:'+data
class Scheduler:
 last_epoch=0
class Optimizer:
 param_groups=[{'lr':0.}]
Actor.init_model.attrs_3141562937={'dispatch_mode':'ONE_TO_ALL','execute_mode':'ALL','blocking':True}
Actor.update_actor.attrs_3141562937={'dispatch_mode':'DP_COMPUTE_PROTO','execute_mode':'ALL','blocking':True}
''',ns)
 return module

def arrange(monkeypatch,tmp_path):
 for key,value in {'DISABLE_WORKER_INIT':'1','WORLD_SIZE':'1','RANK':'0','MASTER_ADDR':'127.0.0.1','MASTER_PORT':'12345','R1_RESOURCE_AUDIT_DIRECTORY':str(tmp_path)}.items():monkeypatch.setenv(key,value)
 monkeypatch.setattr(observer,'pid_identity',lambda:{'pid':10,'job':'CPU_CONTRACT_NOT_GPU'})
 monkeypatch.setattr(observer,'actual_gpu',lambda:[])
 monkeypatch.setattr(observer,'allocator_evidence',lambda t:{'scope':'CPU_FACTORY_ONLY_NO_CUDA_ALLOCATOR_EVIDENCE'})
 monkeypatch.setattr(observer.torch.distributed,'get_rank',lambda:0)
 monkeypatch.setattr(observer.torch.cuda,'memory_allocated',lambda:0);monkeypatch.setattr(observer.torch.cuda,'memory_reserved',lambda:0)
 monkeypatch.setattr(observer,'observed_update',lambda actor,original,data:(original(data),{'scope':'CPU_WRAPPER_CALL_CONTRACT_NOT_ACTUAL_GPU_COUNTS'}))

def construct(module):
 carrier=module.RayClassWithInitArgs
 actor=ray.remote(module.Actor)
 spec=module.create_colocated_worker_cls({'actor':carrier(actor,config='cfg',role='actor'),'ref':carrier(actor,config='cfg',role='ref')})
 cls=spec.cls.__ray_actor_class__;return cls,cls()

def test_true_factory_original_constructor_and_register_metadata(monkeypatch,tmp_path):
 arrange(monkeypatch,tmp_path);m=actual_contract_module();before=(m.Actor.__bases__,m.Actor.__init__);attrs={n:getattr(getattr(m.Actor,n),MAGIC).copy() for n in ('init_model','update_actor')}
 assert observer.install(m.Actor) is m.Actor;assert before==(m.Actor.__bases__,m.Actor.__init__)
 for n in attrs:assert getattr(getattr(m.Actor,n),MAGIC)==attrs[n]
 cls,instance=construct(m)
 assert cls.__base__ is m.Worker and instance._rank==0
 assert instance.actor_init_model()=='original_init:actor' and instance.ref_init_model()=='original_init:ref'
 assert instance.actor_update_actor('rows')=='original_update:rows'
 assert instance.worker_dict['actor']._resource_updates[0]['actual_outer_update_ordinal']==1
 assert instance.worker_dict['actor'].actor._r1_forward_diagnostic_context['actual_outer_update_ordinal']==1
 assert instance.worker_dict['actor'].actor._r1_forward_diagnostic_context['rank']==0
 assert (tmp_path/'actor-updates-rank0-outer1.json').is_file()

def test_actual_ray_cloudpickle_roundtrip_retains_inplace_observer(monkeypatch,tmp_path):
 arrange(monkeypatch,tmp_path);m=actual_contract_module();observer.install(m.Actor);cloudpickle.register_pickle_by_value(m)
 try:
  carrier=m.RayClassWithInitArgs;actor=ray.remote(m.Actor);spec=m.create_colocated_worker_cls({'actor':carrier(actor,config='cfg',role='actor')})
  cls=cloudpickle.loads(cloudpickle.dumps(spec.cls.__ray_actor_class__))
  restored=cls();assert restored.actor_init_model()=='original_init:actor'
  assert restored.actor_update_actor('rows')=='original_update:rows'
  assert restored.worker_dict['actor']._resource_updates[0]['actual_outer_update_ordinal']==1
 finally:cloudpickle.unregister_pickle_by_value(m)

def test_old_subclass_reproduces_constructor_failure(monkeypatch,tmp_path):
 arrange(monkeypatch,tmp_path);m=actual_contract_module()
 class OldObserved(m.Actor):pass
 carrier=m.RayClassWithInitArgs;old=ray.remote(OldObserved)
 spec=m.create_colocated_worker_cls({'actor':carrier(old,config='cfg',role='actor')})
 with pytest.raises(TypeError,match='config.*role'):spec.cls.__ray_actor_class__()
