"""Actual d23 factory and original loss contracts; no GPU/resource assertions."""
import ast,copy,importlib.util,json,os,pathlib,sys,types
from unittest.mock import patch
import pytest,ray,torch
import ray.cloudpickle as cloudpickle
HERE=pathlib.Path(__file__).resolve().parent
D23=HERE.parent/'topd-shared32b-integration-candidate'
MAGIC='attrs_3141562937'

def actual_module():
 fixture=HERE.parent/'search-r1-32b-grpo100-tools-v12/test_observer_factory.py'
 node=next(n for n in ast.parse(fixture.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='actual_contract_module')
 ns=dict(ast=ast,copy=copy,os=os,sys=sys,types=types,pathlib=pathlib,ray=ray,patch=patch,SOURCE=D23,MAGIC=MAGIC)
 exec(compile(ast.Module(body=[node],type_ignores=[]),'actual-d23-constructor-factory','exec'),ns)
 m=ns['actual_contract_module']()
 exec('''
class Engine:
 def generate(self,x):return x
class Rollout:
 def __init__(self):self.inference_engine=Engine()
def generate_sequences(self,x):return self.rollout.inference_engine.generate(x)
Actor.generate_sequences=generate_sequences
Actor.generate_sequences.attrs_3141562937={'dispatch_mode':'DP_COMPUTE_PROTO','execute_mode':'ALL','blocking':True}
''',m.__dict__)
 hook=ast.parse((HERE/'shared_supervision_hooks.py').read_text())
 selected=[n for n in hook.body if isinstance(n,ast.FunctionDef) and n.name in ('_record','observed_ready','observed_phase','install_worker_observation')]
 def register(**metadata):
  def wrap(f):setattr(f,MAGIC,dict(metadata,execute_mode='ALL',blocking=True));return f
  return wrap
 m.ActorPolicy.update_policy=lambda self,data:data
 m.__dict__.update(json=json,Path=pathlib.Path,register=register,Dispatch=types.SimpleNamespace(ONE_TO_ALL='ONE_TO_ALL',DP_COMPUTE_PROTO='DP_COMPUTE_PROTO'),events=[],release_and_ack=lambda *a:m.events.append('ACK'))
 exec(compile(ast.Module(body=selected,type_ignores=[]),'actual-shared-worker-wrappers','exec'),m.__dict__)
 return m

def setup(monkeypatch,tmp_path):
 for k,v in dict(DISABLE_WORKER_INIT='1',WORLD_SIZE='1',RANK='0',MASTER_ADDR='127.0.0.1',MASTER_PORT='12345').items():monkeypatch.setenv(k,v)
 path=tmp_path/'binding.json';path.write_text('{"ipc_directory":"CPU_CONTRACT"}');monkeypatch.setenv('TOPD32B_BINDING',str(path))

def construct(m):
 actor=ray.remote(m.Actor);carrier=m.RayClassWithInitArgs
 spec=m.create_colocated_worker_cls(dict(actor=carrier(actor,config='cfg',role='actor'),ref=carrier(actor,config='cfg',role='ref')))
 return spec.cls.__ray_actor_class__

def instrument(m):
 before=(m.Actor.__bases__,m.Actor.__init__);attrs=getattr(m.Actor.update_actor,MAGIC).copy()
 assert m.install_worker_observation(m.Actor) is m.Actor
 assert before==(m.Actor.__bases__,m.Actor.__init__) and getattr(m.Actor.update_actor,MAGIC)==attrs
 def record(self,event,*args):m.events.append(event);return dict(event=event)
 m.Actor._record=record
 return m

def test_actual_d23_factory_preserves_constructor_and_ACK_order(monkeypatch,tmp_path):
 setup(monkeypatch,tmp_path);m=instrument(actual_module());cls=construct(m);instance=cls();assert cls.__base__ is m.Worker
 data=types.SimpleNamespace(meta_info={'topd_outer_step':1});result=instance.actor_update_actor(data)
 # Real original stub output plus actual wrapper boundaries. No actor/GPU execution.
 assert result is data
 assert m.events==['ACK','ALL8_RELEASED_ACKED_BEFORE_ACTOR_UPDATE','ORIGINAL_ACTOR_UPDATE_RETURNED']
 assert instance.ref_init_model()=='original_init:ref'

def test_actual_Ray_serialization_keeps_original_base(monkeypatch,tmp_path):
 setup(monkeypatch,tmp_path);m=instrument(actual_module());cloudpickle.register_pickle_by_value(m)
 try:
  cls=cloudpickle.loads(cloudpickle.dumps(construct(m)));obj=cls()
  assert obj.actor_init_model()=='original_init:actor' and obj.ref_init_model()=='original_init:ref'
 finally:cloudpickle.unregister_pickle_by_value(m)

def test_old_subclass_has_actual_d23_constructor_failure(monkeypatch,tmp_path):
 setup(monkeypatch,tmp_path);m=actual_module()
 class OldObserved(m.Actor):pass
 carrier=m.RayClassWithInitArgs;spec=m.create_colocated_worker_cls(dict(actor=carrier(ray.remote(OldObserved),config='cfg',role='actor')))
 with pytest.raises(TypeError,match='config.*role'):spec.cls.__ray_actor_class__()

def original_zero_adv_norm():
 def function(path,name):return next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
 ns=dict(torch=torch)
 exec(compile(ast.Module(body=[function(D23/'verl/utils/torch_functional.py','masked_sum'),function(D23/'verl/utils/torch_functional.py','masked_mean')],type_ignores=[]),'original-masked-mean','exec'),ns)
 ns['verl_F']=types.SimpleNamespace(masked_mean=ns['masked_mean'])
 exec(compile(ast.Module(body=[function(D23/'verl/trainer/ppo/core_algos.py','compute_policy_loss')],type_ignores=[]),'original-policy-loss','exec'),ns)
 logp=torch.zeros((4,8),requires_grad=True);loss,*_=ns['compute_policy_loss'](torch.zeros_like(logp),logp,torch.zeros_like(logp),torch.ones_like(logp),.2);loss.backward()
 assert float(loss)==0 and bool(torch.isfinite(logp.grad).all());return float(torch.linalg.vector_norm(logp.grad))

def fixture(tmp_path):
 spec=importlib.util.spec_from_file_location('v6_existing_actual_contracts',HERE/'test_contracts.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 return m,m.acceptance_fixture(tmp_path)

def test_original_zero_adv_gradient_is_legal_completed_step(tmp_path):
 norm=original_zero_adv_norm();assert norm==0
 m,b=fixture(tmp_path)
 for rank in range(8):
  p=tmp_path/'audit'/f'{rank}.jsonl';rows=m.read_complete_jsonl(p);rows[0].update(grad_norms=[norm]*20,grad_nonzero=0);p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
 assert m.actual_acceptance(tmp_path,5,b)['completed_actual_outer_steps']==[1,2,3,4,5]
@pytest.mark.parametrize('norm',[float('nan'),float('inf'),-1.])
def test_nonfinite_or_negative_norm_still_rejected(tmp_path,norm):
 m,b=fixture(tmp_path);p=tmp_path/'audit/0.jsonl';rows=m.read_complete_jsonl(p);rows[0]['grad_norms'][0]=norm;p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
 with pytest.raises(RuntimeError):m.actual_acceptance(tmp_path,5,b)

def test_WindowV5_review_function_is_original_AST():
 a=ast.parse((HERE/'telemetry_policy.py').read_text());b=ast.parse((HERE.parent/'search-r1-32b-grpo100-tools-v12/window_guardian.py').read_text())
 fn=lambda t:next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='validate_policy_review')
 assert ast.dump(fn(a),include_attributes=False)==ast.dump(fn(b),include_attributes=False)
 assert (HERE/'retrieval_startup.py').read_bytes()==(HERE.parent/'search-r1-32b-grpo100-tools-v12/retrieval_startup.py').read_bytes()
