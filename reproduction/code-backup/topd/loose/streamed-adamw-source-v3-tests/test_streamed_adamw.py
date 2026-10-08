import importlib.util,pathlib,sys,copy,types,os
import pytest
import torch
HERE=pathlib.Path(__file__).resolve().parent
SOURCE=pathlib.Path(os.environ.get('R1_STREAM_SOURCE',str(HERE.parent/'search-r1-32b-grpo100-source-v3')))
package=types.ModuleType('_owned_torch24');package.__path__=[str(HERE)];sys.modules[package.__name__]=package
s=importlib.util.spec_from_file_location('_owned_torch24.optimizer',HERE/'native_torch24_optimizer.py');optimizer_original=importlib.util.module_from_spec(s);sys.modules[s.name]=optimizer_original;s.loader.exec_module(optimizer_original)
s=importlib.util.spec_from_file_location('_owned_torch24.adamw',HERE/'native_torch24_adamw.py');original=importlib.util.module_from_spec(s);sys.modules[s.name]=original;s.loader.exec_module(original)
s=importlib.util.spec_from_file_location('owned_streamed_adamw',SOURCE/'verl/workers/actor/streamed_adamw.py');stream=importlib.util.module_from_spec(s);s.loader.exec_module(stream)

def compare(a,b):
 for ga,gb in zip(a.param_groups,b.param_groups):
  for pa,pb in zip(ga['params'],gb['params']):
   assert torch.equal(pa,pb)
   assert set(a.state[pa])==set(b.state[pb])
   for k in a.state[pa]:assert torch.equal(a.state[pa][k],b.state[pb][k])
@pytest.mark.parametrize('amsgrad',[False,True])
def test_exact_original_multi_groups_lazy_existing_zero_no_grad_hooks(amsgrad):
 torch.manual_seed(8);a=[torch.nn.Parameter(torch.randn(3,4)),torch.nn.Parameter(torch.randn(4).t()[::2]),torch.nn.Parameter(torch.randn(2,3))];b=[torch.nn.Parameter(p.detach().clone(memory_format=torch.preserve_format)) for p in a]
 oa=original.AdamW([{'params':a[:2],'lr':.001},{'params':a[2:],'lr':0.}],foreach=False,amsgrad=amsgrad);ob=original.AdamW([{'params':b[:2],'lr':.001},{'params':b[2:],'lr':0.}],foreach=False,amsgrad=amsgrad)
 hooks=[];oa.register_step_pre_hook(lambda *args:hooks.append('pre'));oa.register_step_post_hook(lambda *args:hooks.append('post'))
 for step in range(4):
  for i,(pa,pb) in enumerate(zip(a,b)):
   grad=torch.zeros_like(pa) if step==2 else torch.full_like(pa,step+.3)
   pa.grad=None if i==1 and step==0 else grad;pb.grad=None if pa.grad is None else grad.clone()
  with stream._namespace(oa,original):oa.step()
  ob.step();compare(oa,ob)
 assert hooks==['pre','post']*4;assert all(t.device.type=='cpu' for state in oa.state.values() for t in state.values())
@pytest.mark.parametrize('flag,value',[('foreach',True),('fused',True),('capturable',True),('differentiable',True)])
def test_unsupported_paths_fail_before_step(flag,value):
 p=torch.nn.Parameter(torch.ones(3));o=original.AdamW([p],foreach=False);o.param_groups[0][flag]=value;p.grad=torch.ones_like(p)
 with pytest.raises(RuntimeError):
  with stream._namespace(o,original):o.step()
 assert not o.state and torch.equal(p,torch.ones(3))

def test_exception_partial_state_restored_namespace_and_cpu(monkeypatch):
 p=torch.nn.Parameter(torch.ones(3));q=torch.nn.Parameter(torch.ones(2));o=original.AdamW([p,q],foreach=False);p.grad=torch.ones_like(p);q.grad=torch.ones_like(q);actual=original._single_tensor_adamw;calls=[]
 def failing(*args,**kwargs):
  calls.append(args[0][0]);actual(*args,**kwargs)
  if len(calls)==2:raise RuntimeError('original-singleton-exception')
 monkeypatch.setattr(original,'_single_tensor_adamw',failing)
 with pytest.raises(RuntimeError,match='original-singleton-exception'):
  with stream._namespace(o,original):o.step()
 assert original._single_tensor_adamw is failing;assert '_init_group' not in o.__dict__;assert stream._ACTIVE is False
 assert all(t.device.type=='cpu' for st in o.state.values() for t in st.values());assert all(st['step'].item()==1 for st in o.state.values())

def test_nested_rejected_outer_original_restored():
 p=torch.nn.Parameter(torch.ones(2));o=original.AdamW([p],foreach=False);original_single=original._single_tensor_adamw
 with stream._namespace(o,original):
  with pytest.raises(RuntimeError,match='Nested'):
   with stream._namespace(o,original):pass
 assert original._single_tensor_adamw is original_single

def test_other_optimizer_in_same_namespace_uses_original_math():
 p=torch.nn.Parameter(torch.ones(2));q=torch.nn.Parameter(torch.ones(2));base=torch.nn.Parameter(q.detach().clone());owned=original.AdamW([p],foreach=False);other=original.AdamW([q],foreach=False);reference=original.AdamW([base],foreach=False);q.grad=torch.ones_like(q);base.grad=q.grad.clone()
 with stream._namespace(owned,original):other.step()
 reference.step();compare(other,reference)

def test_closure_once_and_original_step_hooks_once():
 p=torch.nn.Parameter(torch.ones(2));o=original.AdamW([p],foreach=False);calls=[]
 o.register_step_pre_hook(lambda *args:calls.append('pre'));o.register_step_post_hook(lambda *args:calls.append('post'))
 def closure():
  calls.append('closure');o.zero_grad();loss=p.square().sum();loss.backward();return loss
 with stream._namespace(o,original):loss=o.step(closure)
 assert loss.item()==2. and calls==['pre','closure','post'] and o.state[p]['step'].item()==1

def test_setup_failure_restores_active_and_namespace(monkeypatch):
 p=torch.nn.Parameter(torch.ones(2));o=original.AdamW([p],foreach=False);saved=original._single_tensor_adamw
 def fail(*args,**kwargs):raise RuntimeError('injected-before-patch-setup')
 monkeypatch.setattr(stream.types,'FunctionType',fail)
 with pytest.raises(RuntimeError,match='setup'):
  with stream._namespace(o,original):pass
 assert original._single_tensor_adamw is saved and not stream._ACTIVE and '_init_group' not in o.__dict__

def test_other_thread_same_optimizer_rejected_no_ownership_interference():
 import threading
 p=torch.nn.Parameter(torch.ones(2));o=original.AdamW([p],foreach=False);p.grad=torch.ones_like(p);errors=[]
 with stream._namespace(o,original):
  def run():
   try:o.step()
   except BaseException as e:errors.append(e)
  t=threading.Thread(target=run);t.start();t.join(5)
 assert len(errors)==1 and 'concurrently' in str(errors[0]) and torch.equal(p,torch.ones(2))

def test_actual_actor_step_selects_streaming_without_whole_state_load(monkeypatch):
 import ast
 path=SOURCE/'verl/workers/actor/dp_actor.py';tree=ast.parse(path.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DataParallelPPOActor');method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_optimizer_step');method.decorator_list=[];events=[]
 monkeypatch.setattr(torch.cuda,'synchronize',lambda:events.append('sync'))
 def step(o):
  events.append('streamed-original-step')
  with stream._namespace(o,original):o.step()
 ns={'torch':torch,'FSDP':type('NotFSDP',(),{}),'step_cpu_offloaded_adamw':step,'offload_fsdp_optimizer':lambda o:events.append('final-cpu-offload'),'load_fsdp_optimizer':lambda *args,**kwargs:pytest.fail('Whole state H2D must not run')}
 exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),ns)
 model=torch.nn.Linear(3,2);o=original.AdamW(model.parameters(),foreach=False);model(torch.ones(2,3)).sum().backward();actor=types.SimpleNamespace(config=types.SimpleNamespace(grad_clip=1.,fsdp_config={'optimizer_offload':True}),actor_module=model,actor_optimizer=o)
 norm=ns['_optimizer_step'](actor);assert torch.isfinite(norm) and events==['sync','streamed-original-step','final-cpu-offload'];assert all(state['step'].item()==1 for state in o.state.values())

def test_partial_staging_exception_cpu_states_and_namespace(monkeypatch):
 p=torch.nn.Parameter(torch.ones(3));q=torch.nn.Parameter(torch.ones(2));o=original.AdamW([p,q],foreach=False);p.grad=torch.ones_like(p);q.grad=torch.ones_like(q);native_to=torch.Tensor.to;count=[]
 def fail_second_state(self,*args,**kwargs):
  if kwargs.get('device')==p.device:
   count.append(1)
   if len(count)==2:raise RuntimeError('partial-moment-device-transfer')
  return native_to(self,*args,**kwargs)
 monkeypatch.setattr(torch.Tensor,'to',fail_second_state)
 with pytest.raises(RuntimeError,match='partial-moment'):
  with stream._namespace(o,original):o.step()
 assert not stream._ACTIVE and all(t.device.type=='cpu' for state in o.state.values() for t in state.values())


def test_shared_parameter_other_optimizer_state_rejected_before_update():
 p=torch.nn.Parameter(torch.ones(3));owned=original.AdamW([p],foreach=False);other=original.AdamW([p],foreach=False);p.grad=torch.ones_like(p)
 with stream._namespace(owned,original):
  with pytest.raises(RuntimeError,match='Shared parameter'):other.step()
 assert torch.equal(p,torch.ones(3)) and not owned.state and other.state[p]['step'].item()==0

def test_existing_instance_initializer_override_rejected():
 p=torch.nn.Parameter(torch.ones(3));o=original.AdamW([p],foreach=False);native=o._init_group
 def custom(self,*args,**kwargs):return native(*args,**kwargs)
 o._init_group=types.MethodType(custom,o)
 with pytest.raises(RuntimeError,match='initializer'):
  with stream._namespace(o,original):pass
 assert o._init_group.__func__ is custom and not stream._ACTIVE

@pytest.mark.skipif(torch.__version__.split('+')[0]!='2.4.0',reason='Public byte-pinned entry requires actual native Torch2.4')
def test_native_public_entry_actual_installed_torch24():
 p=torch.nn.Parameter(torch.ones(4));q=torch.nn.Parameter(p.detach().clone());a=torch.optim.AdamW([p],foreach=False);b=torch.optim.AdamW([q],foreach=False)
 for i in range(3):
  p.grad=torch.full_like(p,.25*(i+1));q.grad=p.grad.clone();stream.step_cpu_offloaded_adamw(a);b.step();compare(a,b)
