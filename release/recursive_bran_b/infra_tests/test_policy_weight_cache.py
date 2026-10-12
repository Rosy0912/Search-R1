"""Small CPU control-flow test of the actual manager methods; no GPU claim."""
import ast,copy,json,pathlib,time,types,unittest
source=pathlib.Path(__file__).resolve().parents[3]/'verl/workers/sharding_manager/fsdp_vllm.py'
node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='FSDPVLLMShardingManager')
methods=[n for n in node.body if isinstance(n,ast.FunctionDef) and n.name in ('prepare_policy_cache','release_model_weights','__enter__','__exit__')]
class CUDA:
 def __init__(self):self.state=10
 def get_rng_state(self):return self.state
 def set_rng_state(self,x):self.state=x
 def empty_cache(self):pass
class Model:
 def __init__(self):self.value=7;self.gathers=0
 def state_dict(self):self.gathers+=1;return {'weight':self.value}
 def train(self):pass
class Engine:
 def __init__(self):self.value=None;self.syncs=0;self.offloads=0
 def sync_model_weights(self,p,load_format):self.value=p['weight'];self.syncs+=1
 def offload_model_weights(self):self.value=None;self.offloads+=1
class Check(unittest.TestCase):
 def make(self):
  cuda=CUDA();ns=dict(torch=types.SimpleNamespace(cuda=cuda),log_gpu_memory_usage=lambda *a,**k:None,logger=None,json=json,time=time)
  wrapper=ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(name='Manager',bases=[],keywords=[],body=copy.deepcopy(methods),decorator_list=[])],type_ignores=[]))
  exec(compile(wrapper,str(source),'exec'),ns);m=ns['Manager']();m.module=Model();m.inference_engine=Engine();m.full_params=False;m.device_mesh=object();m.gen_random_states=100;m.torch_random_states=10
  m.retain_model_weights=False;m.weights_resident=False;m.cached_policy_version=None;m.requested_policy_version=None;m.weight_sync_calls=0;m.cache_phase_calls=0;m.cache_phase_sync_wall_seconds=0.
  return m,cuda
 def calls(self,m,cuda,retain):
  seen=[]
  for _ in range(3):
   m.prepare_policy_cache(retain,(80,4));m.__enter__();seen.append((m.inference_engine.value,cuda.state));cuda.state+=2;m.__exit__(None,None,None);self.assertEqual(cuda.state,10)
  return seen
 def test_reuse_same_weights_and_rng(self):
  old,c=self.make();expected=self.calls(old,c,False)
  m,c=self.make();self.assertEqual(self.calls(m,c,True),expected);self.assertEqual(m.inference_engine.syncs,1);self.assertEqual(m.module.gathers,1);self.assertEqual(m.inference_engine.offloads,0)
  m.release_model_weights();self.assertEqual(m.inference_engine.offloads,1);self.assertFalse(m.weights_resident)
 def test_changed_policy_refreshes(self):
  m,c=self.make();self.calls(m,c,True);m.module.value=9;m.prepare_policy_cache(True,(100,5));self.assertFalse(m.weights_resident);m.__enter__();self.assertEqual(m.inference_engine.value,9);self.assertEqual(m.inference_engine.syncs,2);m.__exit__(None,None,None)
 def test_validation_disables_retention(self):
  m,c=self.make();self.calls(m,c,True);m.prepare_policy_cache(False,None);m.__enter__();m.__exit__(None,None,None);self.assertFalse(m.weights_resident)
 def test_exception_releases(self):
  m,c=self.make();m.prepare_policy_cache(True,(80,4));m.__enter__();m.__exit__(RuntimeError,RuntimeError('test'),None);self.assertFalse(m.weights_resident);self.assertEqual(c.state,10)
if __name__=='__main__':unittest.main()
