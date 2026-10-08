"""Actual frozen adapter + observer, tiny CPU stand-ins; no helper/GPU/model."""
from pathlib import Path
from types import SimpleNamespace as N
import ast,importlib.util,json,hashlib,torch,traceback
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'tree-behavior-mu-ppo/search_r1/baselines/controller.py'
HOOK=ROOT.parent/'tree-fresh200-mu-runtime-v7/validation_cost_hooks_instruct300.py'
seen={'adapter_rows':0,'legacy_loop_calls':0,'search_run_calls':0}
class WG:
 def generate_sequences(self,batch):return N(batch={'responses':torch.ones((512,2),dtype=torch.long)})
class Manager:
 def __init__(self,wg):self.actor_rollout_wg=wg;self.config=N(num_gpus=8)
 def run_llm_loop(self,batch,initial):seen['legacy_loop_calls']+=1;return self._generate_with_gpu_padding(batch)
 def _generate_with_gpu_padding(self,batch):return self.actor_rollout_wg.generate_sequences(batch)
 def _batch_search(self,queries):return {'result':[[] for _ in queries]}
class SearchRolloutStub:
 def __init__(self,manager,**kwargs):self.manager=manager
 def run(self,batch):
  seen['search_run_calls']+=1;seen['adapter_rows']+=int(batch.batch['input_ids'].shape[0])
  return self.manager._generate_with_gpu_padding(batch),None
node=next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='token_preserving_validation_manager')
module=ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]));env={'SearchRollout':SearchRolloutStub}
exec(compile(module,str(SOURCE),'exec'),env)
adapter_factory=env['token_preserving_validation_manager']
spec=importlib.util.spec_from_file_location('actual_frozen_validation_hook',HOOK);hook=importlib.util.module_from_spec(spec);spec.loader.exec_module(hook)
class Trainer:
 def __init__(self):
  self.config=N(do_search=True);self.global_steps=0;self.tokenizer=N(pad_token_id=0,eos_token_id=2)
  self.val_dataset=list(range(512));self.actor_rollout_wg=WG()
 def _prepare_final_validation(self):pass
 def _validate(self):
  gen=N(batch={'input_ids':torch.ones((512,2),dtype=torch.long),'attention_mask':torch.ones((512,2),dtype=torch.long)})
  ctl=N(config=N(tree=N(token_preservation=N(enabled=True))))
  adapter=adapter_factory(Manager(self.actor_rollout_wg),ctl)
  out=adapter.run_llm_loop(gen,gen.batch['input_ids'])
  assert out.batch['responses'].shape[0]==512
  return {'val/evaluated_count/nq':256,'val/evaluated_count/hotpotqa':256}
AUDIT=ROOT/'w05-validation-counterexample';AUDIT.mkdir(exist_ok=True)
hook.wrap_trainer(Trainer,Manager,AUDIT,final_expected_rows=51713)
try:Trainer()._validate();error=None
except Exception as exc:error={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
reportpaths=list(AUDIT.glob('validation-*.json'));assert len(reportpaths)==1
report=json.loads(reportpaths[0].read_text())
assert seen['adapter_rows']==512 and seen['search_run_calls']==1 and seen['legacy_loop_calls']==0
assert report['dataset_rows']==512 and report['counters']['real_sample_rows']==0
assert report['counters']['generation_completed']==1 and report['counters']['returned_rows']==512
assert error and error['type']=='AssertionError' and 'coverage' in error['message']
result={'seen':seen,'error':error,'observer_report':str(reportpaths[0]),'observer_counters':report['counters'],'status':report['status'],'phase':report['phase'],'controller_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'observer_sha256':hashlib.sha256(HOOK.read_bytes()).hexdigest(),'actual_adapter_function_ast':True,'actual_frozen_observer':True,'SearchRollout_standin':True,'actual_model_data_gpu_remote_slurm_loads':0,'scope':'Actual adapter+observer integration CPU counterexample with512 synthetic rows; no actual _validate/model/SearchRollout generation claim'}
(ROOT/'W05_VALIDATION_OBSERVER_CPU_COUNTEREXAMPLE.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
