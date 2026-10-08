"""Same Hydra baseline entry, with an observer in its actual Ray main task."""
import hashlib,importlib.abc,importlib.machinery,os,sys
from pathlib import Path
from validation_cost_hooks_instruct300 import measured_remote,install as install_validation
from training_cost_hooks import install as install_training
from rolling_training_state_hooks import install as install_rolling_training_state

def install_human_direct_continuation():
 """Keep the existing live trainer; human authority replaces independent step5 approval."""
 from production_contract import read_bound,descriptor,require
 tools=Path(os.environ['TREE_INSTRUCT_TOOLS'])
 package,manifest_sha,_=read_bound(tools/'manifest.json',os.environ['TREE_INSTRUCT_PACKAGE_SHA256'])
 if package.get('execution_mode')!='HUMAN_AUTHORIZED_DIRECT_TRAIN200':return
 authority,authority_sha,_=descriptor(package['human_training_authority'])
 require(authority['status']=='HUMAN_AUTHORIZED_DIRECT_TRAIN200'
         and authority['code_commit']==package['code_commit']
         and authority['same_instance_continuation'] is True
         and authority['completed_outer_goal']==200 and authority['final_evaluation_rows']==51713,
         'Direct same-instance human training authority differs')
 require(package['production_thresholds_accepted'] is False
         and package['original_sampler_actor_gate_accepted'] is False,
         'Human authority cannot imply observed threshold/sampler acceptance')
 from search_r1.baselines import training_state
 def continue_after_five(directory,ready,timeout_seconds=43200):
  expected=dict(job_id=os.environ['SLURM_JOB_ID'],code_commit=package['code_commit'],
      config_sha256=os.environ['SEARCH_R1_BASELINE_CONFIG_SHA256'],
      data_identity=os.environ['SEARCH_R1_BASELINE_DATA_IDENTITY'],
      model_identity=os.environ['SEARCH_R1_MODEL_IDENTITY'])
  run_id=os.environ['SEARCH_R1_BASELINE_RUN_ID']
  target=Path(os.environ['SEARCH_R1_BASELINE_OUTPUT'])/'acceptance-step5'
  require(Path(directory).resolve()==target.resolve()
          and ready['completed_outer_updates']==ready['completed_step']==5
          and ready['next_outer_update']==6 and ready['actual_update_limit']==200
          and all(ready.get(key)==value for key,value in expected.items()),
          'Human continuation requires this actual live fifth credited outer')
  target.mkdir(parents=True,exist_ok=True)
  ready_sha=training_state.atomic_save(ready,target/'ready.json',json_format=True)
  record=dict(status='HUMAN_AUTHORIZED_SAME_INSTANCE_6_TO_200',**expected,
      run_id=run_id,completed_outer_updates=5,next_outer_update=6,actual_update_limit=200,
      ready_sha256=ready_sha,manifest_sha256=manifest_sha,human_training_authority_sha256=authority_sha,
      same_instance_continuation=True,independent_acceptance=False,
      actual_training_accepted=False,production_thresholds_accepted=False,
      original_sampler_actor_gate_accepted=False,
      scope='Latest human instruction authorizes live training continuation; no independent PASS or threshold acceptance')
  training_state.atomic_save(record,target/'human-authorized-continuation.json',json_format=True)
  return record
 training_state.wait_for_acceptance=continue_after_five

def install_production_observers():
 from current_chain_runtime import install_ray_chain_environment
 install_ray_chain_environment()
 install_validation()
 install_training()
 install_rolling_training_state()
 install_human_direct_continuation()
TARGET='verl.trainer.main_ppo'
EXPECTED=os.environ['TREE_FRESH_MAIN_PPO_SHA256']

class Loader(importlib.abc.Loader):
 def __init__(self,original):self.original=original
 def create_module(self,spec):return self.original.create_module(spec)
 def exec_module(self,module):
  assert hashlib.sha256(open(module.__file__,'rb').read()).hexdigest()==EXPECTED
  self.original.exec_module(module)
  original=module.main_task
  # The fixed original source uses @ray.remote with no overridden resources.
  assert original._default_options=={}
  from current_chain_runtime import chain_env_for_ray
  import ray
  env=chain_env_for_ray()
  module.main_task=measured_remote(original._function,installer=install_production_observers)

class Finder(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname!=TARGET:return None
  spec=importlib.machinery.PathFinder.find_spec(fullname,path)
  assert spec is not None and spec.loader is not None
  spec.loader=Loader(spec.loader)
  sys.meta_path.remove(self)
  return spec

if __name__=='__main__':
 assert TARGET not in sys.modules,'Refuse late installation after task import'
 from current_chain_runtime import install_ray_chain_environment
 install_ray_chain_environment()
 sys.meta_path.insert(0,Finder())
 from search_r1.baselines.main import main
 main()
