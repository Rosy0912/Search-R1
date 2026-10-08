"""Same original main_task body with external observed worker; parent has leases."""
import argparse,json,os,pathlib,sys
ROOT=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--command-json',required=True);a=ap.parse_args()
 config=json.loads(pathlib.Path(a.config).read_text());command=json.loads(pathlib.Path(a.command_json).read_text())
 candidate=pathlib.Path(config['candidate']).resolve();os.chdir(candidate);sys.path.insert(0,str(candidate))
 # Keep external observer importable on all same-node Ray worker processes.
 here=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(here))
 os.environ['PYTHONPATH']=str(here)+os.pathsep+os.environ.get('PYTHONPATH','')
 from hydra import compose,initialize_config_dir
 from omegaconf import OmegaConf
 with initialize_config_dir(config_dir=str(candidate/'verl/trainer/config'),version_base=None):resolved=compose(config_name='ppo_trainer',overrides=command[3:])
 from observed_worker import ObservedWorker
 import verl.workers.fsdp_workers as workers
 workers.ActorRolloutRefWorker=ObservedWorker
 import verl.trainer.main_ppo_eco as original_entry
 from terminal import held_fit
 original_fit=original_entry.RayPPOTrainer.fit
 class HeldTrainer(original_entry.RayPPOTrainer):
  def fit(self):
   binding=next(x.split('=',1)[1] for x in command[3:] if x.lstrip('+').startswith('trainer.g1_owned_pipeline_binding='))
   return held_fit(original_fit,self,config,binding)
 original_entry.RayPPOTrainer=HeldTrainer
 main_task=original_entry.main_task
 import ray
 audit=str(pathlib.Path(config['evidence_root'])/'actor-resource-audit');os.environ['G1_RESOURCE_AUDIT_DIRECTORY']=audit
 parent_env={k:v for k,v in os.environ.items() if k in ('CPATH','PYTHONPATH','G1_RESOURCE_AUDIT_DIRECTORY','SLURM_JOB_ID','SLURM_JOB_NODELIST','CUDA_VISIBLE_DEVICES','SEARCH_R1_GPU_LEASES','SEARCH_R1_GPU_PREFLIGHT','G1_GUARDIAN_CONTROL','PROBE_URL','PROBE_MANIFEST_PATH') or k.startswith(('ECO_','HF_','TRANSFORMERS_'))}
 parent_env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='true',NCCL_DEBUG='WARN')
 temp=pathlib.Path('/dev/shm')/('g1r-'+os.environ['SLURM_JOB_ID']);temp.mkdir(mode=0o700,exist_ok=False)
 os.environ.pop('RAY_ADDRESS',None);os.environ.pop('RAY_BOOTSTRAP_ADDRESS',None)
 ray.init(address='local',num_cpus=24,include_dashboard=False,_temp_dir=str(temp),runtime_env={'env_vars':parent_env})
 try:main_task._function(resolved)
 finally:ray.shutdown()
if __name__=='__main__':main()
