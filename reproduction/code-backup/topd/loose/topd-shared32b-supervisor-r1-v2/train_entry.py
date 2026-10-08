"""Owned CPU driver: original source main_task body, observed worker subclass."""
import argparse,json,os,sys
from pathlib import Path
from contracts import *
from dependencies import gate,recipe,load
from lifecycle import publish,wait

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--mode',required=True);p.add_argument('--run',required=True);p.add_argument('--ipc',required=True);a=p.parse_args()
 config=json.loads(Path(a.config).read_text());selected=gate(config,a.mode)
 repo=own(config['repo']);os.chdir(repo);sys.path.insert(0,str(repo))
 ipcgate=load('import_gate',config['owner_import_gate']);ipcgate.verified_ipc()
 from topd32b_shared_pipeline.protocol import process_identity
 actual=process_identity(os.getpid());actual['uid']=os.getuid()
 from hydra import compose,initialize_config_dir
 from omegaconf import OmegaConf
 with initialize_config_dir(config_dir=str(repo/'verl/trainer/config'),version_base=None):
  resolved=compose(config_name='ppo_baselines',overrides=recipe(a.mode,a.run,selected['model'],a.ipc)+config['common_overrides'])
 resolved_recipe(OmegaConf.to_container(resolved,resolve=True))
 from search_r1.baselines.preflight import validate_gpu_lease,validate_assets_and_retriever
 from search_r1.baselines.local_runtime import check_local_runtime
 from search_r1.baselines.holdout import prepare_common_holdout
 from search_r1.baselines.controller import validate_config
 validate_gpu_lease(resolved);check_local_runtime(resolved);split=prepare_common_holdout(resolved)
 validate_config(resolved);validate_assets_and_retriever(resolved)
 yaml=OmegaConf.to_yaml(OmegaConf.create(OmegaConf.to_container(resolved,resolve=True))).encode()
 resolvedsha=__import__('hashlib').sha256(yaml).hexdigest()
 os.environ.update(SEARCH_R1_TOPD_AUDIT_DIR=a.run+'/audit',SEARCH_R1_TOPD_RUN_ID=config['run_id'],SEARCH_R1_TOPD_COMMIT=D23,SEARCH_R1_TOPD_CONFIG_SHA256=resolvedsha,SEARCH_R1_TOPD_DATA_IDENTITY=split['content_identity']['sha256'],TOPD32B_RUN=a.run,TOPD32B_BINDING=a.run+'/binding.json')
 from search_r1.baselines.topd_runtime_audit import config_boundary
 config_boundary(resolved,split)
 publish(Path(a.run)/'driver-hello.json',dict(driver=actual,config_sha256=resolvedsha,data_identity=split['content_identity']['sha256'],source_sha256=config['integrated_source_manifest']['sha256'],student_model_directory=str(resolved.actor_rollout_ref.model.path),code_commit=D23,run_id=config['run_id']))
 wait(Path(a.run)/'actor-start-authorized.json',[],[],3600)
 # Hardware mode is explicit and cannot create algorithm-acceptance receipts.
 if a.mode=='profile':
  import topd32b_shared_pipeline.backend as backend
  original_load=backend.load_production_client
  def hardware_client(manifest_path,teacher_path,student_tokenizer):
   manifest=json.loads(Path(manifest_path).read_text());require(manifest['kind']=='NATIVE_HARDWARE_PROBE_NO_TRAINING','Profile may not use production scope')
   require(Path(teacher_path).resolve()==Path(selected['model']).resolve(),'Profile selected teacher drift')
   return backend.RankTeacherClient(manifest,Path(manifest_path).parent)
  backend.load_production_client=hardware_client
 from shared_supervision_hooks import install
 install(a.mode,resolved)
 import ray
 os.environ['PYTHONPATH']=str(Path(__file__).parent)+os.pathsep+str(repo)+os.pathsep+str(ROOT/'tools/topd-32b-pipeline-ipc-r1-v2')+os.pathsep+os.environ.get('PYTHONPATH','')
 audit_env={k:v for k,v in os.environ.items() if k.startswith(('SEARCH_R1_TOPD_','TOPD32B_'))}
 raytemp=Path('/dev/shm')/('topd32b-ray-'+os.environ['SLURM_JOB_ID']);require(not raytemp.exists(),'Fresh own short Ray directory required');raytemp.mkdir(mode=0o700)
 ray.init(address='local',num_cpus=24,_temp_dir=str(raytemp),runtime_env={'env_vars':dict(audit_env,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH=os.environ['PYTHONPATH'],TOKENIZERS_PARALLELISM='true',NCCL_DEBUG='WARN')})
 check_local_runtime(resolved,ray.nodes())
 import torch
 require(not torch.cuda.is_initialized(),'Parent training driver must not own CUDA context')
 from verl.trainer.main_ppo import main_task
 try:
  main_task._function(resolved)
 finally:ray.shutdown()
 publish(Path(a.run)/'driver-done.json',dict(status='ORIGINAL_DRIVER_RETURNED_UNREVIEWED',mode=a.mode,pid=os.getpid(),parent_cuda_initialized=torch.cuda.is_initialized()))

if __name__=='__main__':main()
