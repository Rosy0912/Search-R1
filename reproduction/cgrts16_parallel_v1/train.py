"""Fresh CGRTS7B 16-GPU main; original training body/objective are retained."""
import json,os,socket,sys,threading,time
from pathlib import Path
from .contract import read_plan,sha,projected_config
from .runtime import save,verify_node_records,validate_cluster_boundary

def configuration(plan,host,ip):
 from omegaconf import OmegaConf
 cfg=OmegaConf.create(plan['parent_config_text'])
 oldmodel=cfg.actor_rollout_ref.model.path;olddata=cfg.baselines.holdout.data_root
 def replace(v):
  if isinstance(v,dict):return {k:replace(x) for k,x in v.items()}
  if isinstance(v,list):return [replace(x) for x in v]
  if isinstance(v,str):return v.replace(oldmodel,plan['model']).replace(olddata,plan['data_root'])
  return v
 cfg=OmegaConf.create(replace(OmegaConf.to_container(cfg,resolve=True)))
 cfg.trainer.nnodes=2;cfg.actor_rollout_ref.actor.ppo_micro_batch_size=16
 cfg.trainer.experiment_name='sevenb-original-main16-v1';cfg.trainer.default_local_dir=plan['root']+'/checkpoints'
 cfg.baselines.local_runtime_host=host;cfg.retriever.url='http://'+ip+':'+str(plan['proxy_port'])+'/retrieve'
 cfg.baselines.holdout.output_dir=plan['data_root']
 projected_config(cfg);return cfg

def main(pp):
 p=read_plan(pp);root=Path(p['root']);entries,_=verify_node_records(p);head=entries[0];assert socket.gethostname()==head['hostname']
 for k in list(os.environ):
  if k.startswith(('TREE_RECOVERY_','TREE_CURRENT_CHAIN_RESUME_','RECURSIVE_A_CONTINUATION_','RECURSIVE_ABLATION_')):raise ValueError('Fresh16 must not inherit recovery environment '+k)
 from omegaconf import OmegaConf
 cfg=configuration(p,head['hostname'].split('.')[0],head['ip']);text=OmegaConf.to_yaml(cfg,resolve=True);(root/'resolved.yaml').open('x').write(text)
 out=root/'output';out.mkdir()
 env=dict(SEARCH_R1_EXECUTION_BACKEND='pj_pt',SEARCH_R1_JOB_ID=p['job'],SEARCH_R1_PJ_NODE=head['hostname'].split('.')[0],
  SEARCH_R1_BASELINE_RUN_ID=p['run_id'],E1_RUN_ID=p['run_id'],SEARCH_R1_BASELINE_COMMIT=p['parent_science_source'],
  SEARCH_R1_BASELINE_CONFIG_SHA256=sha(root/'resolved.yaml'),SEARCH_R1_BASELINE_DATA_IDENTITY=p['data_identity'],
  SEARCH_R1_MODEL_IDENTITY=p['model_identity'],SEARCH_R1_BASELINE_OUTPUT=str(out),SEARCH_R1_BASELINE_AUDIT_DIR=str(out/'audit'),
  SEARCH_R1_TREE_RANK_AUDIT_DIR=str(out/'rank-audit'),SEARCH_R1_TRAJECTORY_DIR=str(out/'trajectories'),TREE_COMMON_DATA=p['data_root'],TREE_INSTRUCT_MODEL=p['model'],
  RECURSIVE_A_PJ_FRESH_PROFILE='1',RECURSIVE_A_OMIT_FULL_BETA='1',RECURSIVE_A_GUARD_ROUND_PROGRESS='1',TREE_LIGHTWEIGHT_CANDIDATE_REWARD='1',RECURSIVE_A_SPARSE_DETAILS='1',RECURSIVE_LUMIA_PAIRED_EVALUATION='1',
  CGRTS16_PARALLEL_PLAN=str(pp),PYTHONPATH=p['source'],TRITON_PTXAS_PATH=p['compiler']['path'],TOKENIZERS_PARALLELISM='false',NCCL_DEBUG='WARN')
 os.environ.update(env);os.environ['RAY_ADDRESS']=head['ip']+':'+str(p['ray_port'])
 from .bootstrap import install,worker_setup
 install()
 from http.server import ThreadingHTTPServer
 from reproduction.recursive_lumia.proxy import Handler
 server=ThreadingHTTPServer((head['ip'],p['proxy_port']),Handler);server.remote=p['retriever_url'];threading.Thread(target=server.serve_forever,daemon=True).start()
 from release.recursive_bran_a.h100_migration.ray_cli_compat import prepare_click
 prepare_click();import ray
 inherited={k:v for k,v in os.environ.items() if k.startswith(('SEARCH_R1_','RECURSIVE_A_','RECURSIVE_LUMIA_','TREE_','CGRTS16_'))}
 inherited.update({k:env[k] for k in ('PYTHONPATH','TRITON_PTXAS_PATH','TOKENIZERS_PARALLELISM','NCCL_DEBUG')})
 error=None
 try:
  ray.init(address=os.environ['RAY_ADDRESS'],namespace=p['job'],runtime_env={'env_vars':inherited,'worker_process_setup_hook':worker_setup})
  deadline=time.monotonic()+240
  while len([n for n in ray.nodes() if n.get('Alive')])!=2:
   assert time.monotonic()<deadline,'Ray did not admit both allocated nodes';time.sleep(2)
  validate_cluster_boundary(cfg,ray.nodes());save(root/'cluster16-admission.json',dict(at=time.time(),nodes=ray.nodes(),layout='2x8',original_science_source=p['parent_science_source'],entry_source=p['source_commit']))
  from search_r1.baselines.main import main
  main.__wrapped__(cfg)
  save(root/'training-returned.json',dict(at=time.time(),job=p['job']))
 except BaseException as e:error=repr(e);raise
 finally:
  ray.shutdown();server.shutdown();server.server_close();save(root/'entry-exit.json',dict(at=time.time(),rc=int(error is not None),error=error))
if __name__=='__main__':main(sys.argv[1])
