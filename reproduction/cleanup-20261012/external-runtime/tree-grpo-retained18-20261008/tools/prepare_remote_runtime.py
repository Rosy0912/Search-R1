import importlib.util,json,os,pathlib,subprocess,time
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/tree-grpo');repo=root/'repo';logs=root/'logs';assets=root/'assets-root'
os.environ.update(SEARCH_R1_BASE=str(root),SEARCH_R1_ASSETS=str(assets),SEARCH_R1_REPO=str(repo),SEARCH_R1_PYTHON='/usr/bin/python3.10',BOOTSTRAP_OFFLINE='1',CUDA_VISIBLE_DEVICES='',SEARCH_R1_BASELINE_OUTPUT=str(root/'runs/protocol-preparation/output'))
os.environ['PATH']=str(root/'tools')+':'+os.environ['PATH']
status=logs/'wheelhouse-status.json';exitfile=logs/'wheelhouse-retry-20261003.exit'
print('WAITING_FOR_HASH_VERIFIED_WHEELHOUSE',flush=True)
while not status.exists() or not json.loads(status.read_text()).get('complete'):
 if exitfile.exists():raise RuntimeError('Dependency downloader finished without complete verified wheelhouse; inspect preserved failures')
 time.sleep(30)
print('BOOTSTRAP_OFFLINE',flush=True)
subprocess.run(['bash',str(repo/'reproduction/h20/bootstrap.sh')],check=True)
python=root/'env/bin/python'
spec=importlib.util.spec_from_file_location('download_assets',repo/'reproduction/h20/download_assets.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
manifest=json.loads((assets/'assets.json').read_text());data=[a for a in manifest if not a['path'].startswith(('wiki/','qwen2.5-3b/','e5-base-v2/'))]
print('WAITING_FOR_VERIFIED_SOURCE_DATA',flush=True)
while not all(helper.valid(assets/'assets'/a['path'],a) for a in data):
 if (logs/'direct-copy-retry-20261003.exit').exists():raise RuntimeError('Asset downloader finished while source data incomplete; inspect retained failures')
 time.sleep(30)
subprocess.run([str(python),str(repo/'reproduction/h20/prepare_paper_data.py')],check=True)
assert json.loads((assets/'data/paper-grpo/dataset-audit.json').read_text())['unique_eval_ids']==51713
code='''import json,pathlib,sys
sys.path.insert(0,%r)
from hydra import initialize_config_dir,compose
from omegaconf import OmegaConf
from search_r1.baselines.holdout import prepare_common_holdout
from search_r1.baselines.controller import validate_config
with initialize_config_dir(config_dir=%r,version_base=None):
 c=compose(config_name='ppo_baselines',overrides=['baselines.method=tree_grpo','actor_rollout_ref.rollout.n_agent=2','trainer.n_gpus_per_node=8','actor_rollout_ref.actor.ppo_micro_batch_size=8','retriever.url=http://127.0.0.1:18082/retrieve','baselines.holdout.output_dir=%s'])
 validate_config(c)
 m=prepare_common_holdout(c)
 pathlib.Path(%r).write_text(json.dumps(m,indent=2))
 pathlib.Path(%r).write_text(OmegaConf.to_yaml(c,resolve=True))
'''%(str(repo),str(repo/'verl/trainer/config'),str(root/'common-data'),str(logs/'split-manifest.json'),str(logs/'formal-protocol-resolved.yaml'))
subprocess.run([str(python),'-c',code],check=True)
(logs/'cpu-runtime-and-data.ready').write_text('Environment and source data/split verified. Full assets/retrieval/GPU acceptance still required.\n')
print('CPU_RUNTIME_AND_DATA_READY_NO_GPU_ACCEPTANCE',flush=True)
