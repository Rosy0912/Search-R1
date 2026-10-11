"""Freeze the fully resolved HKUST contract before allocating model GPUs."""
import argparse,json,hashlib,os,socket,uuid
from pathlib import Path
from hydra import compose,initialize_config_dir
from omegaconf import OmegaConf
from search_r1.baselines.controller import validate_config
from search_r1.baselines.holdout import prepare_common_holdout
p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);a=p.parse_args();b=a.base.resolve();repo=Path(__file__).parents[2]
data=b/'common-data-v2';model=Path('/data/shaoshitong/search-e1-config-study-r1-aligned-8gpu-20261005/assets/model')
os.environ.update(TREE_COMMON_DATA=str(data),TREE_INSTRUCT_MODEL=str(model))
overrides=json.loads((repo/'release/bran_tree_grpo/production_overrides.json').read_text())
replacements={'baselines.local_runtime_host':socket.gethostname().split('.')[0],
 'baselines.tree.roots':'1','baselines.tree.leaves':'5','baselines.tree.expansions':'4',
 'baselines.tree.leaf_quotas':'[5]','actor_rollout_ref.rollout.n_agent':'1',
 'baselines.tree.branch_sampling':'recursive_uniform_v1','baselines.tree.continuation_budget_per_question':'10',
 'actor_rollout_ref.model.path':str(model),'actor_rollout_ref.ref.model_path':str(model),
 'data.train_files':str(data/'train.parquet'),'data.val_files':str(data/'validation.parquet'),
 'data.final_val_files':str(data/'final.parquet'),'baselines.holdout.manifest_path':str(data/'manifest.json'),
 'baselines.holdout.manifest_sha256':hashlib.sha256((data/'manifest.json').read_bytes()).hexdigest(),
 'baselines.holdout.data_root':str(data),'trainer.acceptance_gate_enabled':'false',
 'trainer.experiment_name':'recursive-bran-a-hkust1-200','retriever.url':'http://10.120.17.101:18004/retrieve',
 'trainer.default_local_dir':str(b/'checkpoints')}
overrides=[k+'='+str(replacements.get(k,v)) for k,v in [x.split('=',1) for x in overrides]]
with initialize_config_dir(config_dir=str(repo/'verl/trainer/config'),version_base=None):
 cfg=compose(config_name='ppo_recursive_bran_a',overrides=overrides)
validate_config(cfg);identity=prepare_common_holdout(cfg)
text=OmegaConf.to_yaml(OmegaConf.create(OmegaConf.to_container(cfg,resolve=True)))
r=b/'receipts';r.mkdir(parents=True,exist_ok=True);(r/'resolved.yaml').write_text(text)
record=dict(run_id=uuid.uuid4().hex,config_sha256=hashlib.sha256(text.encode()).hexdigest(),data_identity=identity['content_identity']['sha256'],
 model=str(model),execution_backend='hkust_direct',config=str(r/'resolved.yaml'),overrides=overrides)
(r/'configuration.json').write_text(json.dumps(record,indent=2));print(json.dumps({k:v for k,v in record.items() if k!='overrides'}))
