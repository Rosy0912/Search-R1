"""Fresh official-weight experiment A on the authorized H100 allocation; reuse local assets."""
import hashlib,json,os,socket,subprocess,uuid
from pathlib import Path
from hydra import compose,initialize_config_dir
from omegaconf import OmegaConf
from search_r1.baselines.controller import validate_config
from search_r1.baselines.holdout import prepare_common_holdout
base=Path(os.environ['RECURSIVE_BRAN_BASE']);repo=Path(__file__).resolve().parents[3]
bapo=Path('/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v11-20261009')
original=OmegaConf.load(next((bapo/'runs').glob('formal-aligned-em200-*/resolved-config.yaml')))
model=original.actor_rollout_ref.model.path;data=Path(original.baselines.holdout.data_root)
assert Path(model,'config.json').is_file() and (data/'manifest.json').is_file()
os.environ.update(TREE_COMMON_DATA=str(data),TREE_INSTRUCT_MODEL=str(model),SEARCH_R1_BASELINE_CHECKPOINTS=str(base/'checkpoints'))
replacements={'baselines.local_runtime_host':'h100-11','baselines.tree.roots':'1','baselines.tree.leaves':'5','baselines.tree.expansions':'4','baselines.tree.leaf_quotas':'[5]','actor_rollout_ref.rollout.n_agent':'1','baselines.tree.branch_sampling':'recursive_uniform_v1','baselines.tree.continuation_budget_per_question':'10','actor_rollout_ref.model.path':model,'actor_rollout_ref.ref.model_path':model,'data.train_files':str(data/'train.parquet'),'data.val_files':str(data/'validation.parquet'),'data.final_val_files':str(data/'final.parquet'),'baselines.holdout.manifest_path':str(data/'manifest.json'),'baselines.holdout.manifest_sha256':hashlib.sha256((data/'manifest.json').read_bytes()).hexdigest(),'baselines.holdout.data_root':str(data),'trainer.acceptance_gate_enabled':'false','trainer.experiment_name':'recursive-bran-a-h100-fresh200','retriever.url':'http://127.0.0.1:18086/retrieve','trainer.default_local_dir':str(base/'checkpoints')}
overrides=[k+'='+str(replacements.get(k,v)) for k,v in [x.split('=',1) for x in json.loads((repo/'release/bran_tree_grpo/production_overrides.json').read_text())]]
with initialize_config_dir(config_dir=str(repo/'verl/trainer/config'),version_base=None):cfg=compose(config_name='ppo_recursive_bran_a',overrides=overrides)
cfg.actor_rollout_ref.rollout.tree_action_tag_stop=True;cfg.actor_rollout_ref.rollout.tree_action_stop_mode='search_only'
cfg.retriever.gpu_release_url='http://127.0.0.1:18086/release-gpu';cfg.retriever.control_token_file=str(base/'retriever/receipts/private-probe-token')
validate_config(cfg);identity=prepare_common_holdout(cfg)
text=OmegaConf.to_yaml(OmegaConf.create(OmegaConf.to_container(cfg,resolve=True)))
(base/'receipts').mkdir(exist_ok=True);path=base/'receipts/resolved.yaml';path.write_text(text)
r=dict(run_id=uuid.uuid4().hex,config_sha256=hashlib.sha256(text.encode()).hexdigest(),data_identity=identity['content_identity']['sha256'],model=model,data=str(data),execution_backend='slurm_h100',config=str(path),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),fresh_authorized=True,parent_checkpoint=None)
(base/'receipts/configuration.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
