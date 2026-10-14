"""Reconstruct the published recursive-A scientific configuration on Lumia."""
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import uuid

def prepare(base, arm):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from search_r1.baselines.controller import validate_config
    from search_r1.baselines.holdout import prepare_common_holdout
    base = Path(base).resolve()
    assert arm in ('original', 'curriculum')
    run = base / 'runs' / arm
    receipt = run/'receipts/configuration.json'
    if receipt.exists():
        raise RuntimeError('Existing immutable run; never cold-restart or overwrite')
    assets = json.loads((base/'receipts/assets.json').read_text())
    repo = Path(__file__).resolve().parents[2]
    model, data = assets['model'], Path(assets['data'])
    os.environ.update(TREE_COMMON_DATA=str(data), TREE_INSTRUCT_MODEL=model,
                      SEARCH_R1_BASELINE_CHECKPOINTS=str(run/'checkpoints'))
    replacements = {'baselines.local_runtime_host':socket.gethostname().split('.')[0],
        'baselines.tree.roots':'1', 'baselines.tree.leaves':'5', 'baselines.tree.expansions':'4',
        'baselines.tree.leaf_quotas':'[5]', 'actor_rollout_ref.rollout.n_agent':'1',
        'baselines.tree.branch_sampling':'recursive_uniform_v1', 'baselines.tree.continuation_budget_per_question':'10',
        'actor_rollout_ref.model.path':model, 'actor_rollout_ref.ref.model_path':model,
        'data.train_files':str(data/'train.parquet'), 'data.val_files':str(data/'validation.parquet'),
        'data.final_val_files':str(data/'final.parquet'), 'baselines.holdout.manifest_path':str(data/'manifest.json'),
        'baselines.holdout.manifest_sha256':assets['data_manifest_sha256'], 'baselines.holdout.data_root':str(data),
        'trainer.acceptance_gate_enabled':'false', 'trainer.experiment_name':'recursive-lumia-'+arm,
        'retriever.url':'http://127.0.0.1:29146/retrieve', 'trainer.default_local_dir':str(run/'checkpoints')}
    overrides = [k+'='+str(replacements.get(k,v)) for k,v in
        [x.split('=',1) for x in json.loads((repo/'release/bran_tree_grpo/production_overrides.json').read_text())]]
    with initialize_config_dir(config_dir=str(repo/'verl/trainer/config'), version_base=None):
        cfg = compose(config_name='ppo_recursive_bran_a', overrides=overrides)
    OmegaConf.set_struct(cfg, False)
    cfg.trainer.logger = ['console']
    cfg.actor_rollout_ref.rollout.gpu_memory_utilization = .5
    cfg.actor_rollout_ref.rollout.tree_action_tag_stop = True
    cfg.actor_rollout_ref.rollout.tree_action_stop_mode = 'search_only'
    if arm == 'curriculum':
        plan = json.loads((Path(__file__).parent/'plan.json').read_text())
        cfg.baselines.question_curriculum = dict(enabled=True, **plan['curriculum'])
        cfg.trainer.actual_update_limit = 150
        cfg.trainer.total_training_steps = 151
        cfg.trainer.evaluation_steps = [5,50,100,150]
        cfg.trainer.checkpoint_save_steps = [5,50,100,150]
    validate_config(cfg)
    identity = prepare_common_holdout(cfg)
    for name in ('receipts','output','checkpoints','logs','cache','tmp'):
        (run/name).mkdir(parents=True, exist_ok=True)
    text = OmegaConf.to_yaml(OmegaConf.create(OmegaConf.to_container(cfg, resolve=True)))
    config_path = run/'receipts/resolved.yaml'
    config_path.write_text(text)
    record = dict(run_id=uuid.uuid4().hex, arm=arm, config=str(config_path),
        config_sha256=hashlib.sha256(text.encode()).hexdigest(), data_identity=identity['content_identity']['sha256'],
        model=model, data=str(data), source_commit=os.environ['RECURSIVE_SOURCE_COMMIT'],
        execution_backend='slurm_lumia', host=socket.gethostname().split('.')[0],
        fresh_authorized=True, parent_checkpoint=None, original_recipe_sha256='43e28cc7ecc0d38a93324bf480aafb23a0c1ecdb979e9ea587d0358ae095233d',
        original_recipe_bytes_available=False,
        reproduction_limit='Reconstructed from published overrides; hardware/backend and environment paths differ from author H100 run.')
    receipt.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record), flush=True)

if __name__ == '__main__':
    prepare(sys.argv[1], sys.argv[2])
