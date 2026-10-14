"""Freeze fresh official-model configurations from the actual completed A recipe."""
import argparse
import hashlib
import json
from pathlib import Path
import uuid

from . import EVALUATION_STEPS, ORIGINAL_CONFIG_SHA256, PROTOCOL
from .contracts import options, sha, validate_config_dict


def prepare(template, base, source_commit, group, condition, control_token_file,
            expected_template_sha=ORIGINAL_CONFIG_SHA256):
    from omegaconf import OmegaConf
    template, base = Path(template).resolve(), Path(base).resolve()
    if sha(template) != expected_template_sha or expected_template_sha != ORIGINAL_CONFIG_SHA256:
        raise ValueError('Use the actual A resolved 43e28 recipe, not a similar named branch')
    if len(source_commit) != 40 or any(x not in '0123456789abcdef' for x in source_commit):
        raise ValueError('A published source commit is required')
    receipt = base / 'receipts' / 'configuration.json'
    if receipt.exists():
        raise ValueError('Already prepared; existing fresh state must not be overwritten')
    cfg = OmegaConf.load(template)
    OmegaConf.set_struct(cfg, False)
    cfg.trainer.actual_update_limit = 150
    cfg.trainer.total_training_steps = 151
    cfg.trainer.evaluation_steps = list(EVALUATION_STEPS)
    cfg.trainer.checkpoint_save_steps = list(EVALUATION_STEPS)
    cfg.trainer.val_before_train = False
    cfg.trainer.acceptance_gate_enabled = False
    cfg.trainer.experiment_name = 'recursive-ablation150-' + group
    cfg.trainer.default_local_dir = str(base / 'checkpoints')
    # Preserve actor warmup57/scale200; dataloader initialization is wrapped too.
    cfg.actor_rollout_ref.actor.optim.total_training_steps = 200
    cfg.critic.optim.total_training_steps = 201
    cfg.baselines.recursive_ablation = condition
    cfg.baselines.tree.continuation_budget_per_question = condition['budget']
    cfg.retriever.control_token_file = str(Path(control_token_file).resolve())
    resolved = OmegaConf.to_container(cfg, resolve=True)
    validate_config_dict(resolved)
    text = OmegaConf.to_yaml(OmegaConf.create(resolved))
    for folder in ('receipts', 'logs', 'tmp', 'cache', 'output'):
        (base / folder).mkdir(parents=True, exist_ok=True)
    config_path = base / 'receipts' / 'resolved.yaml'
    with config_path.open('x') as handle:
        handle.write(text)
    data = resolved['baselines']['holdout']['data_root']
    record = dict(protocol=PROTOCOL, group=group, run_id=uuid.uuid4().hex,
        config=str(config_path), config_sha256=hashlib.sha256(text.encode()).hexdigest(),
        template=str(template), template_sha256=expected_template_sha,
        source_commit=source_commit, model=resolved['actor_rollout_ref']['model']['path'],
        data=data, data_identity=resolved['baselines']['holdout']['manifest_sha256'],
        base=str(base), output=str(base / 'output'), fresh_authorized=True,
        parent_checkpoint=None, execution_backend='slurm_h100', condition=condition,
        actual_update_limit=150, evaluation_steps=list(EVALUATION_STEPS),
        scheduler_scale=200, warmup_rounds=57)
    with receipt.open('x') as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', required=True)
    parser.add_argument('--base', required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--group', required=True)
    parser.add_argument('--control-token-file', required=True)
    parser.add_argument('--variant', choices=('baseline', 'root_only', 'random_final'), default='baseline')
    parser.add_argument('--budget', type=int, default=10)
    parser.add_argument('--position', choices=('early', 'uniform', 'late'), default='uniform')
    parser.add_argument('--max-generation', type=int)
    args = parser.parse_args()
    result = prepare(args.template, args.base, args.source_commit, args.group,
        options(args.variant, args.budget, args.position, args.max_generation), args.control_token_file)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
