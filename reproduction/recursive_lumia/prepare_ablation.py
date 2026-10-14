"""Bind ablations to our actual original-A run, without claiming author bytes.

This prepares configurations only. Submission and runtime identity checks are
separate; the H100-only upstream entry point must not launch these Lumia runs.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import uuid

from release.recursive_bran_a.ablation150_v1 import EVALUATION_STEPS, PROTOCOL
from release.recursive_bran_a.ablation150_v1.contracts import (
    planned_conditions, projected_original, sha, validate_config_dict,
)

BASELINE_CONFIG_SHA = '274b6801a78591ca8b28dadcbe68dd0ab46ff5c7b0e4e6be4fc34a82964ceaa0'
BASELINE_RUN_ID = '1fec29ed0ce64a4f989565ee73e1db95'
BASELINE_SOURCE_SHA = '2f85d77d387887555b436f90e29c43512069e9c3'
DATA_SHA = '7b6369e17ebd497ea159d57dc159647eed8adad852b7468cc600b7f203e80f43'
HOSTS = {'gpu-l40s-1', 'gpu-l40s-2', 'gpu-ada6000-2', 'gpu-4090-3'}


def condition_for(group):
    rows = [r for r in planned_conditions() if r['group'] == group]
    if len(rows) != 1:
        raise ValueError('Unknown published study condition')
    if rows[0].get('reuse'):
        raise ValueError('This condition reuses an existing control; do not train it again')
    return rows[0]['options']


def derive(template, group, base, runtime_host):
    condition = condition_for(group)
    if runtime_host not in HOSTS:
        raise ValueError('This preparation supports only the verified Lumia host mapping')
    if template['baselines'].get('question_curriculum', {}).get('enabled'):
        raise ValueError('Mechanism controls must start from original A, not curriculum')
    config = copy.deepcopy(template)
    trainer = config['trainer']
    trainer.update(actual_update_limit=150, total_training_steps=151,
        evaluation_steps=list(EVALUATION_STEPS), checkpoint_save_steps=list(EVALUATION_STEPS),
        val_before_train=False, acceptance_gate_enabled=False,
        experiment_name='recursive-lumia-ablation-'+group,
        default_local_dir=str(Path(base)/'checkpoints'))
    config['baselines']['local_runtime_host'] = runtime_host
    config['baselines']['recursive_ablation'] = condition
    config['baselines']['tree']['continuation_budget_per_question'] = condition['budget']
    # The schedule remains on the original200 scale despite stopping at150.
    config['actor_rollout_ref']['actor']['optim']['total_training_steps'] = 200
    config['critic']['optim']['total_training_steps'] = 201
    validate_config_dict(config)
    return config


def validate_delta(template, candidate, group, base, runtime_host):
    if candidate != derive(template, group, base, runtime_host):
        raise ValueError('Unrequested difference from the actual original-A configuration')


def prepare(baseline_receipt, base, source_commit, group, runtime_host):
    from omegaconf import OmegaConf
    from search_r1.baselines.controller import validate_config
    baseline_receipt, base = Path(baseline_receipt).resolve(), Path(base).resolve()
    baseline = json.loads(baseline_receipt.read_text())
    if (baseline['arm'] != 'original' or baseline['run_id'] != BASELINE_RUN_ID
            or baseline['config_sha256'] != BASELINE_CONFIG_SHA
            or sha(baseline['config']) != BASELINE_CONFIG_SHA
            or baseline['source_commit'] != BASELINE_SOURCE_SHA
            or baseline['data_identity'] != DATA_SHA
            or baseline['execution_backend'] != 'slurm_lumia'):
        raise ValueError('Use the actual task-owned original-A configuration and receipt')
    if len(source_commit) != 40 or any(c not in '0123456789abcdef' for c in source_commit):
        raise ValueError('A published full source SHA is required')
    condition = condition_for(group)
    template = OmegaConf.to_container(OmegaConf.load(baseline['config']), resolve=True)
    if (baseline['model'] != template['actor_rollout_ref']['model']['path']
            or baseline['data'] != template['baselines']['holdout']['data_root']
            or sha(Path(baseline['data'])/'manifest.json') != DATA_SHA):
        raise ValueError('Actual model/data paths differ from the bound original run')
    if sha(baseline['retriever_receipt']) != baseline['retriever_receipt_sha256']:
        raise ValueError('Original pinned retrieval service receipt changed')
    validate_config(OmegaConf.create(template))
    candidate = derive(template, group, base, runtime_host)
    validate_delta(template, candidate, group, base, runtime_host)
    validate_config(OmegaConf.create(projected_original(candidate)))
    text = OmegaConf.to_yaml(OmegaConf.create(candidate))
    base.mkdir(parents=True, exist_ok=False)
    for name in ('receipts', 'output', 'logs', 'tmp', 'cache', 'checkpoints'):
        (base/name).mkdir()
    config_path = base/'receipts/resolved.yaml'
    config_path.write_text(text)
    record = dict(protocol=PROTOCOL, group=group, run_id=uuid.uuid4().hex,
        config=str(config_path), config_sha256=hashlib.sha256(text.encode()).hexdigest(),
        baseline_receipt=str(baseline_receipt), baseline_receipt_sha256=sha(baseline_receipt),
        template=baseline['config'], template_sha256=BASELINE_CONFIG_SHA,
        baseline_run_id=BASELINE_RUN_ID, baseline_source_commit=baseline['source_commit'],
        source_commit=source_commit, model=baseline['model'], data=baseline['data'],
        retriever_receipt=baseline['retriever_receipt'],
        retriever_receipt_sha256=baseline['retriever_receipt_sha256'],
        retriever_url=baseline['retriever_url'],
        data_identity=DATA_SHA, base=str(base), output=str(base/'output'),
        fresh_authorized=True, parent_checkpoint=None, execution_backend='slurm_lumia',
        host=runtime_host, condition=condition, actual_update_limit=150,
        evaluation_steps=list(EVALUATION_STEPS), scheduler_scale=200, warmup_rounds=57,
        original_recipe_bytes_available=False,
        reproduction_limit='Controls matched to this task original-A reconstruction, not author43e28bytes.',
        training_submitted=False)
    (base/'receipts/configuration.json').write_text(json.dumps(record, indent=2)+'\n')
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-receipt', required=True)
    parser.add_argument('--base', required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--group', required=True)
    parser.add_argument('--runtime-host', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.baseline_receipt, args.base, args.source_commit,
                             args.group, args.runtime_host)), flush=True)


if __name__ == '__main__':
    main()
