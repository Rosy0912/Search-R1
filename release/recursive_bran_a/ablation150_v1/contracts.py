"""Small configuration/sequence contracts, usable without GPU dependencies."""
import copy
import hashlib
import json
from pathlib import Path

from . import EVALUATION_STEPS, PROTOCOL


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def options(variant='baseline', budget=10, position='uniform', max_generation=None):
    if variant not in ('baseline', 'root_only', 'random_final', 'root_anchored_recursive'):
        raise ValueError('Unknown recursive ablation variant')
    if budget not in (5, 10, 20) or position not in ('early', 'uniform', 'late'):
        raise ValueError('Unspecified budget or branch-position condition')
    if max_generation not in (None, 1, 2, 4):
        raise ValueError('Unspecified recursive generation condition')
    if variant != 'baseline' and (budget, position, max_generation) != (10, 'uniform', None):
        raise ValueError('Mechanism controls cannot silently combine hyperparameters')
    if variant == 'baseline' and sum((budget != 10, position != 'uniform', max_generation is not None)) > 1:
        raise ValueError('Hyperparameter controls must change one factor')
    result = dict(protocol=PROTOCOL, variant=variant, budget=budget, position=position,
                  max_generation=max_generation, selection_seed=42,
                  position_probability='linear_positive_weights_over_legal_preaction_indices')
    if variant == 'root_anchored_recursive':
        result.update(acceptance_reference='initial_root',
                      branch_boundary='all_preaction_snapshots',
                      parent_sampling='uniform_accepted')
    return result


def projected_original(config):
    """Only explicit schedule/ablation fields are projected for native A validation."""
    expected = copy.deepcopy(config)
    expected['baselines'].pop('recursive_ablation', None)
    expected['baselines']['tree']['continuation_budget_per_question'] = 10
    expected['trainer']['actual_update_limit'] = 200
    expected['trainer']['total_training_steps'] = 201
    expected['actor_rollout_ref']['actor']['optim']['total_training_steps'] = 200
    expected['critic']['optim']['total_training_steps'] = 201
    return expected


def validate_config_dict(config):
    variant = config['baselines']['recursive_ablation']
    if variant != options(variant['variant'], variant['budget'], variant['position'], variant['max_generation']):
        raise ValueError('Ablation condition contract changed')
    trainer = config['trainer']
    if (trainer['actual_update_limit'], trainer['total_training_steps']) != (150, 151):
        raise ValueError('Ablation budget must be 150 consumed complete training rounds')
    if list(trainer['evaluation_steps']) != list(EVALUATION_STEPS) or list(trainer['checkpoint_save_steps']) != list(EVALUATION_STEPS):
        raise ValueError('Only the four authorized evaluation/keypoint boundaries are allowed')
    if trainer.get('val_before_train') or trainer.get('acceptance_gate_enabled'):
        raise ValueError('No unrequested initial test or blocking five-round gate')
    if config['baselines']['tree']['continuation_budget_per_question'] != variant['budget']:
        raise ValueError('Planner budget and frozen config disagree')
    if config['actor_rollout_ref']['actor']['optim']['total_training_steps'] not in (150, 200):
        raise ValueError('Unexpected scheduler scale before the initialization wrapper')
    if trainer['n_gpus_per_node'] != 8 or trainer['nnodes'] != 1:
        raise ValueError('This publication is the authorized single-node eight-GPU control')
    return variant


def read_receipt(path):
    path = Path(path).resolve()
    record = json.loads(path.read_text())
    if record['protocol'] != PROTOCOL or sha(record['config']) != record['config_sha256']:
        raise ValueError('Immutable ablation configuration changed')
    return record


def planned_conditions():
    """Eleven requested labels with matched A/C1 reuse, not repeated trainings."""
    rows = [dict(group='C1-root-only', options=options('root_only')),
            dict(group='C2-random-final', options=options('random_final'))]
    for budget in (5, 10, 20):
        rows.append(dict(group=f'budget-{budget}', options=options(budget=budget),
                         reuse='A-original' if budget == 10 else None))
    for position in ('early', 'uniform', 'late'):
        rows.append(dict(group=f'position-{position}', options=options(position=position),
                         reuse='A-original' if position == 'uniform' else None))
    for generation in (1, 2, 4):
        rows.append(dict(group=f'generation-{generation}', options=options(max_generation=generation),
                         reuse='C1-root-only' if generation == 1 else ('A-original' if generation == 4 else None)))
    # With root included and target5, every accepted child increments generation;
    # generation4 can only be accepted as the fifth record, ending the original tree.
    return rows
