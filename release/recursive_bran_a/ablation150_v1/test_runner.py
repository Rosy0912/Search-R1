"""Focused CPU integration contracts; no model, HTTP, or production mutation."""
import unittest

from .contracts import options, planned_conditions, projected_original, validate_config_dict
from . import EVALUATION_STEPS


def minimal_config(condition=None):
    condition = options('root_only') if condition is None else condition
    return dict(baselines=dict(recursive_ablation=condition, tree=dict(continuation_budget_per_question=condition['budget'])),
        trainer=dict(actual_update_limit=150, total_training_steps=151, evaluation_steps=list(EVALUATION_STEPS),
            checkpoint_save_steps=list(EVALUATION_STEPS), val_before_train=False, acceptance_gate_enabled=False,
            n_gpus_per_node=8, nnodes=1), actor_rollout_ref=dict(actor=dict(optim=dict(total_training_steps=200))),
        critic=dict(optim=dict(total_training_steps=201)))


class RunnerContracts(unittest.TestCase):
    def test_schedule_stops150_but_preserves_scheduler200(self):
        config = minimal_config()
        validate_config_dict(config)
        projected = projected_original(config)
        self.assertEqual(config['trainer']['actual_update_limit'], 150)
        self.assertEqual(projected['trainer']['actual_update_limit'], 200)
        self.assertEqual(config['actor_rollout_ref']['actor']['optim']['total_training_steps'], 200)

    def test_rejects_additional_evaluation(self):
        config = minimal_config()
        config['trainer']['evaluation_steps'].append(10)
        with self.assertRaises(ValueError):
            validate_config_dict(config)

    def test_same_factor_only(self):
        with self.assertRaises(ValueError):
            options('root_only', budget=5)
        with self.assertRaises(ValueError):
            options(budget=5, position='early')

    def test_requested_grid_uses_existing_matched_controls(self):
        rows = planned_conditions()
        self.assertEqual([r['group'] for r in rows[:2]], ['C1-root-only', 'C2-random-final'])
        self.assertEqual(len(rows), 11)
        self.assertEqual(len([r for r in rows if not r.get('reuse')]), 7)
        self.assertEqual(next(r for r in rows if r['group'] == 'generation-1')['reuse'], 'C1-root-only')
        self.assertEqual(next(r for r in rows if r['group'] == 'generation-4')['reuse'], 'A-original')

    def test_bootstrap_accepts_complete_frozen_condition(self):
        from unittest.mock import patch
        from . import bootstrap
        condition = options('root_only')
        with patch.object(bootstrap, 'active', return_value=True), \
                patch.object(bootstrap, 'receipt', return_value=dict(condition=condition)), \
                patch('search_r1.baselines.recursive_ablation.install') as install:
            bootstrap._rollout(None)
            self.assertEqual(install.call_args.args[0], dict(variant='root_only', position='uniform',
                                                          max_generation=None, selection_seed=42))


if __name__ == '__main__':
    unittest.main()
