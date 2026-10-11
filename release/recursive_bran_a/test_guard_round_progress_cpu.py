"""Tiny actual zero-write/guard accounting contracts, no distributed GPU claim."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace, ModuleType
from unittest.mock import patch
import sys
import os
import unittest
import torch
from search_r1.baselines.tree_guard_continue import (
    PROTOCOL, normal_guard_summary, round_progress_enabled, validate_ledger)
_root = Path(__file__).resolve().parents[2]
_refusal_node = next(x for x in ast.parse((_root / 'search_r1/baselines/policy_safety_runtime.py').read_text()).body
    if isinstance(x, ast.ClassDef) and x.name == 'PolicySafetyRefusal')
_refusal_namespace = {}
exec(compile(ast.fix_missing_locations(ast.Module(body=[_refusal_node], type_ignores=[])),
             'actual_PolicySafetyRefusal', 'exec'), _refusal_namespace)
PolicySafetyRefusal = _refusal_namespace['PolicySafetyRefusal']


def ledger(writes):
    return [dict(index=i+1, rows=16, local_row_indices=list(range(i*16, (i+1)*16)),
        actual_writes=int(i < writes), status=('SUCCESS_ADAM_WRITE' if i < writes else
            'SKIPPED_SAFETY_GUARD_BEFORE_WRITE' if i == writes else 'SKIPPED_BY_PRIOR_ROLLOUT_BRAKE'))
        for i in range(20)]


def actual_compact_helper():
    path = Path(__file__).resolve().parents[2] / 'verl/workers/fsdp_workers.py'
    fn = next(x for x in ast.parse(path.read_text()).body
              if isinstance(x, ast.FunctionDef) and x.name == 'compact_normal_guard_after_actor')
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    namespace = dict(torch=torch)
    exec(compile(module, str(path), 'exec'), namespace)
    actual = namespace[fn.name]
    def call(*args):
        dependency = ModuleType('search_r1.baselines.policy_safety_runtime')
        dependency.PolicySafetyRefusal = PolicySafetyRefusal
        with patch.dict(sys.modules, {'search_r1.baselines.policy_safety_runtime': dependency}):
            return actual(*args)
    return call


class Checks(unittest.TestCase):
    def test_explicit_opt_in_required(self):
        actor = dict(guard_continue_protocol=PROTOCOL)
        env = dict(RECURSIVE_A_GUARD_ROUND_PROGRESS='1', RECURSIVE_A_CONTINUATION_BINDING='/bound')
        self.assertTrue(round_progress_enabled(actor, env))
        self.assertFalse(round_progress_enabled(actor, {}))
        self.assertFalse(round_progress_enabled({}, env))
        self.assertFalse(round_progress_enabled(actor, dict(RECURSIVE_A_GUARD_ROUND_PROGRESS='1')))

    def test_zero_and_partial_write_ledgers_are_truthful(self):
        refusal = PolicySafetyRefusal('native threshold', matched_all_ranks=True, safety_guard=True)
        for writes in (0, 1, 7, 20):
            result = normal_guard_summary(refusal, writes, ledger(writes))
            self.assertEqual(result['optimizer_writes'], writes)
            self.assertEqual(result['opportunities_processed'], 20)
            self.assertEqual(result['opportunities_skipped'], 20-writes)
            self.assertEqual(result['successful_update_claim'], writes > 0)
            self.assertEqual(result['zero_write_round'], writes == 0)
            self.assertFalse(result['model_optimizer_snapshot_saved'])

    def test_fault_and_incomplete_ledger_are_not_normal_skips(self):
        for refusal in (RuntimeError('programming error'),
                PolicySafetyRefusal('code fault', matched_all_ranks=True, safety_guard=False),
                PolicySafetyRefusal('unmatched threshold', matched_all_ranks=False, safety_guard=True)):
            with self.assertRaises(ValueError):
                normal_guard_summary(refusal, 0, ledger(0))
        refusal = PolicySafetyRefusal('threshold', matched_all_ranks=True, safety_guard=True)
        with self.assertRaises(ValueError):
            normal_guard_summary(refusal, 0, ledger(0)[:-1])
        bad = ledger(0); bad[7]['actual_writes'] = 1
        with self.assertRaises(ValueError):
            normal_guard_summary(refusal, 0, bad)

    def test_zero_write_round_validation_requires_contract_and_complete_ledger(self):
        from search_r1.baselines.tree_fresh200 import validate_completed_outer
        metrics = {'actor/optimizer_opportunities_processed': 20, 'actor/optimizer_steps': 0,
            'actor/optimizer_opportunities_skipped': 20, 'actor/scheduler_step_before_outer': 4,
            'actor/scheduler_step_after_outer': 5, 'actor/guard_round_progress_enabled': 1,
            'actor/zero_write_sampling_round': 1}
        with patch.dict(os.environ, RECURSIVE_A_GUARD_ROUND_PROGRESS='0'):
            with self.assertRaises(RuntimeError): validate_completed_outer(metrics, 5)
        with patch.dict(os.environ, RECURSIVE_A_GUARD_ROUND_PROGRESS='1',
                        RECURSIVE_A_CONTINUATION_BINDING='/bound'):
            validate_completed_outer(metrics, 5)
            bad = dict(metrics, **{'actor/optimizer_opportunities_processed': 19})
            with self.assertRaises(RuntimeError): validate_completed_outer(bad, 5)

    def test_old_accounting_schema_while_parent_verified_then_new_explicit_schema(self):
        from search_r1.baselines.tree_guard_continue import accounting
        trainer = SimpleNamespace(actual_outer_updates=4,
            actor_rollout_wg=SimpleNamespace(world_size=1),
            config=SimpleNamespace(actor_rollout_ref=SimpleNamespace(actor=dict(guard_continue_protocol=PROTOCOL))))
        with patch.dict(os.environ, RECURSIVE_A_GUARD_ROUND_PROGRESS='1',
                        RECURSIVE_A_CONTINUATION_BINDING='/bound'):
            trainer._guard_round_contract_active = False
            self.assertNotIn('round_progress_contract', accounting(trainer))
            trainer._guard_round_contract_active = True
            new = accounting(trainer)
            self.assertEqual(new['consumed_sampling_rounds'], 4)
            self.assertEqual(new['zero_write_sampling_rounds'], 0)
            self.assertEqual(new['optimizer_write_rounds'], 4)

    def test_actual_driver_restore_preserves_new_zero_counts_and_keeps_old_minimum(self):
        from search_r1.baselines import training_state as ts
        state = dict(protocol=PROTOCOL, training_attempts=200, refused_attempts=0,
            guard_refused_attempts=0, completed_outer_updates=200,
            partial_optimizer_calls_per_rank={}, completed_optimizer_calls_per_rank={'0': 20},
            total_optimizer_calls_per_rank={'0':20}, partial_evidence=[],
            round_progress_contract='consumed_sampling_rounds_separate_adam_v1',
            consumed_sampling_rounds=200, zero_write_sampling_rounds=199,
            optimizer_write_rounds=1, normal_guard_rounds=199)
        identity = dict(guard_continue_accounting=state)
        sampler = SimpleNamespace(load_state_dict=lambda value: None)
        controller = SimpleNamespace(method='tree_grpo', stage=SimpleNamespace(load_state_dict=lambda value: None), tree_rng=None)
        trainer = SimpleNamespace(baseline_controller=controller, actor_rollout_wg=SimpleNamespace(world_size=1), checkpoint_sampler=sampler)
        saved = dict(version=1, identity=identity, completed_step=200, actual_outer_updates=200,
            next_step=201, guard_continue_accounting=state,
            sampler=dict(rows=512*256, batch_size=512, epoch=0, cursor=512*200),
            algorithm=dict(method='tree_grpo', stage={}, tree_rng=None), rng={})
        with patch.object(ts, 'restore_rng', lambda value: None):
            ts.restore_driver(trainer, saved, identity)
        self.assertEqual(trainer.zero_write_sampling_rounds, 199)
        self.assertEqual(trainer.normal_guard_rounds, 199)
        self.assertEqual(trainer.completed_optimizer_calls_per_rank, {'0':20})
        old = copy.deepcopy(saved)
        for key in ('round_progress_contract', 'consumed_sampling_rounds',
                    'zero_write_sampling_rounds', 'optimizer_write_rounds', 'normal_guard_rounds'):
            old['guard_continue_accounting'].pop(key)
        # Identity is copied with the same aliased state in saved.
        with patch.object(ts, 'restore_rng', lambda value: None):
            with self.assertRaisesRegex(ValueError, 'actual optimizer writes'):
                ts.restore_driver(trainer, old, old['identity'])

    def test_actual_compact_helper_clears_grad_without_model_adam_write(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model(torch.ones(1,2)).sum().backward(); optimizer.step(); optimizer.zero_grad()
        before_model = copy.deepcopy(model.state_dict())
        before_adam = copy.deepcopy(optimizer.state_dict())
        model(torch.ones(1,2)).sum().backward()
        checks = []
        def finite_check(phase):
            checks.append(phase)
            self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
        actor = SimpleNamespace(actor_module=model, _actual_committed_optimizer_calls=0,
            _fresh_opportunity_ledger=ledger(0),
            _policy_safety=SimpleNamespace(parameters_and_adam=finite_check))
        worker = SimpleNamespace(actor=actor, actor_optimizer=optimizer)
        result = actual_compact_helper()(worker,
            PolicySafetyRefusal('threshold', matched_all_ranks=True, safety_guard=True))
        self.assertEqual(len(checks), 1)
        self.assertEqual(result['ranks'][0]['optimizer_writes'], 0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before_model[key]))
        after_adam = optimizer.state_dict()
        self.assertEqual(before_adam['param_groups'], after_adam['param_groups'])
        for key, state in before_adam['state'].items():
            for name, value in state.items():
                self.assertTrue(torch.equal(value, after_adam['state'][key][name]))


if __name__ == '__main__':
    unittest.main()
