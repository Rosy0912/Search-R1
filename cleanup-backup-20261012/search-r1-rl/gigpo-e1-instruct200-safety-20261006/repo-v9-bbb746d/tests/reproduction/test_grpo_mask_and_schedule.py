"""CPU checks of real core functions: groups, loss provenance and run length."""
import ast
from collections import defaultdict
import os
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
try:
    from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage, compute_policy_loss, kl_penalty
    from verl.utils.torch_functional import masked_mean, get_constant_schedule_with_warmup
except ImportError:
    if os.environ.get('SEARCH_R1_REQUIRE_REAL_DATAPROTO') == '1':
        raise
    namespace = dict(torch=torch, np=np, defaultdict=defaultdict,
                     Optimizer=torch.optim.Optimizer, LambdaLR=torch.optim.lr_scheduler.LambdaLR)
    funcs = ast.parse((ROOT / 'verl/utils/torch_functional.py').read_text())
    selected = [node for node in funcs.body if isinstance(node, ast.FunctionDef)
                and node.name in ('masked_mean', 'get_constant_schedule_with_warmup')]
    exec(compile(ast.Module(body=selected, type_ignores=[]), 'torch_functional.py', 'exec'), namespace)
    namespace['verl_F'] = SimpleNamespace(masked_mean=namespace['masked_mean'])
    funcs = ast.parse((ROOT / 'verl/trainer/ppo/core_algos.py').read_text())
    selected = [node for node in funcs.body if isinstance(node, ast.FunctionDef)
                and node.name in ('compute_grpo_outcome_advantage', 'compute_policy_loss', 'kl_penalty')]
    exec(compile(ast.Module(body=selected, type_ignores=[]), 'core_algos.py', 'exec'), namespace)
    globals().update({name: namespace[name] for name in (
        'compute_grpo_outcome_advantage', 'compute_policy_loss', 'kl_penalty',
        'masked_mean', 'get_constant_schedule_with_warmup')})


class GRPOBoundary(unittest.TestCase):
    def test_mixed_unique_groups_survive_permutation_and_equal_reward_group(self):
        # Two unequal groups and an all-zero group, five trajectories each.
        uid = np.array([0] * 5 + [79168] * 5 + [79169] * 5)
        outcome = torch.tensor([1., 0., 0., 1., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 0.])
        reward = torch.zeros((15, 3))
        reward[:, -1] = outcome
        mask = torch.ones_like(reward)
        advantage, _ = compute_grpo_outcome_advantage(reward, mask, uid)
        expected = torch.cat([(group - group.mean()) / (group.std() + 1e-6)
                              for group in outcome.split(5)])
        torch.testing.assert_close(advantage[:, 0], expected)
        self.assertTrue(torch.equal(advantage[-5:], torch.zeros((5, 3))))
        order = torch.tensor([7, 0, 12, 4, 8, 1, 14, 3, 6, 13, 2, 10, 5, 11, 9])
        permuted, _ = compute_grpo_outcome_advantage(reward[order], mask[order], uid[order.numpy()])
        torch.testing.assert_close(permuted, advantage[order])
        self.assertTrue(torch.isfinite(permuted).all())

    def test_observation_tokens_have_zero_pg_kl_and_entropy_gradient(self):
        # Same finite log probabilities as model positions; only provenance differs.
        mask = torch.tensor([[1., 0., 0., 1.]])
        old = torch.tensor([[-2., -2., -2., -2.]])
        for objective in ('pg', 'kl', 'entropy'):
            with self.subTest(objective=objective):
                value = torch.tensor([[-1.95, -1.95, -1.95, -1.95]], requires_grad=True)
                if objective == 'pg':
                    loss, _, _ = compute_policy_loss(old, value, torch.ones_like(value), mask, .2)
                elif objective == 'kl':
                    loss = masked_mean(kl_penalty(value, old, 'low_var_kl'), mask)
                else:
                    loss = masked_mean(value, mask)
                loss.backward()
                torch.testing.assert_close(value.grad[:, 1:3], torch.zeros((1, 2)))
                self.assertTrue(torch.all(value.grad[:, [0, 3]] != 0))

    def test_501_threshold_produces_500_updates_and_identical_warmup_sequence(self):
        flow = runpy.run_path(str(ROOT / 'tests/reproduction/test_final_checkpoint.py'))
        trainer = flow['FakeTrainer'](1, 501)
        execute = flow['real_iteration_tail']()
        while trainer.global_steps < trainer.total_training_steps:
            execute(trainer)
        updates = [event for event in trainer.events if event[0] == 'update']
        self.assertEqual(len(updates), 500)
        self.assertEqual(updates[-1][1], 500)
        self.assertEqual([event for event in trainer.events if event[0] == 'save'][-1][1], 500)
        sequences = []
        for horizon in (500, 501):
            warmup = int(.285 * horizon)
            self.assertEqual(warmup, 142)
            parameter = torch.nn.Parameter(torch.ones(1))
            optimizer = torch.optim.AdamW([parameter], lr=1e-6)
            scheduler = get_constant_schedule_with_warmup(optimizer, warmup)
            rates = []
            for _ in range(500):
                rates.append(optimizer.param_groups[0]['lr'])
                optimizer.step()
                scheduler.step()
            sequences.append(rates)
        self.assertEqual(sequences[0], sequences[1])


if __name__ == '__main__':
    unittest.main()
