"""Analytic paper counterexamples, and actual actor empty-minibatch safety."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
import torch
from omegaconf import OmegaConf
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from verl import DataProto
from search_r1.baselines.objectives import attach_policy_weights, weighted_policy_loss
from verl.workers.actor.dp_actor import DataParallelPPOActor


def batch(mask, uids):
    mask = torch.tensor(mask, dtype=torch.float32)
    return DataProto.from_dict(dict(responses=torch.zeros_like(mask).long(), attention_mask=mask.long(), info_mask=mask.long()),
                               non_tensors={'uid': np.asarray(uids)})


class PaperMeasures(unittest.TestCase):
    def test_tree_two_unequal_length_trajectories_cancel(self):
        data = batch([[1, 0, 0], [1, 1, 1]], ['q', 'q'])
        attach_policy_weights(data, 'tree_grpo')
        theta = torch.tensor(0., requires_grad=True)
        current = theta.expand(2, 3)
        loss, _, _ = weighted_policy_loss(torch.zeros_like(current), current,
            torch.tensor([[1., 0, 0], [-1., -1, -1]]), data.batch['info_mask'], data.batch['policy_weights'], .2, 2.)
        loss.backward()
        torch.testing.assert_close(theta.grad, torch.tensor(0.), atol=1e-6, rtol=0)

    def test_topd_question_measure_survives_micro_splitting(self):
        data = batch([[1, 0, 0, 0], [1, 0, 0, 0], [1, 1, 1, 1]], ['a', 'a', 'b'])
        attach_policy_weights(data, 'topd')
        # Question a has 2 tokens (A=1), b has 4 (A=3): equal-question gradient=-2.
        theta = torch.tensor(0., requires_grad=True)
        for row in range(3):
            current = theta.expand(1, 4)
            advantage = torch.full_like(current, 1. if row < 2 else 3.)
            loss, _, _ = weighted_policy_loss(torch.zeros_like(current), current, advantage,
                data.batch['info_mask'][row:row+1], data.batch['policy_weights'][row:row+1], .2, 3.)
            loss.backward()
        torch.testing.assert_close(theta.grad, torch.tensor(-2.))

    def test_gigpo_joint_action_clip_differs_from_individual_token_clip(self):
        data = batch([[1, 1]], ['q'])
        data.batch['step_ids'] = torch.tensor([[0, 0]])
        attach_policy_weights(data, 'gigpo')
        # Both token ratios=1.1 are inside the clip interval, joint action=1.21 exceeds it.
        theta = torch.tensor(float(np.log(1.1)), requires_grad=True)
        current = theta.expand(1, 2)
        loss, clip, _ = weighted_policy_loss(torch.zeros_like(current), current, torch.ones_like(current),
            data.batch['info_mask'], data.batch['policy_weights'], .2, 1., data.batch['step_ids'])
        torch.testing.assert_close(loss, torch.tensor(-1.2))
        self.assertEqual(clip.item(), 1.)
        loss.backward()
        self.assertEqual(theta.grad.item(), 0.)

    def test_gigpo_actions_get_equal_weight_and_joint_gradient(self):
        data = batch([[1, 1, 1]], ['q'])
        data.batch['step_ids'] = torch.tensor([[0, 1, 1]])
        attach_policy_weights(data, 'gigpo')
        torch.testing.assert_close(data.batch['policy_weights'], torch.tensor([[.5, .25, .25]]))
        current = torch.zeros(1, 3, requires_grad=True)
        loss, _, _ = weighted_policy_loss(torch.zeros_like(current), current, torch.ones_like(current),
            data.batch['info_mask'], data.batch['policy_weights'], .2, 1., data.batch['step_ids'])
        loss.backward()
        torch.testing.assert_close(current.grad, torch.tensor([[-.5, -.5, -.5]]))

    def test_actor_global_empty_minibatch_does_not_decay_or_move_momentum(self):
        model = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.1, weight_decay=.1)
        # Populate optimizer momentum first; zero loss must skip step, not just produce zero gradients.
        model(torch.ones(1, 1)).sum().backward(); optimizer.step(); optimizer.zero_grad()
        before = model.weight.detach().clone()
        actor = DataParallelPPOActor.__new__(DataParallelPPOActor)
        actor.actor_module, actor.actor_optimizer = model, optimizer
        actor.config = OmegaConf.create(dict(ppo_mini_batch_size=2, ppo_micro_batch_size=1, ppo_epochs=2,
            state_masking=True, use_kl_loss=False, use_dynamic_bsz=False, baseline_method='topd'))
        tensors = dict(responses=torch.zeros(2, 1).long(), input_ids=torch.ones(2, 2).long(),
            attention_mask=torch.ones(2, 2).long(), position_ids=torch.zeros(2, 2).long(),
            old_log_probs=torch.zeros(2, 1), advantages=torch.zeros(2, 1), loss_mask=torch.zeros(2, 1))
        data = DataProto.from_dict(tensors, meta_info={'temperature': 1.})
        metrics = actor.update_policy(data)
        self.assertEqual(metrics['actor/optimizer_steps'], 0)
        self.assertEqual(metrics['actor/skipped_empty_minibatches'], 2)
        torch.testing.assert_close(model.weight, before, atol=0, rtol=0)
