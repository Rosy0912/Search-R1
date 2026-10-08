"""Regression counterexamples from the independent audit, not synthetic happy paths."""
from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from verl.trainer.ppo.core_algos import compute_policy_loss
from verl.utils.torch_functional import masked_mean
from search_r1.baselines.advantages import token_mean_ppo
from search_r1.baselines.distillation import global_token_count


class NumericSafety(unittest.TestCase):
    def test_masked_overflow_never_contaminates_valid_gradient(self):
        for global_mean in (False, True):
            current = torch.tensor([[0., float('nan'), 0.]], requires_grad=True)
            old = torch.tensor([[0., float('inf'), -100.]])
            advantage = torch.tensor([[1., float('nan'), 0.]])
            mask = torch.tensor([[1., 0., 0.]])
            args = (old, current, advantage, mask, .2)
            loss, clip, kl = token_mean_ppo(*args, 1.) if global_mean else compute_policy_loss(*args)
            torch.testing.assert_close(loss, torch.tensor(-1.))
            self.assertTrue(torch.isfinite(clip) and torch.isfinite(kl))
            loss.backward()
            torch.testing.assert_close(current.grad, torch.tensor([[-1., 0., 0.]]))

    def test_empty_mask_has_zero_loss_and_gradient(self):
        current = torch.tensor([[float('nan'), 200.]], requires_grad=True)
        loss, clip, kl = compute_policy_loss(torch.zeros_like(current), current,
                                             torch.ones_like(current), torch.zeros_like(current), .2)
        self.assertEqual((loss.item(), clip.item(), kl.item()), (0., 0., 0.))
        loss.backward()
        torch.testing.assert_close(current.grad, torch.zeros_like(current))
        count, world = global_token_count(torch.zeros_like(current))
        self.assertEqual((count.item(), world), (0., 1))

    def test_fractional_mask_denominator_is_not_clamped_to_one(self):
        values = torch.tensor([3., float('nan')], requires_grad=True)
        mean = masked_mean(values, torch.tensor([.25, 0.]))
        self.assertEqual(mean.item(), 3.)
        mean.backward()
        torch.testing.assert_close(values.grad, torch.tensor([1., 0.]))

    def test_invalid_supervised_probability_fails_explicitly(self):
        for value in (float('nan'), 100.):
            with self.assertRaises(ValueError):
                compute_policy_loss(torch.tensor([0.]), torch.tensor([value]),
                                    torch.tensor([1.]), torch.tensor([1.]), .2)

    def test_masked_nonfinite_logits_fail_before_entropy_jacobian(self):
        # Audit counterexample: final masking gave a finite loss but NaN gradients
        # through the earlier softmax. Reject the raw distribution before that op.
        from unittest.mock import patch
        from verl.utils.torch_functional import entropy_from_logits
        for invalid in (float('inf'), float('-inf'), float('nan')):
            logits = torch.tensor([[[0., 1.], [invalid, 0.]]], requires_grad=True)
            with patch('torch.nn.functional.softmax') as softmax:
                with self.assertRaisesRegex(ValueError, 'Nonfinite logits'):
                    masked_mean(entropy_from_logits(logits), torch.tensor([[1., 0.]]))
                softmax.assert_not_called()
            self.assertIsNone(logits.grad)
        logits = torch.tensor([[[0., 1.], [100., 0.]]], requires_grad=True)
        masked_mean(entropy_from_logits(logits), torch.tensor([[1., 0.]])).backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        torch.testing.assert_close(logits.grad[0, 1], torch.zeros(2), rtol=0, atol=0)
