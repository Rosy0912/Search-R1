import unittest
import torch
from verl.trainer.ppo import core_algos
from search_e1.grpo_batching import trajectory_mean, trajectory_policy_loss


class TestTrajectoryBatching(unittest.TestCase):
    def test_loss_and_gradient_match_single_trajectory_accumulation(self):
        torch.manual_seed(71)
        old = torch.randn(4, 7, dtype=torch.float64)
        ref = old + torch.randn_like(old) * 0.1
        advantages = torch.randn_like(old)
        mask = (torch.arange(7)[None, :] < torch.tensor([1, 3, 5, 7])[:, None]).double()
        initial = old + torch.randn_like(old) * 0.3

        single = initial.clone().requires_grad_()
        baseline = single.sum() * 0.
        for i in range(4):
            part = slice(i, i+1)
            pg, _, _ = core_algos.compute_policy_loss(old[part], single[part], advantages[part], mask[part], 0.2)
            kl = core_algos.kl_penalty(single[part], ref[part], 'low_var_kl')
            baseline = baseline + (pg + 0.001 * trajectory_mean(kl, mask[part])) / 4
        baseline.backward()

        batched = initial.clone().requires_grad_()
        loss = batched.sum() * 0.
        for start in (0, 2):
            part = slice(start, start+2)
            pg, _, _ = trajectory_policy_loss(old[part], batched[part], advantages[part], mask[part], 0.2)
            kl = core_algos.kl_penalty(batched[part], ref[part], 'low_var_kl')
            loss = loss + (pg + 0.001 * trajectory_mean(kl, mask[part])) / 2
        loss.backward()
        torch.testing.assert_close(loss, baseline, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(batched.grad, single.grad, rtol=1e-12, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
