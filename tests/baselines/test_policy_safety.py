import importlib.util
import math
from pathlib import Path
import tempfile
import unittest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


SOURCE = Path(__file__).resolve().parents[2] / 'search_r1/baselines/policy_safety.py'
spec = importlib.util.spec_from_file_location('policy_safety_tested', SOURCE)
safety = importlib.util.module_from_spec(spec)
spec.loader.exec_module(safety)


def worker(rank, rendezvous, output):
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=2)
    try:
        old = torch.zeros(2, dtype=torch.float64)
        new = torch.full((2,), .2 + rank * .2, dtype=torch.float64)
        adv = -torch.ones_like(old)
        weights = torch.ones_like(old) * (1 if rank == 0 else 3)
        stats, tail = safety.update_statistics(old, new, adv, torch.ones_like(old), weights)
        decision = safety.admission(stats, 1., .5)
        quantiles = safety.global_negative_quantiles(tail)
        # One-rank corrupt valid probability: matching collectives refuse both.
        if rank == 1: new[0] = float('nan')
        bad, _ = safety.update_statistics(old, new, adv, torch.ones_like(old), weights)
        refusal = safety.admission(bad, 1., .5)
        torch.save(dict(decision=decision, refusal=refusal, quantiles=quantiles),
                   Path(output) / f'{rank}.pt')
    finally:
        dist.destroy_process_group()


class PolicySafetyTests(unittest.TestCase):
    def test_formula_and_gradient(self):
        current = torch.tensor([-6., -.8, -.1, .05, .4, 2., 7.], dtype=torch.float64)
        for sign in (-1., 0., 1.):
            x = current.clone().requires_grad_()
            a = torch.full_like(x, sign)
            actual = safety.dual_clip_terms(torch.zeros_like(x), x, a, torch.ones_like(x))
            r = x.exp()
            ordinary = torch.maximum(-a*r, -a*r.clamp(.8, 1.2))
            expected = torch.where(a < 0, torch.minimum(ordinary, 3*a.abs()), ordinary)
            torch.testing.assert_close(actual, expected)
            grad = torch.autograd.grad(actual.sum(), x, retain_graph=True)[0]
            direct = torch.autograd.grad(expected.sum(), x)[0]
            torch.testing.assert_close(grad, direct)

    def test_extreme_and_zero_advantage(self):
        x = torch.tensor([800., -800., 800., -800., 800.], dtype=torch.float64, requires_grad=True)
        a = torch.tensor([1., 1., -1., -1., 0.], dtype=torch.float64)
        loss = safety.dual_clip_terms(torch.zeros_like(x), x, a, torch.ones_like(x))
        torch.testing.assert_close(loss, torch.tensor([-1.2, 0., 3., .8, 0.], dtype=torch.float64))
        grad = torch.autograd.grad(loss.sum(), x)[0]
        self.assertTrue(torch.isfinite(grad).all())
        torch.testing.assert_close(grad, torch.zeros_like(grad))

    def test_clamp_boundary_subgradient_convention(self):
        # PyTorch clamp uses the interior derivative at its boundary. Direct
        # minimum/maximum may choose another valid subgradient at a tie.
        x = torch.tensor([math.log(3.), math.log(.8), math.log(1.2)],
                         dtype=torch.float64, requires_grad=True)
        loss = safety.dual_clip_terms(torch.zeros_like(x), x,
                                     torch.tensor([-1., -1., 1.]), torch.ones(3))
        torch.testing.assert_close(torch.autograd.grad(loss.sum(), x)[0],
                                   torch.tensor([3., .8, -1.2], dtype=torch.float64))

    def test_padding_masked_before_difference(self):
        old = torch.tensor([0., float('nan')], dtype=torch.float64)
        new = torch.tensor([.1, float('inf')], dtype=torch.float64, requires_grad=True)
        adv = torch.tensor([-1., float('nan')], dtype=torch.float64)
        mask = torch.tensor([1, 0])
        loss = safety.dual_clip_terms(old, new, adv, mask)
        self.assertTrue(torch.isfinite(loss).all())
        self.assertTrue(torch.isfinite(torch.autograd.grad(loss.sum(), new)[0]).all())
        with self.assertRaises(ValueError): safety.dual_clip_terms(old, new, adv, torch.ones(2))

    def test_kl_tail_empty_and_overflow_refusal(self):
        old = torch.zeros(3, dtype=torch.float64)
        adv = torch.tensor([-1., -1., 1.], dtype=torch.float64)
        mask = torch.ones_like(old)
        stats, tail = safety.update_statistics(old, torch.tensor([.1, 2., .2]), adv, mask, mask)
        result = safety.admission(stats, .01, .1)
        self.assertFalse(result['allowed'])
        self.assertEqual(result['negative_tail_fraction'], .5)
        self.assertIn('sampled_old_policy_kl', result['reasons'])
        self.assertIn('negative_tail_fraction', result['reasons'])
        overflow, _ = safety.update_statistics(old, torch.tensor([800., .1, .1]), adv, mask, mask)
        rejected = safety.admission(overflow, .01, 1.)
        self.assertFalse(rejected['kl_computation_valid'])
        self.assertIsNone(rejected['sampled_old_policy_kl'])
        self.assertEqual(rejected['overflow_count'], 1)
        empty, _ = safety.update_statistics(old, old, adv, mask*0, mask*0)
        self.assertFalse(safety.admission(empty, .01, 1.)['allowed'])

    def test_invalid_weight_statistics_are_finite(self):
        x = torch.zeros(2)
        stats, _ = safety.update_statistics(x, x, -torch.ones(2), torch.ones(2),
                                             torch.tensor([float('nan'), -1.]))
        self.assertTrue(all(torch.isfinite(v) for v in stats.values()))
        self.assertFalse(safety.admission(stats, .01, 1.)['allowed'])

    def test_original_leaf_measure_split_invariance(self):
        torch.manual_seed(7)
        old = torch.zeros((6, 8), dtype=torch.float64)
        current = torch.randn_like(old) * 2
        advantages = torch.randn_like(old)
        mask = torch.arange(8)[None, :] < torch.arange(1, 7)[:, None]
        weights = mask / mask.sum(-1, keepdim=True)
        x = current.clone().requires_grad_()
        total = (safety.dual_clip_terms(old, x, advantages, mask)*weights).sum() / 6
        g = torch.autograd.grad(total, x)[0]
        y = current.clone().requires_grad_()
        split = sum((safety.dual_clip_terms(old[i:i+2], y[i:i+2], advantages[i:i+2], mask[i:i+2])
                     *weights[i:i+2]).sum()/6 for i in range(0, 6, 2))
        torch.testing.assert_close(total, split)
        torch.testing.assert_close(g, torch.autograd.grad(split, y)[0])

    def test_real_distributed_global_measure_and_refusal(self):
        with tempfile.TemporaryDirectory() as folder:
            mp.spawn(worker, args=(folder+'/rendezvous', folder), nprocs=2, join=True)
            outputs = [torch.load(Path(folder)/f'{r}.pt', weights_only=False) for r in range(2)]
        self.assertEqual(outputs[0], outputs[1])
        result = outputs[0]['decision']
        self.assertTrue(result['allowed'])
        expected = ((math.expm1(.2)-.2) + 3*(math.expm1(.4)-.4))/4
        self.assertAlmostEqual(result['sampled_old_policy_kl'], expected)
        self.assertEqual(result['mass'], 8.)
        self.assertFalse(outputs[0]['refusal']['allowed'])
        self.assertEqual(outputs[0]['refusal']['invalid_count'], 1.)
        self.assertEqual(outputs[0]['quantiles']['negative_log_ratio_p99'], .4)


if __name__ == '__main__': unittest.main()
