"""Read unchanged frozen source; tiny CPU mathematics and actual log evidence only."""
import copy
import hashlib
import json
import math
from pathlib import Path
import types
import unittest

import torch

H = Path(__file__).resolve().parent
REPO = H.parent / 'opsd-unclipped-multitraj200-teacherrecompute'
BINDING = H / 'runtime-tools-bound-fcecbea5d939-v5.json'
assert hashlib.sha256(BINDING.read_bytes()).hexdigest() == '8cec4d887f656cecf7d533b91ac020da169ded4142a14890351501c9c76b2228'
BOUND = json.loads(BINDING.read_bytes())
assert BOUND['source_commit'] == 'fcecbea5d939e41dcfb3b118a26ef049a9bda3d6'


def unchanged_module(name, relative):
    path = REPO / relative
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == BOUND['source_files'][relative]
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec', dont_inherit=True), module.__dict__)
    return module


DISTILL = unchanged_module('unchanged_cpu_distillation', 'search_r1/baselines/distillation.py')
SCHEDULE = unchanged_module('unchanged_cpu_torch_functional', 'verl/utils/torch_functional.py')
GATE = unchanged_module('unchanged_cpu_checkpoint_gate', 'search_r1/baselines/checkpoint_gate.py')


def actual_rows():
    raw = (H / 'job3304-snapshot-10/train.log').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == 'e9badb53ac417ca51da23a5103abbc4a4cb5664a8e818b5bb015efee1e557f13'
    rows = []
    for line in raw.decode().splitlines():
        if 'OPSD_E1_OUTER_JSON=' in line:
            value = GATE.parse_log_marker(line, 'OPSD_E1_OUTER_JSON=', driver_pid=2548319, job_id='3304')
            rows.append(value)
    return rows


class IndependentMathRuntime(unittest.TestCase):
    def test_forward_kl_teacher_detached_and_analytic_gradient(self):
        student = torch.tensor([[.2, -.9, .7, 1.3]], requires_grad=True)
        teacher = torch.tensor([[1.1, .3, -.8, .7]], requires_grad=True)
        loss = DISTILL.generalized_jsd(student, teacher, beta=0., temperature=1., component_clip=None).sum()
        p = torch.softmax(student.detach(), -1)
        q = torch.softmax(teacher.detach(), -1)
        manual = (q * (q.log() - p.log())).sum()
        torch.testing.assert_close(loss.detach(), manual, atol=2e-7, rtol=2e-6)
        loss.backward()
        torch.testing.assert_close(student.grad, p - q, atol=2e-7, rtol=2e-6)
        self.assertIsNone(teacher.grad)
        updated = student.detach() - .01 * student.grad
        self.assertLess(float(DISTILL.generalized_jsd(updated, teacher, beta=0., temperature=1., component_clip=None).sum()), float(loss))

    def test_unequal_rank_tokens_eight_micro_accumulations_equal_global_token_mean(self):
        g = torch.Generator().manual_seed(8304)
        features = torch.randn(256, 7, 3, generator=g)
        teacher = torch.randn(256, 7, 5, generator=g)
        mask = torch.zeros(256, 7)
        for rank in range(8):
            for row in range(32):
                mask[rank * 32 + row, :1 + (rank + row) % 7] = 1
        n = mask.sum()
        weight = torch.randn(3, 5, generator=g)
        full = weight.clone().requires_grad_()
        dense_loss = (DISTILL.generalized_jsd(features @ full, teacher, 0., 1., None) * mask).sum() / n
        dense_loss.backward()
        accumulated = []
        local_means = []
        losses = []
        for rank in range(8):
            w = weight.clone().requires_grad_()
            wrong = weight.clone().requires_grad_()
            rank_loss = 0.
            for start in range(rank * 32, (rank + 1) * 32, 4):
                stop = start + 4
                micro = (DISTILL.generalized_jsd(features[start:stop] @ w, teacher[start:stop], 0., 1., None) * mask[start:stop]).sum() / (n / 8)
                micro.backward()
                rank_loss += float(micro.detach())
            wrong_loss = (DISTILL.generalized_jsd(features[rank*32:(rank+1)*32] @ wrong, teacher[rank*32:(rank+1)*32], 0., 1., None) * mask[rank*32:(rank+1)*32]).sum() / mask[rank*32:(rank+1)*32].sum()
            wrong_loss.backward()
            accumulated.append(w.grad)
            local_means.append(wrong.grad)
            losses.append(rank_loss)
        torch.testing.assert_close(torch.stack(accumulated).mean(0), full.grad, atol=2e-7, rtol=2e-6)
        self.assertAlmostEqual(sum(losses) / 8, float(dense_loss), places=6)
        self.assertGreater(float((torch.stack(local_means).mean(0) - full.grad).abs().max()), 1e-6)

    def test_global_l2_clip_preclip_norm_can_exceed_one(self):
        shards = [torch.nn.Parameter(torch.zeros(3)) for _ in range(8)]
        original = []
        for rank, parameter in enumerate(shards):
            parameter.grad = torch.tensor([rank + .3, rank + .7, -rank - .2])
            original.append(parameter.grad.clone())
        before = torch.linalg.vector_norm(torch.cat(original))
        reported = torch.nn.utils.clip_grad_norm_(shards, 1.)
        torch.testing.assert_close(reported, before)
        self.assertGreater(float(reported), 1.)
        after = torch.linalg.vector_norm(torch.cat([p.grad for p in shards]))
        self.assertLessEqual(float(after), 1.000001)
        torch.testing.assert_close(torch.cat([p.grad for p in shards]), torch.cat(original) / (before + 1e-6), atol=2e-7, rtol=2e-6)

    def test_real_scheduler_helper_fifty_adam_calls_and_five_outer_ticks(self):
        value = torch.nn.Parameter(torch.tensor(1.))
        optimizer = torch.optim.AdamW([value], lr=1e-6)
        scheduler = SCHEDULE.get_constant_schedule_with_warmup(optimizer, 57)
        original = value.detach().clone()
        self.assertEqual(scheduler.last_epoch, 0)
        for outer in range(1, 6):
            self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 1e-6 * (outer - 1) / 57, places=20)
            for _ in range(10):
                optimizer.zero_grad()
                value.square().backward()
                optimizer.step()
            if outer == 1:
                torch.testing.assert_close(value.detach(), original, atol=0, rtol=0)
            scheduler.step()
            self.assertEqual(scheduler.last_epoch, outer)
            self.assertAlmostEqual(scheduler.get_last_lr()[0], 1e-6 * outer / 57, places=20)
        self.assertEqual(float(optimizer.state[value]['step']), 50.)
        self.assertNotEqual(float(value.detach()), float(original))

    def test_actual_five_outer_eight_rank_clocks_finiteness_and_learning_rates(self):
        rows = actual_rows()
        self.assertEqual([r['actual_outer_updates'] for r in rows], list(range(1, 6)))
        for outer, row in enumerate(rows, 1):
            self.assertEqual(row['source_commit'], BOUND['source_commit'])
            self.assertEqual(row['config_sha256'], '0f0fc3a6aff4648ce0d2c8ec28eb104e27678315ad09824c90da043956a0b990')
            self.assertEqual(row['driver_pid'], 2548319)
            self.assertEqual(row['job_id'], '3304')
            metrics = row['metrics']
            GATE.validate_outer(metrics, profile='multitraj200')
            for rank in range(8):
                suffix = '/rank_' + str(rank)
                self.assertAlmostEqual(metrics['actor/optimizer_lr_used_mean' + suffix], 1e-6 * (outer - 1) / 57, places=20)
                self.assertEqual(metrics['teacher/cuda_update_entries_cumulative_per_rank' + suffix], outer)
                self.assertEqual(metrics['actor/loss_mask_valid' + suffix], 1)
                self.assertTrue(math.isfinite(metrics['actor/opsd_loss_mean' + suffix]))
                self.assertTrue(math.isfinite(metrics['actor/grad_norm_mean' + suffix]))
            self.assertAlmostEqual(metrics['actor/lr'], 1e-6 * outer / 57, places=20)
            self.assertEqual(metrics['rollout/trajectories'], 2560)
            for key in ('critic/advantages/min', 'critic/advantages/mean', 'critic/advantages/max', 'critic/returns/min', 'critic/returns/mean', 'critic/returns/max'):
                self.assertEqual(metrics[key], 0)
            self.assertEqual(metrics['diagnostic/student_archived_count'], 2560)
            self.assertEqual(metrics['diagnostic/teacher_archived_count'], 512)
            self.assertEqual(metrics['diagnostic/paired_archive_complete'], 1)
        wrong = copy.deepcopy(rows[3]['metrics'])
        wrong['actor/optimizer_steps/rank_7'] = 9
        with self.assertRaises(RuntimeError):
            GATE.validate_outer(wrong, profile='multitraj200')
        wrong = copy.deepcopy(rows[3]['metrics'])
        wrong['actor/opsd_loss_finite/rank_7'] = 0
        with self.assertRaises(RuntimeError):
            GATE.validate_outer(wrong, profile='multitraj200')


if __name__ == '__main__':
    unittest.main(verbosity=2)
