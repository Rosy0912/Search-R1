"""Unexecuted new CPU IO/tiny-Adam fixtures, not actual FSDP or multi-rank acceptance."""
import os
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from search_r1.baselines.partial_update import _adam_steps, preserve_worker, collect_worker_receipts
from search_r1.baselines.policy_safety import PolicySafetyRejected


class PartialPreservation(unittest.TestCase):
    def test_empty_fresh_slots_have_no_adam_steps(self):
        self.assertEqual(_adam_steps({'optimizer': {'state': {0: {}, 1: {}}}}), 0)

    def test_unreadable_or_different_slots_not_fabricated(self):
        for slots in ({0: {'exp_avg': torch.zeros(1)}},
                {0: {'step': torch.tensor(7)}, 1: {'step': torch.tensor(8)}}):
            with self.subTest(slots=slots), self.assertRaises(RuntimeError):
                _adam_steps({'optimizer': {'state': slots}})

    def test_actual_tiny_cpu_adam_partial_saved_without_step_or_reset(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(step / 57., 1.))
        for _ in range(7):
            optimizer.zero_grad()
            model(torch.ones(1, 2)).sum().backward()
            optimizer.step()
        before_weights = [v.detach().clone() for v in model.parameters()]
        before_steps = [int(v['step']) for v in optimizer.state.values()]
        before_rng = torch.get_rng_state().clone()
        worker = SimpleNamespace(actor_module_fsdp=model, actor_optimizer=optimizer,
            actor_lr_scheduler=scheduler, rank=0, world_size=1, rollout_sharding_manager=None)
        # This is one CPU worker fixture; it never asserts real eight-rank coverage.
        data = SimpleNamespace(meta_info={'baseline_outer_step': 1})
        refusal = PolicySafetyRejected('matched test fixture', dict(outer_step=1, mini_index=6,
            phase='after', actual_optimizer_writes=7))
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {
                'SEARCH_R1_BASELINE_AUDIT_DIR': folder, 'SEARCH_R1_BASELINE_RUN_ID': 'cpu-only-fixture',
                'SEARCH_R1_BASELINE_COMMIT': 'f' * 40, 'SEARCH_R1_BASELINE_CONFIG_SHA256': 'a' * 64,
                'SLURM_JOB_ID': '0'}):
            result = preserve_worker(worker, data, refusal)
            state = torch.load(result['state_receipt']['path'], map_location='cpu', weights_only=False)
            self.assertEqual(state['partial_update']['actual_adam_total'], 7)
            self.assertEqual(state['partial_update']['completed_outer_updates'], 0)
            self.assertEqual(state['partial_update']['actual_extra_adam_beyond_complete_outers'], 7)
            self.assertEqual(state['partial_update']['scheduler_step'], 0)
            self.assertFalse(state['partial_update']['automatic_rollback'])
            self.assertFalse(state['partial_update']['exact_resume_accepted'])
            self.assertTrue((Path(folder) / 'partial-updates/outer_000001/rank_0.json').is_file())
            self.assertEqual([int(v['step']) for v in optimizer.state.values()], before_steps)
            self.assertEqual(scheduler.last_epoch, 0)
            self.assertTrue(torch.equal(torch.get_rng_state(), before_rng))
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(before_weights, model.parameters())))

    def test_timeout_keeps_available_receipts_and_reports_missing_rank(self):
        from search_r1.baselines.recovery import atomic_torch_save
        with tempfile.TemporaryDirectory() as value, patch.dict(os.environ, {
                'SEARCH_R1_BASELINE_RUN_ID': 'current-run', 'SEARCH_R1_BASELINE_COMMIT': 'f'*40,
                'SEARCH_R1_BASELINE_CONFIG_SHA256': 'a'*64, 'SLURM_JOB_ID': '123'}):
            folder = Path(value)
            for rank in range(7):
                receipt = atomic_torch_save({'actual_saved_rank': rank}, folder / f'rank_{rank}.pt')
                row = dict(status='PARTIAL_WORKER_STATE_SAVED', rank=rank, job_id='123',
                    run_id='current-run', source_commit='f'*40, config_sha256='a'*64,
                    state_receipt=receipt, partial_update=dict(completed_outer_updates=0,
                        outer_step=1, in_progress_outer_id=1, actual_partial_adam_steps=7))
                (folder / f'rank_{rank}.json').write_text(json.dumps(row))
            result = collect_worker_receipts(folder, 0, 1)
            self.assertEqual(len(result['workers']), 7)
            self.assertEqual(result['missing'], [7])
            self.assertEqual(result['errors'], [])
            failure = dict(status='PARTIAL_WORKER_STATE_SAVE_FAILED', rank=3, job_id='123',
                run_id='current-run', source_commit='f'*40, config_sha256='a'*64, outer_step=1)
            (folder / 'rank_3-save-failed.json').write_text(json.dumps(failure))
            result = collect_worker_receipts(folder, 0, 1)
            self.assertEqual(len(result['workers']), 7)
            self.assertEqual(len(result['failures']), 1)
            old = json.loads((folder / 'rank_3.json').read_text())
            old['run_id'] = 'stale-run'
            (folder / 'rank_3.json').write_text(json.dumps(old))
            result = collect_worker_receipts(folder, 0, 1)
            self.assertEqual(len(result['workers']), 6)
            self.assertEqual(len(result['errors']), 1)
            self.assertEqual(len(result['failures']), 1)

    def test_rank_receipt_cannot_point_to_another_snapshot(self):
        from search_r1.baselines.recovery import atomic_torch_save
        with tempfile.TemporaryDirectory() as value, patch.dict(os.environ, {
                'SEARCH_R1_BASELINE_RUN_ID': 'current-run', 'SEARCH_R1_BASELINE_COMMIT': 'f'*40,
                'SEARCH_R1_BASELINE_CONFIG_SHA256': 'a'*64, 'SLURM_JOB_ID': '123'}):
            folder = Path(value)
            receipt = atomic_torch_save({}, folder / 'rank_1.pt')
            row = dict(status='PARTIAL_WORKER_STATE_SAVED', rank=0, job_id='123',
                run_id='current-run', source_commit='f'*40, config_sha256='a'*64,
                state_receipt=receipt, partial_update=dict(completed_outer_updates=0,
                    outer_step=1, in_progress_outer_id=1, actual_partial_adam_steps=0))
            (folder / 'rank_0.json').write_text(json.dumps(row))
            result = collect_worker_receipts(folder, 0, 1)
            self.assertEqual(result['workers'], [])
            self.assertEqual(result['errors'][0]['rank'], 0)

    def test_matched_gradient_refusal_survives_diagnostic_io_failure(self):
        from search_r1.baselines.gradient_norm import MatchedGradientRejected, _matched_gradient_rejection
        with patch('search_r1.baselines.gradient_norm._failure', side_effect=OSError('disk full')):
            with self.assertRaises(MatchedGradientRejected) as caught:
                _matched_gradient_rejection([], torch.tensor(float('nan')))
        self.assertIn('disk full', caught.exception.diagnostic_error)
        restored = pickle.loads(pickle.dumps(caught.exception))
        self.assertEqual(restored.diagnostic_error, caught.exception.diagnostic_error)
        self.assertEqual(str(restored), str(caught.exception))


if __name__ == '__main__': unittest.main()
