"""Actual tiny CPU Adam files and driver failure flow; no Ray/GPU claims."""
import ast
import copy
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace as N
import unittest
from unittest.mock import patch

import torch

from search_r1.baselines.training_state import (CheckpointBatchSampler, atomic_save, worker_snapshot,
                                               restore_driver, file_sha)
from search_r1.baselines.tree_partial_state import (PROTOCOL, partial_context, partial_driver_snapshot,
    stop_on_partial_actor_result, preserve_unmatched_actor_failure)
from search_r1.baselines.policy_safety_runtime import PolicySafetyRefusal


ENV = dict(SLURM_JOB_ID='cpu-only', SEARCH_R1_BASELINE_RUN_ID='synthetic-partial',
    SEARCH_R1_BASELINE_COMMIT='cpu-source', SEARCH_R1_BASELINE_CONFIG_SHA256='cpu-config',
    SEARCH_R1_BASELINE_DATA_IDENTITY='cpu-data', SEARCH_R1_MODEL_IDENTITY='tiny-cpu-model')


class Batch:
    def __init__(self):
        self.batch = dict(input_ids=torch.tensor([[1, 2], [3, 4]]),
                          old_log_probs=torch.tensor([[-2., float('nan')], [-3., -4.]]),
                          loss_mask=torch.tensor([[1, 0], [1, 1]]))
        self.non_tensor_batch = dict(binding=['a', 'b'])
        self.meta_info = dict(temperature=1., diagnosis='keep invalid diagnostic values')
    def __len__(self): return 2


def trainer(directory):
    sampler = CheckpointBatchSampler(40, 4, 42, 'tiny-data')
    iterator = iter(sampler)
    next(iterator); next(iterator)  # The failed second outer has already consumed its batch.
    return N(global_steps=2, actual_outer_updates=1, checkpoint_sampler=sampler,
        config=N(trainer=N(default_local_dir=str(directory)),
                 baselines=N(training_protocol='tree_fresh200_five_leaf_v1')),
        actor_rollout_wg=N(world_size=8), baseline_controller=N(method='tree_grpo',
            tree_rng=random.Random(42), stage=N(state_dict=lambda: {'stage': 'training'})))


def envelope_for(trainer, calls=1, scheduler_advanced=False):
    context = partial_context(trainer)
    records = []
    for rank in range(8):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda k: 1.)
        # Represent the one actually completed outer from the controller boundary.
        optimizer.zero_grad(); model(torch.ones(1, 2)).sum().backward(); optimizer.step(); scheduler.step()
        for _ in range(calls):
            optimizer.zero_grad(); model(torch.ones(1, 2)).square().sum().backward(); optimizer.step()
        if scheduler_advanced: scheduler.step()
        identity = dict(context['identity'], rank=rank, world_size=8)
        saved = worker_snapshot(model, optimizer, scheduler, identity)
        path = Path(context['directory']) / f'worker-{rank:02d}.pt'
        sha = atomic_save(saved, path)
        records.append(dict(identity=identity, rank=rank, world_size=8,
            committed_optimizer_calls=calls, scheduler_step_before_actual=1,
            scheduler_step_after_actual=scheduler.last_epoch, state_finite=True,
            status='PARTIAL_WORKER_STATE_SAVED', path=str(path), sha256=sha,
            bytes=path.stat().st_size))
    return dict(protocol=PROTOCOL, identity=context['identity'], status='PARTIAL_WORKER_STATE_SAVED',
                worker_states=records, outer_completed=False, automatic_rollback=False)


class PartialDriverTests(unittest.TestCase):
    def test_actual_partial_adam_and_consumed_batch_saved_without_outer_credit(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, ENV):
            t = trainer(tmp); batch = Batch(); e = envelope_for(t, calls=19)
            before_rng = torch.get_rng_state().clone()
            with self.assertRaises(PolicySafetyRefusal):
                stop_on_partial_actor_result(t, batch, N(meta_info={'tree_partial_state': e}))
            self.assertEqual((t.global_steps, t.actual_outer_updates, t.checkpoint_sampler.cursor), (2, 1, 8))
            self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
            import json
            m = json.loads(Path(t.partial_outer_evidence['path']).read_text())
            self.assertEqual(m['status'], 'PARTIAL_OUTER_STATE_SAVED_CONTROLLED_STOP')
            self.assertEqual(m['actual_optimizer_calls_per_rank'], {str(r): 19 for r in range(8)})
            self.assertFalse(m['outer_completed']); self.assertFalse(m['automatic_resume_supported'])
            saved = torch.load(m['driver']['path'], weights_only=False)
            self.assertEqual(saved['consumed_sampler']['cursor'], 8)
            with self.assertRaises(ValueError): restore_driver(t, saved, e['identity'])
            inp = torch.load(m['actor_inputs']['path'], weights_only=False)
            self.assertTrue(torch.equal(inp['batch']['input_ids'], batch.batch['input_ids']))
            self.assertTrue(torch.isnan(inp['batch']['old_log_probs'][0, 1]))
            for r in e['worker_states']:
                s = torch.load(r['path'], weights_only=False)
                self.assertTrue(all(v['step'].item() == 20 for v in s['optimizer']['state'].values()))
                self.assertEqual(s['scheduler']['last_epoch'], 1)
                self.assertEqual(file_sha(r['path']), r['sha256'])

    def test_saved_scheduler_actual_advance_is_reported_without_outer_credit(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, ENV):
            t = trainer(tmp); e = envelope_for(t, calls=20, scheduler_advanced=True)
            with self.assertRaises(PolicySafetyRefusal):
                stop_on_partial_actor_result(t, Batch(), N(meta_info={'tree_partial_state': e}))
            import json
            m = json.loads(Path(t.partial_outer_evidence['path']).read_text())
            self.assertEqual(m['actual_scheduler_steps_per_rank'], {str(r): 2 for r in range(8)})
            self.assertEqual(t.actual_outer_updates, 1)

    def test_missing_rank_or_tampered_file_is_incomplete_but_driver_not_lost(self):
        for damage in ('missing-rank', 'bytes'):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, ENV):
                t = trainer(tmp); e = envelope_for(t)
                if damage == 'missing-rank': e['worker_states'].pop()
                else:
                    with Path(e['worker_states'][0]['path']).open('ab') as f: f.write(b'corrupted')
                with self.assertRaises(PolicySafetyRefusal):
                    stop_on_partial_actor_result(t, Batch(), N(meta_info={'tree_partial_state': e}))
                import json
                m = json.loads(Path(t.partial_outer_evidence['path']).read_text())
                self.assertEqual(m['status'], 'PARTIAL_OUTER_STATE_INCOMPLETE_CONTROLLED_STOP')
                self.assertFalse(m['worker_states_verified'])
                self.assertTrue(Path(m['driver']['path']).is_file())
                self.assertEqual(t.actual_outer_updates, 1)

    def test_prewrite_refusal_keeps_zero_counts_and_ordinary_output_is_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, ENV):
            t = trainer(tmp); b = Batch()
            stop_on_partial_actor_result(t, b, N(meta_info={'metrics': {'actor/optimizer_steps': 20}}))
            self.assertFalse((Path(tmp) / 'training_state').exists())
            e = envelope_for(t, calls=0)
            with self.assertRaises(PolicySafetyRefusal):
                stop_on_partial_actor_result(t, b, N(meta_info={'tree_partial_state': e}))
            import json
            m = json.loads(Path(t.partial_outer_evidence['path']).read_text())
            self.assertEqual(m['actual_optimizer_calls_per_rank'], {str(r): 0 for r in range(8)})

    def test_unmatched_worker_failure_never_requests_another_collective_or_claims_saved_workers(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, ENV):
            t = trainer(tmp)
            class NoWorkerCalls:
                def __getattr__(self, name): raise AssertionError('must not call surviving workers')
            t.actor_rollout_wg = NoWorkerCalls()
            path = preserve_unmatched_actor_failure(t, Batch(), RuntimeError('one rank died'))
            import json
            m = json.loads(Path(path).read_text())
            self.assertEqual(m['status'], 'UNMATCHED_WORKER_FAILURE_DRIVER_ONLY')
            self.assertIsNone(m['actual_optimizer_calls_per_rank'])
            self.assertFalse(m['worker_states_verified'])

    def test_actual_trainer_guard_handling_precedes_reduction_and_outer_credit(self):
        source = ast.parse(Path('verl/trainer/ppo/ray_trainer.py').read_text())
        fit = next(n for n in ast.walk(source) if isinstance(n, ast.FunctionDef) and n.name == 'fit')
        calls = {n.func.id: n.lineno for n in ast.walk(fit) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                 and n.func.id in ('stop_on_partial_actor_result', 'validate_completed_outer')}
        after_actor = [n.lineno for n in ast.walk(fit) if isinstance(n, ast.Call)
                       and isinstance(n.func, ast.Attribute) and n.func.attr == 'after_actor']
        actor_reduction = [n.lineno for n in ast.walk(fit) if isinstance(n, ast.Assign)
                           and any(isinstance(t, ast.Name) and t.id == 'actor_output_metrics' for t in n.targets)]
        self.assertLess(calls['stop_on_partial_actor_result'], min(after_actor + actor_reduction))
        self.assertLess(calls['stop_on_partial_actor_result'], calls['validate_completed_outer'])


if __name__ == '__main__': unittest.main()
