"""RAM-only namespace rebinding, stochastic Adam and data/reward continuation."""
import copy
import random
import socket
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from reproduction.common8.recoverable_attempt import AttemptProgress
from reproduction.common8.verify_data import MANIFEST_SHA
from search_g1.actor_probe_contract import MODE
from search_g1.actor_probe_recovery import (validate_recovery, rebound_worker_payload,
    rebound_driver_payload, rebound_probe_manifest, apply_driver_state, identity)
from search_g1.actor_probe_sampler import RecordedRandomSampler
from search_g1.actor_probe_state import worker_payload, restore_worker_payload, tree_digest, rng_state, validate_rng
from tests.reproduction import test_actor_probe_model_restore as model_restore


def documents():
    parent = dict(run_id='a'*32, execution_id='b'*32, source_commit='c'*40,
        execution_backend='ssh_local', execution_host=socket.gethostname(), profile='common8', slurm_job_id=None,
        input_sha256={'manifest.json': MANIFEST_SHA, 'calibration-300.parquet': '1'*64},
        initial_model_sha256={'config.json': '2'*64},
        model_revision='aa8e72537993ba99e69dfaafa59ed015b17504d1',
        initial_model='/data/unit/official', reference_model='/data/unit/official',
        budget={'world_size': 4, 'lr_warmup_steps': 57}, training_world_size=4,
        training_physical_indices=[0,1,2,3], auxiliary_physical_indices=[5,6],
        excluded_physical_indices=[4,7], reward_schedule='g1_through100_then_binary',
        reward_schedule_boundary=100, probe_representation_source=MODE,
        probe_refresh_interval=10, probe_parameter_ownership='independent_binary_heads_only',
        policy_safety_mode='inline_first_five', contract_kind='fresh200_warmup57',
        target_updates=200, predecessor_completion={'run_id': '9'*32},
        vllm_gpu_memory_utilization=.25)
    child = copy.deepcopy(parent); child.update(run_id='d'*32, execution_id='e'*32,
        source_commit='f'*40, actor_probe_experiment_root_run_id=parent['run_id'])
    binding = dict(role='actor_probe_same_experiment_full_state_recovery',
        state_semantics='full_model_optimizer_scheduler_rng_sampler_and_heads',
        parent_run_id=parent['run_id'], parent_execution_id=parent['execution_id'],
        parent_source_commit=parent['source_commit'], experiment_root_run_id=parent['run_id'],
        completed_outer=5, parent_manifest_sha256='3'*64, full_state_receipt_sha256='4'*64)
    receipt = dict(role='actor_probe_full_state_checkpoint', identity=identity(parent),
        completed_outer=5, next_logical_update=6, actual_optimizer_writes=5,
        workers=[dict(rank=r, completed_outer=5, optimizer_writes=5) for r in range(4)],
        probe_version=parent['run_id']+':0:unit', probe_actor_step=0,
        probe_head_hashes={'commit_probe.pkl':'5'*64, 'boundary_probe.pkl':'6'*64})
    return parent, child, binding, receipt


class ActorProbeRecoveryTests(unittest.TestCase):
    def test_cross_namespace_restore_reproduces_next_stochastic_adam_without_mutating_parent(self):
        parent, child, binding, receipt = documents()
        mechanics = model_restore.ActorModelRestoreTests()
        torch.manual_seed(614); random.seed(96); np.random.seed(12)
        original = mechanics.make_worker()
        for _ in range(5): mechanics.one_update(original)
        payload = copy.deepcopy(worker_payload(original, parent, 5))
        model_state = copy.deepcopy(original.actor_module_fsdp.state_dict())
        parent_sha = tree_digest(payload)
        expected_loss = mechanics.one_update(original)
        expected_model = tree_digest(original.actor_module_fsdp.state_dict())
        expected_state = worker_payload(original, parent, 6)
        restored = mechanics.make_worker()
        rebound = rebound_worker_payload(payload, binding, child, parent, receipt, 0)
        self.assertEqual(tree_digest(payload), parent_sha)
        restored.actor_module_fsdp.load_state_dict(model_state, strict=True)
        restore_worker_payload(restored, child, rebound)
        self.assertTrue(torch.equal(mechanics.one_update(restored), expected_loss))
        self.assertEqual(tree_digest(restored.actor_module_fsdp.state_dict()), expected_model)
        actual = worker_payload(restored, child, 6)
        self.assertEqual(tree_digest(actual), tree_digest(dict(expected_state, identity=identity(child))))

    def test_wrong_experiment_recipe_or_rank_set_is_rejected_before_mutation(self):
        parent, child, binding, receipt = documents()
        before = tree_digest((parent, child, binding, receipt))
        cases = [(dict(child, actor_probe_experiment_root_run_id='8'*32), parent, receipt),
                 (dict(child, initial_model='/data/unit/other', reference_model='/data/unit/other'), parent, receipt),
                 (dict(child, execution_host='another-host'), parent, receipt),
                 (dict(child, budget={'world_size':4, 'lr_warmup_steps':0}), parent, receipt),
                 (dict(child, input_sha256={'manifest.json':'0'*64}), parent, receipt),
                 (child, dict(parent, reward_schedule='g1_through50_then_binary'), receipt),
                 (child, parent, dict(receipt, workers=receipt['workers'][:3]))]
        for wrong_child, wrong_parent, wrong_receipt in cases:
            with self.assertRaises((AssertionError, ValueError)):
                validate_recovery(binding, wrong_child, wrong_parent, wrong_receipt, rank=3)
            self.assertEqual(tree_digest((parent, child, binding, receipt)), before)

    def test_head_rebinding_preserves_version_fitting_step_and_both_head_hashes(self):
        parent, child, binding, receipt = documents()
        manifest = dict(run_id=parent['run_id'], version=receipt['probe_version'], round=0,
            actor_step=0, representation_source=MODE, calibration_data_sha256='1'*64,
            head_sha256=receipt['probe_head_hashes'], commit='/data/parent/commit_probe.pkl',
            boundary='/data/parent/boundary_probe.pkl')
        before = copy.deepcopy(manifest)
        result = rebound_probe_manifest(manifest, binding, child, parent, receipt,
            {'commit':'/data/child/commit_probe.pkl','boundary':'/data/child/boundary_probe.pkl'})
        self.assertEqual(manifest, before)
        self.assertEqual(result['version'], manifest['version'])
        self.assertEqual(result['head_sha256'], manifest['head_sha256'])
        self.assertEqual(result['run_id'], child['run_id'])
        with self.assertRaises(ValueError):
            rebound_probe_manifest(dict(manifest, calibration_data_sha256='7'*64), binding,
                child, parent, receipt, {'commit':'a','boundary':'b'})

    def test_missing_gpu_rng_is_rejected_without_creating_cuda_contexts(self):
        state = rng_state(); before = tree_digest(state)
        with patch('torch.cuda.is_initialized', return_value=True):
            with self.assertRaises(AssertionError): validate_rng(state)
        self.assertEqual(tree_digest(rng_state()), before)

    def test_driver_rebinding_preserves_tail_next_epoch_reward_and_rng(self):
        parent, child, binding, receipt = documents()
        torch.manual_seed(85); random.seed(12); np.random.seed(49)
        dataset = list(range(37)); sampler = RecordedRandomSampler(dataset)
        loader = DataLoader(dataset, batch_size=4, sampler=sampler, drop_last=True)
        iterator = iter(loader)
        for _ in range(3): next(iterator)
        payload = dict(role='actor_probe_full_driver_state', identity=identity(parent),
            completed_outer=5, next_logical_update=6, epoch=0, consumed_batch_index=2,
            sampler=sampler.state_dict(), rng=rng_state(), probe_version=receipt['probe_version'],
            probe_actor_step=0, attempt_progress=dict(world_size=4, pending=None, completed_outer=5,
                cumulative_writes=5, attempt_index=5, aborted_attempts=0),
            reward_state=dict(alpha=.3, ema_em=.4, ema_trust=.6, _init=True, _calls=5))
        before = tree_digest(payload)
        expected_tail = list(iterator); expected_epoch = list(loader)
        expected_rng = tree_digest(rng_state())
        other = DataLoader(dataset, batch_size=4, sampler=RecordedRandomSampler(dataset), drop_last=True)
        trainer = SimpleNamespace(global_steps=0, train_dataloader=other, reward_fn=SimpleNamespace())
        attempts = AttemptProgress(child)
        rebound = rebound_driver_payload(payload, binding, child, parent, receipt)
        result = apply_driver_state(trainer, attempts, rebound, child)
        self.assertEqual(tree_digest(payload), before)
        actual_tail = list(result['iterator']); actual_epoch = list(other)
        self.assertEqual(len(actual_tail), len(expected_tail))
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(actual_tail, expected_tail)))
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(actual_epoch, expected_epoch)))
        self.assertEqual(tree_digest(rng_state()), expected_rng)
        self.assertEqual((trainer.global_steps, attempts.cumulative_writes, result['next_logical_update']), (5,5,6))
        self.assertEqual(vars(trainer.reward_fn), dict(payload['reward_state'], expected_probe_version=receipt['probe_version']))
        broken = copy.deepcopy(rebound); broken['sampler']['permutation'][1] = broken['sampler']['permutation'][0]
        fresh = SimpleNamespace(global_steps=0, train_dataloader=other, reward_fn=SimpleNamespace())
        progress = AttemptProgress(child)
        with self.assertRaises(ValueError): apply_driver_state(fresh, progress, broken, child)
        self.assertEqual((fresh.global_steps, progress.cumulative_writes, vars(fresh.reward_fn)), (0,0,{}))
        invalid_rng = copy.deepcopy(rebound); invalid_rng['rng']['python'] = ('broken',)
        with self.assertRaises(ValueError): apply_driver_state(fresh, progress, invalid_rng, child)
        self.assertEqual((fresh.global_steps, progress.cumulative_writes, vars(fresh.reward_fn)), (0,0,{}))


if __name__ == '__main__': unittest.main()
