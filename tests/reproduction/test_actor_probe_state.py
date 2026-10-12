import copy
import io
import random
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

from search_g1.actor_probe_sampler import RecordedRandomSampler, restored_iterator
from search_g1.actor_probe_state import (isolation_state, tree_digest, rng_state,
    worker_payload, restore_worker_payload)


class ActorProbeStateTests(unittest.TestCase):
    def test_sampler_matches_existing_shuffle_and_resumes_actual_yielded_cursor(self):
        dataset = list(range(37))
        torch.manual_seed(73)
        original = list(DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True))
        torch.manual_seed(73)
        sampler = RecordedRandomSampler(dataset)
        iterator = iter(DataLoader(dataset, batch_size=4, sampler=sampler, drop_last=True))
        prefix = [next(iterator) for _ in range(3)]
        saved = copy.deepcopy(sampler.state_dict())
        self.assertEqual(saved['cursor'], 12)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(prefix, original[:3])))
        restored = RecordedRandomSampler(dataset); restored.load_state_dict(saved)
        tail = list(DataLoader(dataset, batch_size=4, sampler=restored, drop_last=True))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(tail, original[3:])))
        with self.assertRaises(ValueError): restored.load_state_dict(saved | {'permutation': [0] * 37})

    def test_isolation_digest_catches_parameters_gradients_and_optimizer_mutation(self):
        model = torch.nn.Linear(3, 2)
        optim = torch.optim.Adam(model.parameters(), lr=.01)
        model(torch.ones(2, 3)).sum().backward(); optim.step()
        initial = isolation_state(model, optim)
        self.assertEqual(initial, isolation_state(model, optim))
        with torch.no_grad(): model.weight.grad.add_(1.)
        changed = isolation_state(model, optim)
        self.assertNotEqual(initial['gradients_sha256'], changed['gradients_sha256'])
        self.assertEqual(initial['parameters_sha256'], changed['parameters_sha256'])
        optim.state[model.weight]['exp_avg'].add_(1.)
        self.assertNotEqual(changed['optimizer_sha256'], isolation_state(model, optim)['optimizer_sha256'])
        self.assertNotEqual(tree_digest(torch.zeros(2)), tree_digest(torch.ones(2)))

    def test_restored_iterator_preserves_tail_next_epoch_and_all_driver_rng(self):
        for size, completed_batches in ((37, 3), (32, 8)):
            torch.manual_seed(432); random.seed(29); np.random.seed(92)
            dataset = list(range(size)); sampler = RecordedRandomSampler(dataset)
            loader = DataLoader(dataset, batch_size=4, sampler=sampler, drop_last=True)
            iterator = iter(loader)
            for _ in range(completed_batches): next(iterator)
            state = copy.deepcopy(sampler.state_dict()); saved_rng = rng_state()
            tail = list(iterator); next_epoch = list(loader)
            expected_rng = tree_digest(rng_state())
            restored = RecordedRandomSampler(dataset)
            other = DataLoader(dataset, batch_size=4, sampler=restored, drop_last=True)
            resumed_tail = list(restored_iterator(other, state, saved_rng))
            resumed_next = list(other)
            self.assertEqual(len(tail), len(resumed_tail))
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(tail, resumed_tail)))
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(next_epoch, resumed_next)))
            self.assertEqual(expected_rng, tree_digest(rng_state()))

    def test_worker_load_restores_moments_counters_scheduler_and_both_rollout_rng(self):
        from reproduction.common8.scheduler import fixed_outer_warmup_scheduler
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
        scheduler = fixed_outer_warmup_scheduler(optimizer, warmup_steps=57)
        for _ in range(5):
            optimizer.zero_grad(); model(torch.ones(2, 3)).sum().backward()
            optimizer.step(); scheduler.step()
        evidence = SimpleNamespace(completed_outer=5, pending=None, prepared=None, rejected=False,
            actual_writes=5, last_attempt_index=5, used_attempt_ids={str(i) for i in range(5)})
        worker = SimpleNamespace(rank=0, actor_module_fsdp=model, actor_optimizer=optimizer,
            actor_lr_scheduler=scheduler, actor=SimpleNamespace(_optimizer_writes=5),
            _outer_scheduler_evidence=evidence,
            rollout_sharding_manager=SimpleNamespace(torch_random_states=torch.get_rng_state(),
                                                     gen_random_states=torch.get_rng_state()))
        run = dict(run_id='a'*32, execution_id='b'*32, source_commit='c'*40,
            execution_backend='ssh_local', training_world_size=4, profile='common8', slurm_job_id=None)
        serialized = io.BytesIO(); torch.save(worker_payload(worker, run, 5), serialized)
        serialized.seek(0)
        saved = torch.load(serialized, map_location='cpu', weights_only=False)
        expected = tree_digest(saved)
        parameters = isolation_state(model)
        optimizer.state[model.weight]['exp_avg'].add_(2)
        scheduler.last_epoch = 99; evidence.actual_writes = 777; evidence.completed_outer = 99
        worker.actor._optimizer_writes = 777
        worker.rollout_sharding_manager.gen_random_states = torch.zeros_like(torch.get_rng_state())
        torch.rand(4); random.random(); np.random.rand(2)
        restore_worker_payload(worker, run, saved)
        self.assertEqual(expected, tree_digest(worker_payload(worker, run, 5)))
        self.assertEqual(parameters, isolation_state(model))
        wrong = copy.deepcopy(saved); wrong['flat_parameter_layout'][0]['shape'] = [99]
        before = isolation_state(model, optimizer, scheduler)
        with self.assertRaises(AssertionError): restore_worker_payload(worker, run, wrong)
        self.assertEqual(before, isolation_state(model, optimizer, scheduler))
