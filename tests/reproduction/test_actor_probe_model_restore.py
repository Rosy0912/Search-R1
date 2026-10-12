"""CPU recovery mechanics only; no model files or production updates are created."""
import copy
import io
import random
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from reproduction.common8.scheduler import fixed_outer_warmup_scheduler
from search_g1.actor_probe_model_state import validate_parameter_state
from search_g1.actor_probe_state import worker_payload, restore_worker_payload, tree_digest


class ActorModelRestoreTests(unittest.TestCase):
    def make_worker(self):
        module = torch.nn.Sequential(torch.nn.Dropout(.2), torch.nn.Linear(3, 2))
        optimizer = torch.optim.Adam(module.parameters(), lr=1e-6)
        scheduler = fixed_outer_warmup_scheduler(optimizer, warmup_steps=57)
        return SimpleNamespace(rank=0, actor_module_fsdp=module, actor_optimizer=optimizer,
            actor_lr_scheduler=scheduler, actor=SimpleNamespace(_optimizer_writes=0,
                _complete_outers=0, _last_attempt_index=0, _used_attempt_ids=set(), _policy_rejections=0),
            _outer_scheduler_evidence=SimpleNamespace(completed_outer=0, actual_writes=0,
                last_attempt_index=0, used_attempt_ids=set(), pending=None, prepared=None, rejected=False),
            rollout_sharding_manager=SimpleNamespace(torch_random_states=torch.get_rng_state(),
                                                     gen_random_states=torch.get_rng_state()))

    def one_update(self, worker):
        # Deliberately make the next loss depend on all CPU RNG streams, as well
        # as the separately saved generation stream and nonzero Adam moments.
        generation = torch.Generator(); generation.set_state(worker.rollout_sharding_manager.gen_random_states)
        offset = torch.rand(1, generator=generation) + random.random() + np.random.rand()
        worker.rollout_sharding_manager.gen_random_states = generation.get_state()
        x = torch.randn(8, 3) + offset
        worker.actor_optimizer.zero_grad()
        loss = worker.actor_module_fsdp(x).square().mean()
        loss.backward(); worker.actor_optimizer.step(); worker.actor_lr_scheduler.step()
        worker.actor._optimizer_writes += 1
        worker.actor._complete_outers += 1; worker.actor._last_attempt_index += 1
        worker.actor._used_attempt_ids.add(str(worker.actor._last_attempt_index))
        evidence = worker._outer_scheduler_evidence
        evidence.actual_writes = worker.actor._optimizer_writes
        evidence.completed_outer = worker.actor._complete_outers
        evidence.last_attempt_index = worker.actor._last_attempt_index
        evidence.used_attempt_ids = set(worker.actor._used_attempt_ids)
        worker.rollout_sharding_manager.torch_random_states = torch.get_rng_state()
        return loss.detach()

    def test_new_worker_model_and_state_restore_reproduces_next_actual_cpu_adam_update(self):
        torch.manual_seed(391); random.seed(51); np.random.seed(28)
        run = dict(run_id='a'*32, execution_id='b'*32, source_commit='c'*40,
            execution_backend='ssh_local', training_world_size=4, profile='common8', slurm_job_id=None)
        original = self.make_worker()
        for _ in range(5): self.one_update(original)
        model = copy.deepcopy(original.actor_module_fsdp.state_dict())
        serialized = io.BytesIO(); torch.save(worker_payload(original, run, 5), serialized)
        serialized.seek(0); payload = torch.load(serialized, map_location='cpu', weights_only=False)
        expected_loss = self.one_update(original)
        expected_parameters = tree_digest(original.actor_module_fsdp.state_dict())
        expected_state = tree_digest(worker_payload(original, run, 6))
        restored = self.make_worker()
        shapes = {name: tensor.shape for name, tensor in restored.actor_module_fsdp.state_dict().items()}
        validate_parameter_state(model, shapes)
        restored.actor_module_fsdp.load_state_dict(model, strict=True)
        restore_worker_payload(restored, run, payload)
        actual_loss = self.one_update(restored)
        self.assertTrue(torch.equal(actual_loss, expected_loss))
        self.assertEqual(tree_digest(restored.actor_module_fsdp.state_dict()), expected_parameters)
        self.assertEqual(tree_digest(worker_payload(restored, run, 6)), expected_state)

    def test_bad_weights_are_rejected_before_any_model_mutation(self):
        worker = self.make_worker(); model = worker.actor_module_fsdp
        good = copy.deepcopy(model.state_dict()); shapes = {name: tensor.shape for name, tensor in good.items()}
        before = tree_digest(model.state_dict())
        bad = copy.deepcopy(good); bad['1.weight'][0, 0] = float('nan')
        for candidate in (bad, good | {'unexpected': torch.zeros(1)},
                          good | {'1.weight': good['1.weight'].bfloat16()}):
            with self.assertRaises(AssertionError): validate_parameter_state(candidate, shapes)
            self.assertEqual(tree_digest(model.state_dict()), before)


if __name__ == '__main__': unittest.main()
