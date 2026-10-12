import copy
import unittest
import torch
from torch.utils.data import DataLoader, RandomSampler

from search_g1.actor_probe_sampler import RecordedRandomSampler
from search_g1.actor_probe_state import isolation_state, tree_digest


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
