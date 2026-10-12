import copy
import random
import unittest
import numpy as np
import torch
from search_g1.actor_probe_features import detached_last_token_features


class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.linear = torch.nn.Linear(4, 4); self.drop = torch.nn.Dropout(.1)

    def forward(self, hidden): return (self.drop(self.linear(hidden)),)


class Actor(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.model = torch.nn.Module()
        self.model.embed = torch.nn.Embedding(8, 4)
        self.model.layers = torch.nn.ModuleList([Layer(), Layer()])
        self.lm_head = torch.nn.Linear(4, 8)

    def forward(self, input_ids, attention_mask, use_cache, **kwargs):
        h = self.model.embed(input_ids)
        for layer in self.model.layers: h = layer(h)[0]
        return self.lm_head(h)


class ActorFeatureTests(unittest.TestCase):
    def test_stochastic_readout_and_failure_preserve_all_cpu_rng_streams(self):
        from search_g1.actor_probe_state import rng_state, tree_digest
        class StochasticActor(Actor):
            def forward(self,*args,**kwargs):
                random.random(); np.random.random(); torch.rand(3)
                return super().forward(*args,**kwargs)
        actor=StochasticActor().train();ids=torch.tensor([[1,2],[2,3]]);mask=torch.ones_like(ids)
        before=tree_digest(rng_state())
        detached_last_token_features(actor,ids,mask,(1,2),micro_batch_size=1)
        self.assertEqual(before,tree_digest(rng_state()))
        with torch.no_grad():actor.model.layers[1].linear.bias.fill_(float('nan'))
        with self.assertRaises(ValueError):detached_last_token_features(actor,ids,mask,(1,2))
        self.assertEqual(before,tree_digest(rng_state()))
        self.assertTrue(actor.training)
        self.assertTrue(all(not layer._forward_hooks for layer in actor.model.layers))

    def test_head_learning_preserves_actor_generation_reference_adam_and_gradients(self):
        torch.manual_seed(4); actor = Actor().train(); actor.model.layers[0].eval()
        reference = copy.deepcopy(actor); adam = torch.optim.Adam(actor.parameters(), lr=.01)
        ids = torch.tensor([[0, 1, 2], [3, 2, 1], [1, 2, 3]]); mask = torch.ones_like(ids)
        actor(ids, mask, False).sum().backward(); adam.step()
        saved = {k: v.clone() for k, v in actor.state_dict().items()}
        grads = {k: p.grad.clone() for k, p in actor.named_parameters()}
        state = copy.deepcopy(adam.state_dict()); ref = copy.deepcopy(reference.state_dict())
        modes = [m.training for m in actor.modules()]
        features = detached_last_token_features(actor, ids, mask, (1, 2), micro_batch_size=2)
        self.assertEqual(features[2].shape, (3, 4)); self.assertFalse(features[2].requires_grad)
        self.assertIsNone(features[2].grad_fn)
        head = torch.nn.Linear(4, 2); before = head.weight.clone()
        head_optim = torch.optim.Adam(head.parameters(), lr=.01)
        torch.nn.functional.cross_entropy(head(features[2]), torch.tensor([0, 1, 0])).backward()
        head_optim.step(); self.assertFalse(torch.equal(before, head.weight))
        for key, value in actor.state_dict().items(): self.assertTrue(torch.equal(value, saved[key]), key)
        for key, param in actor.named_parameters(): self.assertTrue(torch.equal(param.grad, grads[key]), key)
        for key, value in reference.state_dict().items(): self.assertTrue(torch.equal(value, ref[key]), key)
        for param, values in adam.state_dict()['state'].items():
            for key, value in values.items(): self.assertTrue(torch.equal(value, state['state'][param][key]))
        self.assertEqual([m.training for m in actor.modules()], modes)
        for layer in actor.model.layers: self.assertFalse(layer._forward_hooks)

    def test_current_actor_changes_are_visible_and_failure_restores_modes_and_hooks(self):
        actor = Actor().train(); ids = torch.tensor([[1, 2]]); mask = torch.ones_like(ids)
        before = detached_last_token_features(actor, ids, mask, (1, 2))[2]
        with torch.no_grad(): actor.model.layers[1].linear.bias.add_(1.)
        after = detached_last_token_features(actor, ids, mask, (1, 2))[2]
        self.assertFalse(torch.equal(before, after))
        with torch.no_grad(): actor.model.layers[1].linear.bias.fill_(float('nan'))
        with self.assertRaises(ValueError): detached_last_token_features(actor, ids, mask, (1, 2))
        self.assertTrue(actor.training)
        for layer in actor.model.layers: self.assertFalse(layer._forward_hooks)
