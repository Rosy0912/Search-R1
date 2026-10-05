"""CPU hook/schema regressions; not native CUDA precision evidence."""
import copy
from types import SimpleNamespace as NS
import unittest
import torch
from reproduction.common8.precision_evidence import PrecisionObservation, verify_precision


def fixture_proof():
    common = dict(forward_calls=1, logit_dtypes=['torch.float32'], activation_dtypes=['torch.float32'],
        parameter_dtypes=['torch.float32'], compute_parameter_dtypes=['torch.float32'],
        buffer_dtypes=['torch.float32'], autocast_states=[False])
    actor = dict(common, **{k: 'torch.float32' for k in
        ('mixed_parameter_dtype', 'mixed_reduce_dtype', 'mixed_buffer_dtype')})
    ref = dict(common, **{k: 'None' for k in
        ('mixed_parameter_dtype', 'mixed_reduce_dtype', 'mixed_buffer_dtype')})
    return dict(roles=dict(actor=actor, reference=ref), torch_matmul_allow_tf32=False,
        torch_cudnn_allow_tf32=False, rollout=dict(forward_calls=1, logits_calls=1,
        cache_tensor_count=1, parameter_dtypes=['torch.float32'], hidden_dtypes=['torch.float32'],
        cache_dtypes=['torch.float32'], logit_dtypes=['torch.float32'], autocast_states=[False],
        attention_backend='CPU.fixture.only'))


class HF(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.embed = torch.nn.Embedding(8, 4)
        self.proj = torch.nn.Linear(4, 4)
        self.register_buffer('frequency', torch.ones(2))
    def get_input_embeddings(self):
        return self.embed
    def forward(self, ids):
        return NS(logits=self.proj(self.embed(ids)))


class Wrapped(torch.nn.Module):
    def __init__(self, reference=False):
        super().__init__(); self.module = HF()
        v = None if reference else torch.float32
        self.mixed_precision = NS(param_dtype=v, reduce_dtype=v, buffer_dtype=v)
    def forward(self, ids):
        return self.module(ids)


class EngineModel(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.proj = torch.nn.Linear(4, 4)
        self.model = NS(layers=[NS(self_attn=NS(attn=NS(impl=object())))])
    def forward(self, *, input_ids, positions, kv_caches, attn_metadata):
        return self.proj(input_ids)
    def compute_logits(self, hidden, metadata):
        return self.proj(hidden)


class PrecisionTests(unittest.TestCase):
    def test_precision_rejects_missing_bf16_or_unobserved_cache(self):
        proof = fixture_proof(); self.assertTrue(verify_precision(proof))
        for key, value in (('cache_dtypes', ['torch.bfloat16']), ('cache_tensor_count', 0),
                           ('logits_calls', 0), ('autocast_states', [True]), ('attention_backend', None)):
            bad = copy.deepcopy(proof); bad['rollout'][key] = value
            with self.assertRaises(AssertionError): verify_precision(bad)
        bad = copy.deepcopy(proof); bad['roles']['reference']['logit_dtypes'] = []
        with self.assertRaises(AssertionError): verify_precision(bad)
        bad = copy.deepcopy(proof); bad['torch_matmul_allow_tf32'] = 'False'
        with self.assertRaises(AssertionError): verify_precision(bad)

    def test_actual_cpu_hooks_observe_outputs_and_are_removed(self):
        self._observe_fixture(False)

    def test_unregistered_live_fsdp_tensor_views_are_observed(self):
        self._observe_fixture(True)

    def _observe_fixture(self, live_views):
        a, r, v = Wrapped(), Wrapped(True), EngineModel()
        if live_views:
            for layer in (a.module.proj, r.module.proj):
                for name in ('weight', 'bias'):
                    value = getattr(layer, name).detach()
                    del layer._parameters[name]
                    setattr(layer, name, value)
                self.assertEqual(list(layer.parameters(recurse=False)), [])
        actor = NS(actor_module_fsdp=a, rollout=NS(inference_engine=NS(llm_engine=NS(
            model_executor=NS(worker=NS(model_runner=NS(model=v)))))))
        reference = NS(ref_module_fsdp=r)
        original = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        observation = PrecisionObservation(actor, reference)
        try:
            ids = torch.tensor([[1, 2]]); a(ids); r(ids)
            x = torch.ones(2, 4); before = v.proj(x)
            hidden = v(input_ids=x, positions=torch.arange(2), kv_caches=[torch.zeros(2, 4)], attn_metadata=None)
            self.assertTrue(torch.equal(before, hidden))
            self.assertTrue(torch.equal(v.proj(hidden), v.compute_logits(hidden, None)))
            self.assertTrue(verify_precision(observation.snapshot()))
            v(input_ids=x, positions=None, kv_caches=[torch.zeros(2, 4, dtype=torch.bfloat16)], attn_metadata=None)
            with self.assertRaises(AssertionError): verify_precision(observation.snapshot())
            snapshot = observation.snapshot(); observation.close()
            a(ids); r(ids); v(input_ids=x, positions=None, kv_caches=[], attn_metadata=None)
            v.compute_logits(hidden, None)
            self.assertEqual(observation.snapshot(), snapshot)
        finally:
            observation.close()
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = original


if __name__ == '__main__': unittest.main()
