"""Tiny CPU hook/geometry fixtures; no installed GPU backend or model assets."""
import ast
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from native_boundary_observer import (NativeBoundaryObserver, build_boundary_modules,
                                      compare_boundary_observers, fixed_anchors)

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = ROOT / 'work/tree-vllm-window-runner'


def source_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


capture_core = source_module('accepted_response_map_cpu_fixture',
                            ORIGINAL / 'search_r1/baselines/probability_probe_capture.py')
window_core = source_module('accepted_window_mapping_cpu_fixture', ORIGINAL / 'window_observer.py')
CONFIG = dict(num_hidden_layers=36, hidden_size=2048)
PROVENANCE = dict(scope='SYNTHETIC_CPU_FIXTURE_NOT_ACTUAL_GPU_SOURCE_PROOF')


class Embedding(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.calls = 0; self.last = None

    def forward(self, ids):
        self.calls += 1
        self.last = ids.to(torch.bfloat16).unsqueeze(-1).expand(*ids.shape, 2048).clone()
        return self.last


class Norm(torch.nn.Module):
    def __init__(self, tuple_output=False, shift=0.):
        super().__init__(); self.calls = 0; self.tuple_output = tuple_output; self.shift = shift
        self.bad_shape = False; self.nonfinite = False; self.raise_native = False

    def forward(self, value, residual=None):
        self.calls += 1
        if self.raise_native:
            raise RuntimeError('original native fixture exception')
        if self.shift:
            value = value + self.shift
        if self.bad_shape:
            value = value[..., :1]
        if self.nonfinite:
            value = value.clone(); value[..., 0] = float('nan')
        return (value, residual if residual is not None else value) if self.tuple_output else value


class Core(torch.nn.Module):
    def __init__(self, backend):
        super().__init__(); self.backend = backend
        self.embed_tokens = Embedding(); self.layers = torch.nn.ModuleList()
        for index in range(36):
            layer = torch.nn.Module()
            layer.input_layernorm = Norm(backend == 'vllm' and index != 0)
            layer.post_attention_layernorm = Norm(backend == 'vllm')
            self.layers.append(layer)
        self.norm = Norm(backend == 'vllm')

    def forward(self, ids):
        value = self.embed_tokens(ids); residual = value
        for layer in self.layers:
            result = layer.input_layernorm(value, residual)
            value, residual = result if type(result) is tuple else (result, residual)
            result = layer.post_attention_layernorm(value, residual)
            value, residual = result if type(result) is tuple else (result, residual)
        result = self.norm(value, residual)
        return result[0] if type(result) is tuple else result


class ActorRoot(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.model = Core('actor'); self.calls = 0; self.last = None

    def forward(self, *, input_ids, position_ids, attention_mask, use_cache):
        self.calls += 1; self.last = self.model(input_ids)
        return self.last


def observer(backend, rank=0, root=None):
    model = root.model if backend == 'actor' else root
    return NativeBoundaryObserver(backend, rank, fixed_anchors(), build_boundary_modules(model, CONFIG),
        CONFIG, SimpleNamespace(**CONFIG), PROVENANCE, root_module=root if backend == 'actor' else None)


def micro(offset=0):
    response_width = 20
    responses = torch.zeros((4, response_width), dtype=torch.int64)
    for row in range(4):
        responses[row, :offset+1] = torch.arange(7, 8+offset)
    ids = torch.cat((torch.tensor([[0, 2, 3]]*4), responses), dim=1)
    attention = torch.zeros_like(ids); attention[:, 1:3+offset+1] = 1
    positions = torch.zeros_like(ids)
    positions[:, 1:3+offset+1] = torch.arange(2+offset+1)
    return dict(responses=responses, input_ids=ids, attention_mask=attention, position_ids=positions)


def mapping(data, packed):
    diagnostic = dict(data, loss_mask=torch.ones_like(data['responses'], dtype=torch.bool))
    return capture_core.response_prediction_map(diagnostic, packed)


def invoke_actor(root, obs, data, packed=True, start=0, call=0, change_positions=False):
    m = mapping(data, packed)
    obs.prepare_actor_call(data, m, start, call)
    valid = data['attention_mask'].bool()
    ids = data['input_ids'][valid].unsqueeze(0) if packed else data['input_ids']
    positions = data['position_ids'][valid].unsqueeze(0) if packed else data['position_ids']
    if change_positions:
        positions = positions + 1
    return root(input_ids=ids, position_ids=positions,
                attention_mask=None if packed else data['attention_mask'], use_cache=False)


def outcomes_for_actor(obs, absent_second=False):
    result = []
    for anchor in obs.anchors:
        point = obs.points.get(anchor['id'])
        if point is None and absent_second:
            result.append(dict(anchor, status='absent_native_early_finish', request_id='request160',
                               true_response_length=8, finish_reason='stop'))
        elif point is None:
            result.append(dict(anchor, status='observed', request_id='unobserved',
                context_token_ids=[2, 3], context_positions=[0, 1], target_token_id=7))
        else:
            result.append(dict(anchor, status='observed', request_id='request%d' % anchor['input_row'],
                context_token_ids=point['context_token_ids'], context_positions=point['context_positions'],
                target_token_id=point['target_token_id']))
    return result


def actual_vllm_mapping(rank=0, row=0, offset=0):
    """Use original WindowObserver geometry, not a hand-made hidden index."""
    prompts = [[2, 3] for _ in range(max(2, row+1))]
    core = window_core.WindowObserver(None, {}, fixed_anchors(), rank)
    core.observe_prompts(prompts)
    core.register_request('request%d' % row, prompts[row], row)
    other = 1 if row == 0 else 0
    core.register_request('other%d' % other, prompts[other], other)
    prior = list(range(7, 7+offset))
    contexts = [prompts[row]+prior, prompts[other]]
    ids = torch.tensor(contexts[0]+contexts[1], dtype=torch.int64)
    positions = torch.tensor(list(range(len(contexts[0]))) + list(range(len(contexts[1]))))
    settings = SimpleNamespace(n=1, temperature=1., top_p=1., top_k=-1, min_p=0.,
        presence_penalty=0., frequency_penalty=0., repetition_penalty=1., min_tokens=0,
        max_tokens=500, logprobs=1, prompt_logprobs=None, seed=None, logits_processors=None)
    group = SimpleNamespace(seq_ids=[101, 102], sample_indices=[0, 1], prompt_logprob_indices=[],
        sampling_params=settings, is_prompt=offset == 0,
        seq_data={101:SimpleNamespace(prompt_token_ids=prompts[row], output_token_ids=prior),
                  102:SimpleNamespace(prompt_token_ids=prompts[other], output_token_ids=[])})
    metadata = SimpleNamespace(selected_token_indices=torch.tensor([len(contexts[0])-1, len(ids)-1]),
                               seq_groups=[group], skip_sampler_cpu_output=False)
    model_input = SimpleNamespace(input_tokens=ids, input_positions=positions,
        request_ids_to_seq_ids={'request%d'%row:[101], 'other%d'%other:[102]},
        query_lens=[len(contexts[0]), len(contexts[1])], sampling_metadata=metadata)
    # Decode offset17 needs the actual prefix ledger from preceding native calls.
    if offset:
        core._context_input_ledger[101] = dict(request_id='request%d'%row,
            inputs={i:(token,i) for i,token in enumerate(contexts[0])})
        group.is_prompt = True  # Full original prefill fixture of this exact prefix.
    core.observe_model_input(model_input)
    if core.errors:
        raise AssertionError(core.errors)
    anchor = next(a for a in fixed_anchors() if a['rank'] == rank and a['input_row'] == row
                  and a['response_offset'] == offset)
    point = core._step['rows'][0]
    outcome = dict(anchor, status='observed', request_id='request%d'%row,
        context_token_ids=point['context_token_ids'], context_positions=point['context_positions'],
        target_token_id=7+offset)
    return core._step, ids, outcome


class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.assertFalse(torch.cuda.is_initialized())

    def test_original_anchor_function_exact(self):
        tree = ast.parse((ORIGINAL/'window_contracts.py').read_text())
        method = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'anchors')
        module = ast.Module(body=[method], type_ignores=[]); namespace = {}
        exec(compile(module, 'original_window_contracts_anchors_only', 'exec'), namespace)
        self.assertEqual(fixed_anchors(), namespace['anchors']())

    def test_actual_config_object_and_mismatch(self):
        model = Core('vllm'); modules = build_boundary_modules(model, CONFIG)
        NativeBoundaryObserver('vllm', 0, fixed_anchors(), modules, CONFIG, SimpleNamespace(**CONFIG), PROVENANCE)
        with self.assertRaisesRegex(ValueError, 'Actual configuration differs'):
            NativeBoundaryObserver('vllm', 0, fixed_anchors(), modules, CONFIG,
                                   dict(CONFIG, hidden_size=1024), PROVENANCE)

    def test_bool_rank_cannot_alias_original_anchor_zero(self):
        model = Core('vllm'); anchors = fixed_anchors(); anchors[0]['rank'] = False
        with self.assertRaisesRegex(ValueError, 'anchor plan changed'):
            NativeBoundaryObserver('vllm', 0, anchors, build_boundary_modules(model, CONFIG),
                                   CONFIG, CONFIG, PROVENANCE)

    def test_module_count_and_unique_refs(self):
        model = Core('vllm'); model.layers = model.layers[:-1]
        with self.assertRaisesRegex(ValueError, 'module count'):
            build_boundary_modules(model, CONFIG)
        model = Core('vllm'); model.norm = model.layers[0].input_layernorm
        with self.assertRaisesRegex(ValueError, 'aliased'):
            build_boundary_modules(model, CONFIG)

    def test_vllm_tensor_tuple_hidden_index_and_native_identity(self):
        model = Core('vllm'); obs = observer('vllm', root=model)
        step, ids, outcome = actual_vllm_mapping()
        saved = ids.clone()
        with obs:
            obs.observe_vllm_step(step); returned = model(ids)
            self.assertIs(returned, model.embed_tokens.last)
            self.assertEqual(float(obs.vectors['rank00-first']['embedding'][0]), 3.)
            self.assertNotEqual(step['rows'][0]['hidden_state_index'], 0)
            obs.check_errors(); obs.clear_step()
        receipt = obs.finalize([outcome])
        self.assertTrue(receipt['observer_evidence_complete']); self.assertEqual(receipt['captured_vector_count'], 74)
        self.assertTrue(torch.equal(saved, ids)); self.assertTrue(all(m.calls == 1 for m in obs.modules.values()))

    def test_packed_predecessor_actual_call_and_inputs_untouched(self):
        root = ActorRoot(); obs = observer('actor', root=root); data = micro()
        saved = {k:v.clone() for k,v in data.items()}; refs = dict(data)
        before_rng = torch.get_rng_state().clone()
        with obs:
            result = invoke_actor(root, obs, data); self.assertIs(result, root.last)
            self.assertEqual(float(obs.vectors['rank00-first']['embedding'][0]), 3.)
            self.assertEqual(obs.points['rank00-first']['context_positions'], [0, 1])
            obs.check_errors(); obs.clear_step()
        self.assertTrue(obs.finalize(outcomes_for_actor(obs))['observer_evidence_complete'])
        self.assertEqual(root.calls, 1); self.assertTrue(all(m.calls == 1 for m in obs.modules.values()))
        self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
        self.assertTrue(all(data[k] is refs[k] and torch.equal(data[k], saved[k]) for k in data))

    def test_dense_left_padding_same_causal_token(self):
        root = ActorRoot(); obs = observer('actor', root=root)
        with obs:
            invoke_actor(root, obs, micro(), packed=False); obs.check_errors()
        receipt = obs.finalize(outcomes_for_actor(obs)); self.assertTrue(receipt['observer_evidence_complete'])
        self.assertEqual(obs.points['rank00-first']['index'], 2)
        self.assertEqual(obs.points['rank00-first']['context_token_ids'], [2, 3])

    def test_actual_position_change_refused_only_after_native(self):
        root = ActorRoot(); obs = observer('actor', root=root)
        with obs:
            invoke_actor(root, obs, micro(), change_positions=True)
            self.assertEqual(root.calls, 1)
            with self.assertRaisesRegex(ValueError, 'position_ids'):
                obs.check_errors()
        self.assertFalse(obs.finalize(outcomes_for_actor(obs))['observer_evidence_complete'])
        self.assertTrue(all(not m._forward_hooks for m in obs.modules.values()))

    def test_wrong_boundary_shape_deferred_and_cleanup(self):
        root = ActorRoot(); root.model.norm.bad_shape = True
        obs = observer('actor', root=root)
        with obs:
            result = invoke_actor(root, obs, micro())
            self.assertEqual(result.shape[-1], 1)
            with self.assertRaisesRegex(ValueError, 'bound hidden'):
                obs.check_errors()
        self.assertFalse(obs.finalize(outcomes_for_actor(obs))['observer_evidence_complete'])
        self.assertFalse(root._forward_pre_hooks)

    def test_inplace_after_hook_does_not_change_snapshot(self):
        model = Core('vllm'); obs = observer('vllm', root=model)
        step, ids, outcome = actual_vllm_mapping()
        with obs:
            obs.observe_vllm_step(step); model(ids)
            snapshot = obs.vectors['rank00-first']['embedding'].clone()
            model.embed_tokens.last.add_(100)
            self.assertTrue(torch.equal(obs.vectors['rank00-first']['embedding'], snapshot))
            obs.check_errors()
        self.assertTrue(obs.finalize([outcome])['observer_evidence_complete'])

    def test_repeat_anchor_is_refused_without_replacing_second_output(self):
        model = Core('vllm'); obs = observer('vllm', root=model)
        step, ids, outcome = actual_vllm_mapping()
        with obs:
            obs.observe_vllm_step(step); model(ids)
            obs.observe_vllm_step(step); returned = model(ids)
            self.assertIs(returned, model.embed_tokens.last)
            self.assertEqual(model.norm.calls, 2)
            with self.assertRaisesRegex(ValueError, 'Repeated'):
                obs.check_errors()
        self.assertFalse(obs.finalize([outcome])['observer_evidence_complete'])

    def test_final_context_position_identity_refusal(self):
        root = ActorRoot(); obs = observer('actor', root=root)
        with obs:
            invoke_actor(root, obs, micro()); obs.check_errors()
        outcomes = outcomes_for_actor(obs); outcomes[0]['context_positions'] = [1, 2]
        self.assertFalse(obs.finalize(outcomes)['observer_evidence_complete'])

    def test_real_native_early_eos_only_allowed_missing(self):
        root = ActorRoot(); obs = observer('actor', rank=4, root=root)
        with obs:
            invoke_actor(root, obs, micro()); obs.check_errors()
        receipt = obs.finalize(outcomes_for_actor(obs, absent_second=True))
        self.assertTrue(receipt['observer_evidence_complete'])
        self.assertEqual(receipt['captured_vector_count'], 74)
        self.assertEqual(receipt['maximum_vectors_per_rank'], 148)
        self.assertEqual(receipt['records']['rank04-row160-offset17']['status'], 'absent_native_early_finish')

    def test_unexplained_missing_does_not_become_early_eos(self):
        root = ActorRoot(); obs = observer('actor', rank=4, root=root)
        with obs:
            invoke_actor(root, obs, micro()); obs.check_errors()
        self.assertFalse(obs.finalize(outcomes_for_actor(obs))['observer_evidence_complete'])
        with self.assertRaises(ValueError):
            obs.tensor_evidence()

    def test_nonfinite_is_post_native_error(self):
        root = ActorRoot(); root.model.norm.nonfinite = True
        obs = observer('actor', root=root)
        with obs:
            result = invoke_actor(root, obs, micro())
            self.assertTrue(torch.isnan(result).any()); self.assertEqual(root.calls, 1)
            with self.assertRaisesRegex(ValueError, 'nonfinite'):
                obs.check_errors()

    def test_native_exception_propagates_and_hooks_restored(self):
        root = ActorRoot(); root.model.layers[2].input_layernorm.raise_native = True
        obs = observer('actor', root=root)
        with self.assertRaisesRegex(RuntimeError, 'original native fixture exception'):
            with obs:
                invoke_actor(root, obs, micro())
        self.assertFalse(root._forward_pre_hooks)
        self.assertTrue(all(not m._forward_hooks for m in obs.modules.values()))
        self.assertFalse(obs.finalize(outcomes_for_actor(obs))['observer_evidence_complete'])

    def test_rank4_second_call_original_offset_and_budget(self):
        root = ActorRoot(); obs = observer('actor', rank=4, root=root)
        with obs:
            invoke_actor(root, obs, micro()); obs.check_errors(); obs.clear_step()
            invoke_actor(root, obs, micro(offset=17), start=160, call=40)
            obs.check_errors(); obs.clear_step()
        receipt = obs.finalize(outcomes_for_actor(obs))
        self.assertTrue(receipt['observer_evidence_complete']); self.assertEqual(receipt['captured_vector_count'], 148)
        self.assertEqual(len(obs.points['rank04-row160-offset17']['context_positions']), 19)
        self.assertEqual(receipt['maximum_vectors_global_per_backend'], 666)

    def test_legal_numeric_difference_is_observation_not_acceptance(self):
        native = Core('vllm'); nv = observer('vllm', root=native)
        step, ids, outcome = actual_vllm_mapping()
        with nv:
            nv.observe_vllm_step(step); native(ids); nv.check_errors()
        self.assertTrue(nv.finalize([outcome])['observer_evidence_complete'])
        root = ActorRoot(); root.model.layers[7].input_layernorm.shift = 1.
        av = observer('actor', root=root)
        with av:
            invoke_actor(root, av, micro()); av.check_errors()
        self.assertTrue(av.finalize([outcome])['observer_evidence_complete'])
        comparison = compare_boundary_observers(av, nv)
        self.assertFalse(comparison['probability_accepted']); self.assertFalse(comparison['training_accepted'])
        self.assertEqual(comparison['anchors'][0]['first_observed_different_boundary'], 'layers.07.input_layernorm')
        self.assertFalse(comparison['anchors'][0]['unique_kernel_cause_proven'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
