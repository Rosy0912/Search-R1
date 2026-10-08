"""Actual diagnostic/compute AST, real CPU tensors, mock actor; no HF/GPU assets."""
import ast
from contextlib import nullcontext
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

ROOT = Path(__file__).resolve().parent


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fixture = load_file('adapter_cpu_capture_fixture', ROOT/'tests/baselines/test_probability_probe_capture.py')
P = fixture.P
scope = dict(torch=torch, SimpleNamespace=SimpleNamespace, nullcontext=nullcontext,
             require=P.require, tensor_digest=P.tensor_digest,
             response_prediction_map=P.response_prediction_map, capture_original_call=P.capture_original_call)
runner_path = ROOT/'search_r1/baselines/probability_probe_runner.py'
nodes = [n for n in ast.parse(runner_path.read_text()).body if isinstance(n, ast.FunctionDef)
         and n.name in ('rng_state', 'input_identity', 'immutable_native_call')]
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(runner_path), 'exec'), scope)
window_path = ROOT/'single_window_runner.py'
nodes = [n for n in ast.parse(window_path.read_text()).body
         if isinstance(n, (ast.FunctionDef, ast.ClassDef))
         and n.name in ('native_four_key_diagnostic_view', 'ActorWindowRecorder')]
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(window_path), 'exec'), scope)
view = scope['native_four_key_diagnostic_view']
Recorder = scope['ActorWindowRecorder']


class Batch(dict):
    def split(self, rows):
        return [Batch({key: value[start:start+rows] for key, value in self.items()})
                for start in range(0, len(self['input_ids']), rows)]


class Proto:
    def __init__(self, batch):
        self.batch = Batch(batch)
        self.meta_info = dict(micro_batch_size=4, temperature=1., use_dynamic_bsz=False)

    def select(self, batch_keys):
        return Proto({key: self.batch[key] for key in batch_keys})


original_path = ROOT/'verl/workers/actor/dp_actor.py'
original_class = next(n for n in ast.parse(original_path.read_text()).body
                      if isinstance(n, ast.ClassDef) and n.name == 'DataParallelPPOActor')
compute = next(n for n in original_class.body if isinstance(n, ast.FunctionDef) and n.name == 'compute_log_prob')
original_scope = dict(torch=torch, DataProto=Proto)
exec(compile(ast.Module(body=[compute], type_ignores=[]), str(original_path), 'exec'), original_scope)


class Actor(fixture.Actor):
    compute_log_prob = original_scope['compute_log_prob']

    def __init__(self, remove_padding=True):
        super().__init__(remove_padding=remove_padding)
        self.native_inputs = []

    def _forward_micro_batch(self, data, temperature):
        self.native_inputs.append((data, tuple(data), dict(data), temperature))
        if set(data) != {'responses','input_ids','attention_mask','position_ids'}:
            raise AssertionError('Diagnostic mask reached native input')
        return super()._forward_micro_batch(data, temperature)


def raw(rows=2):
    value = fixture.batch()
    return Batch({key: value[key].repeat((rows//2, 1))
                  for key in ('responses','input_ids','attention_mask','position_ids')})


class Coordinator:
    def __init__(self): self.labels = []
    def local_stage(self, label, operation):
        self.labels.append(label)
        return operation()


class Tests(unittest.TestCase):
    def test_real_capture_mapping_and_exact_native_object_for_packed_and_dense(self):
        for packed in (True, False):
            with self.subTest(packed=packed):
                actor = Actor(packed); data = raw(); original = dict(data)
                diagnostic, proxy = view(actor, actor._forward_micro_batch, data, 2.)
                captured, receipt = scope['immutable_native_call'](proxy, diagnostic, 2., None)
                self.assertEqual(actor.entries, 1); self.assertEqual(actor.actor_module.calls, 1)
                self.assertIs(actor.native_inputs[0][0], data)
                self.assertEqual(actor.native_inputs[0][1], tuple(original))
                self.assertTrue(all(actor.native_inputs[0][2][key] is original[key] for key in original))
                self.assertEqual(actor.native_inputs[0][3], 2.)
                self.assertTrue(torch.equal(captured['mapping']['effective_mask'], data['attention_mask'][:,-3:].bool()))
                self.assertTrue(receipt['rng_unchanged']); self.assertEqual(receipt['original_actor_calls'], 1)
                self.assertFalse(receipt['backward_called']); self.assertFalse(receipt['optimizer_instantiated'])
                self.assertEqual(len(actor.actor_module._forward_hooks), 0)

    def test_original_compute_ast_keeps_two_micro_calls_order_and_values(self):
        actor = Actor(); data = raw(6); coordinator = Coordinator()
        checks = []; adapter = SimpleNamespace(check=checks.append)
        original = actor._forward_micro_batch; weight = actor.actor_module.weight.detach().clone()
        rng = torch.get_rng_state().clone()
        with Recorder(actor, data, coordinator, None, [], None, None, adapter) as recorder:
            result = actor.compute_log_prob(Proto(data))
        self.assertEqual(recorder.calls, 2); self.assertEqual(actor.entries, 2)
        self.assertEqual(actor.actor_module.calls, 2)
        self.assertEqual([len(item[0]['input_ids']) for item in actor.native_inputs], [4, 2])
        for start, item in zip((0, 4), actor.native_inputs):
            self.assertEqual(item[1], ('responses','input_ids','attention_mask','position_ids'))
            for key in item[0]: self.assertTrue(torch.equal(item[0][key], data[key][start:start+4]))
        self.assertEqual(tuple(result.shape), (6, 3))
        self.assertTrue(torch.equal(actor.actor_module.weight, weight))
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertIsNone(actor.actor_module.weight.grad)
        self.assertFalse(actor.actor_module.training)
        self.assertEqual(actor._forward_micro_batch, original)
        self.assertEqual(checks, ['actor-micro4-000', 'actor-micro4-001'])
        self.assertEqual(len(recorder.records), 2)
        self.assertTrue(all(r['native_forward_receipt']['original_native_input_object_preserved'] for r in recorder.records))

    def test_missing_or_extra_native_keys_refuse_before_any_model_call(self):
        for change in ('missing', 'extra'):
            actor = Actor(); data = raw()
            if change == 'missing': del data['position_ids']
            else: data['loss_mask'] = torch.ones_like(data['responses'])
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'exactly four keys'):
                view(actor, actor._forward_micro_batch, data, 1.)
            self.assertEqual(actor.entries, 0)

    def test_replaced_view_reference_or_temperature_cannot_reach_native(self):
        for change in ('view', 'reference', 'temperature'):
            actor = Actor(); data = raw(); diagnostic, proxy = view(actor, actor._forward_micro_batch, data, 1.)
            observed = diagnostic; temperature = 1.
            if change == 'view': observed = dict(diagnostic)
            if change == 'reference': diagnostic['input_ids'] = diagnostic['input_ids'].clone()
            if change == 'temperature': temperature = .5
            with self.subTest(change=change), self.assertRaises(ValueError):
                proxy._forward_micro_batch(observed, temperature)
            self.assertEqual(actor.entries, 0)

    def test_invalid_predictor_mapping_refuses_in_prepare_before_native_stage(self):
        actor = Actor(); data = raw(); data['attention_mask'][0,3] = 0
        coordinator = Coordinator(); adapter = SimpleNamespace(check=lambda label: None)
        with self.assertRaisesRegex(ValueError, 'attention-masked native prediction'):
            with Recorder(actor, data, coordinator, None, [], None, None, adapter):
                actor.compute_log_prob(Proto(data))
        self.assertEqual(coordinator.labels, ['actor micro4 prepare 000'])
        self.assertEqual(actor.entries, 0)

    def test_original_native_failure_removes_hook_and_restores_actor(self):
        actor = Actor(); data = raw()
        def failure(value, temperature): raise RuntimeError('native fixture failure')
        actor._forward_micro_batch = failure
        coordinator = Coordinator(); adapter = SimpleNamespace(check=lambda label: None)
        with self.assertRaisesRegex(RuntimeError, 'native fixture failure'):
            with Recorder(actor, data, coordinator, None, [], None, None, adapter):
                actor.compute_log_prob(Proto(data))
        self.assertIs(actor._forward_micro_batch, failure)
        self.assertEqual(len(actor.actor_module._forward_hooks), 0)


if __name__ == '__main__': unittest.main(verbosity=2)
