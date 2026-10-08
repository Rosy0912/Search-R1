"""Tiny synthetic CPU runtime contracts; no checkpoints, vLLM, GPU or SSH."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import window_runtime as R

CORE_FILE = HERE.parent / 'tree-vllm-window-core/vllm_window_weights.py'
spec = importlib.util.spec_from_file_location('runtime_test_bound_weight_core', CORE_FILE)
C = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = C
spec.loader.exec_module(C)


def entry(path):
    return dict(path=str(path), sha256=R.file_sha(path))


def temporary_source(directory, name, text):
    path = Path(directory) / name
    path.write_text(text)
    namespace = {'__name__': 'synthetic_' + name.replace('.', '_')}
    exec(compile(text, str(path), 'exec'), namespace)
    return path, namespace


class FakeObserver:
    def __init__(self, failure=None):
        self.records = []; self.failure = failure

    def observe(self, label, value):
        self.records.append((label, value))
        if self.failure == label:
            raise ValueError('synthetic observer failure ' + label)

    def register_request(self, request_id, tokens, row):
        self.observe('register', (request_id, tokens, row))

    def observe_model_input(self, model_input):
        self.observe('execute', model_input)

    def observe_generated(self, request_outputs):
        self.observe('output', request_outputs)


class FakeSyncWeights:
    def __init__(self):
        self.records = []

    def after_sync(self, actor_weights):
        self.records.append(actor_weights)


class FakeRunner:
    def __init__(self):
        self.calls = []; self.result = object(); self.fail = False

    def execute_model(self, model_input, kv_caches=None, intermediate_tensors=None):
        self.calls.append((model_input, kv_caches, intermediate_tensors))
        if self.fail:
            raise RuntimeError('synthetic native execute failure')
        return self.result


class FakeEngine:
    def __init__(self, runner):
        self.model_executor = NS(worker=NS(model_runner=runner))
        self.calls = []; self.result = object()

    def add_request(self, request_id, prompt, params=None, arrival_time=None, *, inputs=None):
        self.calls.append((request_id, prompt, params, arrival_time, inputs))
        return self.result


class FakeFacade:
    def __init__(self):
        self.runner = FakeRunner(); self.llm_engine = FakeEngine(self.runner)
        self.sync_calls = []; self.output_calls = []; self.sync_result = object(); self.output_result = object()
        self.fail_output = False

    def sync_model_weights(self, actor_weights, load_format='dtensor'):
        self.sync_calls.append((actor_weights, load_format))
        return self.sync_result

    def _post_process_outputs(self, request_outputs, use_tqdm=False):
        self.output_calls.append((request_outputs, use_tqdm))
        if self.fail_output:
            raise RuntimeError('synthetic native output failure after observer')
        return self.output_result


def wrappers(observer=None):
    llm = FakeFacade(); observer = observer or FakeObserver(); weights = FakeSyncWeights()
    source = entry(__file__)
    expected = dict(engine_add_request=source, model_runner=source, facade=source, decorators=[])
    return llm, observer, weights, R.NativeWindowWrappers(llm, observer, weights, expected)


def native_window(llm, tokens=(1, 2, 3), actor_weights=None):
    actor_weights = {} if actor_weights is None else actor_weights
    results = [llm.sync_model_weights(actor_weights=actor_weights, load_format='dtensor')]
    results.append(llm.llm_engine.add_request(prompt={'prompt_token_ids': tokens}, request_id='request-0', params='params'))
    results.append(llm.runner.execute_model(model_input='actual input', kv_caches='actual cache'))
    results.append(llm._post_process_outputs(request_outputs='original raw RequestOutput', use_tqdm=False))
    return results


class GlobalRefusal(ValueError):
    pass


class FakeCoordinator:
    """Records CPU ordering; the injected peer refusal is NOT a real collective."""
    def __init__(self, peer_preflight_failure=False):
        self.stages = []; self.signatures = []; self.peer_preflight_failure = peer_preflight_failure

    def local_stage(self, label, operation):
        self.stages.append(label)
        try:
            result = operation()
        except Exception as error:
            raise GlobalRefusal('synthetic all-rank local-stage refusal: ' + label + ': ' + str(error)) from error
        if self.peer_preflight_failure and label == 'same-window all-rank pre-gather source contract':
            raise GlobalRefusal('synthetic peer preflight refusal before all gathers')
        return result

    def signature(self, values):
        self.signatures.append(list(values))


class FakeLogicalDTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, tensor):
        return torch.Tensor._make_subclass(cls, tensor, False)

    def __init__(self, tensor):
        self.full_tensor_calls = 0

    def full_tensor(self):
        self.full_tensor_calls += 1
        return self.as_subclass(torch.Tensor)


class TinyLiveModel(torch.nn.Module):
    def __init__(self, values):
        super().__init__()
        self.actual = {}
        for number, (name, value) in enumerate(values.items()):
            if name == 'lm_head.weight':
                parameter = self.actual['model.embed_tokens.weight']
            else:
                parameter = torch.nn.Parameter(value, requires_grad=False)
            self.register_parameter('p%03d' % number, parameter)
            self.actual[name] = parameter
        self.register_buffer('synthetic_static_buffer', torch.arange(3))
        self.eval()

    def named_parameters(self, prefix='', recurse=True, remove_duplicate=True):
        seen = set()
        for name, value in self.actual.items():
            if remove_duplicate and id(value) in seen:
                continue
            seen.add(id(value))
            yield name, value


def live_weights(mismatch=False, coordinator=None):
    cfg = NS(num_hidden_layers=1, vocab_size=8, hidden_size=4, intermediate_size=6,
             num_attention_heads=2, num_key_value_heads=1, tie_word_embeddings=True)
    source = {}
    for item in C.qwen2_tp1_mapping(1):
        for name in item.sources:
            if name in source:
                continue
            if name in ('model.embed_tokens.weight', 'lm_head.weight'):
                shape = (8, 4)
            elif name == 'model.norm.weight' or name.endswith('layernorm.weight'):
                shape = (4,)
            elif '.self_attn.' in name:
                rows = 4 if '.q_proj.' in name or '.o_proj.' in name else 2
                shape = (rows,) if name.endswith('.bias') else (rows, 4)
            elif name.endswith('mlp.down_proj.weight'):
                shape = (4, 6)
            else:
                shape = (6, 4)
            count = 1
            for dim in shape:
                count *= dim
            source[name] = (torch.arange(count).reshape(shape).float() + len(source) * 10) / 7
    source['lm_head.weight'] = source['model.embed_tokens.weight'].clone()
    actual = {item.target: (torch.cat([source[name] for name in item.sources], 0)
                           if len(item.sources) > 1 else source[item.sources[0]]).bfloat16().clone()
              for item in C.qwen2_tp1_mapping(1)}
    if mismatch:
        actual['model.norm.weight'][0] += 1
    model = TinyLiveModel(actual)
    logical = {name: FakeLogicalDTensor(value) for name, value in source.items()}
    coordinator = coordinator or FakeCoordinator()
    return R.LiveWindowWeights(C, model, cfg, 'cpu', coordinator), logical, model, coordinator


class SourceBindingTests(unittest.TestCase):
    def test_cross_source_wrapped_method_chain_is_fully_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            native_file, native = temporary_source(directory, 'native.py',
                'class Native:\n    def call(self, value):\n        return value\n')
            wrapper_file, wrapper = temporary_source(directory, 'wrapper.py',
                'import functools\ndef decorate(original):\n    @functools.wraps(original)\n    def outer(*args, **kwargs):\n        return original(*args, **kwargs)\n    return outer\n')
            native['Native'].call = wrapper['decorate'](native['Native'].call)
            obj = native['Native']()
            receipt = R.bind_method(obj.call, entry(native_file), [entry(wrapper_file)])
            self.assertEqual([row['path'] for row in receipt], [str(wrapper_file), str(native_file)])
            token = object(); self.assertIs(obj.call(token), token)

    def test_unbound_wrapper_changed_bytes_and_wrong_final_source_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            native_file, native = temporary_source(directory, 'native.py',
                'class Native:\n    def call(self, value):\n        return value\n')
            wrapper_file, wrapper = temporary_source(directory, 'wrapper.py',
                'import functools\ndef decorate(original):\n    @functools.wraps(original)\n    def outer(*args, **kwargs):\n        return original(*args, **kwargs)\n    return outer\n')
            native['Native'].call = wrapper['decorate'](native['Native'].call)
            method = native['Native']().call
            with self.assertRaisesRegex(ValueError, 'source path differs'):
                R.bind_method(method, entry(native_file))
            with self.assertRaisesRegex(ValueError, 'Final unwrapped'):
                R.bind_method(method, entry(wrapper_file), [entry(native_file)])
            stale = entry(wrapper_file); wrapper_file.write_text(wrapper_file.read_text() + '\n# changed\n')
            with self.assertRaisesRegex(ValueError, 'source SHA changed'):
                R.bind_method(method, entry(native_file), [stale])


class PassiveWrapperTests(unittest.TestCase):
    def test_actual_kwargs_bound_native_called_once_and_return_identity_preserved(self):
        llm, observer, weights, wrapper = wrappers(); actual_weights = {'actual': object()}
        with wrapper:
            values = native_window(llm, actor_weights=actual_weights)
        self.assertIs(values[0], llm.sync_result); self.assertIs(values[1], llm.llm_engine.result)
        self.assertIs(values[2], llm.runner.result); self.assertIs(values[3], llm.output_result)
        self.assertEqual((len(llm.sync_calls), len(llm.llm_engine.calls), len(llm.runner.calls), len(llm.output_calls)), (1, 1, 1, 1))
        self.assertIs(weights.records[0], actual_weights)
        self.assertEqual(observer.records, [('register', ('request-0', (1, 2, 3), 0)),
            ('execute', 'actual input'), ('output', 'original raw RequestOutput')])
        receipt = wrapper.receipt()
        self.assertEqual(receipt['counts'], dict(sync=1, execute=1, output=1))
        self.assertFalse(receipt['distribution_changed'])
        for obj, name in ((llm.llm_engine, 'add_request'), (llm.runner, 'execute_model'),
                          (llm, 'sync_model_weights'), (llm, '_post_process_outputs')):
            self.assertNotIn(name, vars(obj))

    def test_sync_default_load_format_is_bound_as_native_default(self):
        llm, observer, weights, wrapper = wrappers(); actual = {'state': object()}
        with wrapper:
            llm.sync_model_weights(actor_weights=actual)
            llm.runner.execute_model(model_input='input')
            llm._post_process_outputs(request_outputs='outputs')
        self.assertEqual(wrapper.errors, [])
        self.assertIs(weights.records[0], actual)
        wrapper.receipt()

    def test_observer_failure_deferred_and_native_result_preserved_but_receipt_refused(self):
        llm, observer, weights, wrapper = wrappers(FakeObserver('execute'))
        with wrapper:
            results = native_window(llm)
        self.assertIs(results[2], llm.runner.result)
        self.assertEqual(len(llm.runner.calls), 1)
        self.assertEqual(len(llm.output_calls), 1)
        with self.assertRaisesRegex(ValueError, 'passive-wrapper evidence failed'):
            wrapper.receipt()

    def test_changed_source_refuses_before_any_instance_patch_or_native_call(self):
        llm, observer, weights, wrapper = wrappers()
        wrapper.sources['engine_add_request'] = dict(entry(__file__), sha256='0' * 64)
        with self.assertRaisesRegex(ValueError, 'source SHA changed'):
            wrapper.__enter__()
        self.assertEqual(wrapper.saved, []); self.assertEqual(llm.llm_engine.calls, [])
        self.assertNotIn('add_request', vars(llm.llm_engine))

    def test_second_window_sync_or_output_count_refuses_normal_receipt(self):
        for duplicate in ('sync', 'output'):
            llm, observer, weights, wrapper = wrappers()
            with wrapper:
                native_window(llm)
                if duplicate == 'sync':
                    llm.sync_model_weights(actor_weights={}, load_format='dtensor')
                else:
                    llm._post_process_outputs(request_outputs='duplicate')
            with self.assertRaisesRegex(ValueError, 'exactly one native generation window'):
                wrapper.receipt()

    def test_bad_prompt_or_sync_format_defers_refusal_without_replacing_native(self):
        for mode in ('prompt', 'format'):
            llm, observer, weights, wrapper = wrappers()
            with wrapper:
                llm.sync_model_weights(actor_weights={}, load_format='hf' if mode == 'format' else 'dtensor')
                result = llm.llm_engine.add_request(request_id='request', prompt={'text': 'bad'}
                    if mode == 'prompt' else {'prompt_token_ids': [1]})
                llm.runner.execute_model(model_input='input')
                llm._post_process_outputs(request_outputs='outputs')
            self.assertIs(result, llm.llm_engine.result)
            with self.assertRaisesRegex(ValueError, 'passive-wrapper evidence failed'):
                wrapper.receipt()

    def test_native_exception_propagates_and_hooks_restore(self):
        llm, observer, weights, wrapper = wrappers(); llm.runner.fail = True
        with self.assertRaisesRegex(RuntimeError, 'native execute failure'):
            with wrapper:
                llm.sync_model_weights(actor_weights={}, load_format='dtensor')
                llm.runner.execute_model(model_input='input')
        self.assertNotIn('execute_model', vars(llm.runner))
        with self.assertRaises(ValueError):
            wrapper.receipt()

    def test_native_output_exception_after_observation_cannot_publish_success(self):
        llm, observer, weights, wrapper = wrappers(); llm.fail_output = True
        with self.assertRaisesRegex(RuntimeError, 'native output failure after observer'):
            with wrapper:
                native_window(llm)
        self.assertEqual(wrapper.counts, dict(sync=1, execute=1, output=1))
        self.assertNotIn('_post_process_outputs', vars(llm))
        with self.assertRaises(ValueError):
            wrapper.receipt()


class LiveWeightTests(unittest.TestCase):
    def test_missing_extra_non_tensor_plain_tensor_or_non_fp32_refused_before_gather(self):
        for kind in ('missing', 'extra', 'non_tensor', 'plain_tensor', 'half', 'noncallable_gather'):
            live, logical, model, coordinator = live_weights()
            key = 'model.norm.weight'
            if kind == 'missing':
                logical.pop(key)
            elif kind == 'extra':
                logical['unbound.weight'] = FakeLogicalDTensor(torch.ones(1))
            elif kind == 'non_tensor':
                logical[key] = object()
            elif kind == 'plain_tensor':
                logical[key] = torch.ones(4)
            elif kind == 'noncallable_gather':
                logical[key].full_tensor = None
            else:
                logical[key] = FakeLogicalDTensor(torch.ones(4).half())
            with self.assertRaisesRegex(GlobalRefusal, 'pre-gather source contract'):
                live.after_sync(logical)
            self.assertEqual(coordinator.stages, ['same-window all-rank pre-gather source contract'])
            self.assertEqual(coordinator.signatures, [])
            self.assertEqual(sum(value.full_tensor_calls for value in logical.values()
                                 if isinstance(value, FakeLogicalDTensor)), 0)
            self.assertIsNone(live.pre)

    def test_peer_preflight_failure_stops_valid_local_rank_before_all_gathers(self):
        coordinator = FakeCoordinator(peer_preflight_failure=True)
        live, logical, model, _ = live_weights(coordinator=coordinator)
        with self.assertRaisesRegex(GlobalRefusal, 'peer preflight'):
            live.after_sync(logical)
        self.assertEqual(coordinator.signatures, [])
        self.assertTrue(all(value.full_tensor_calls == 0 for value in logical.values()))

    def test_live_finite_value_mismatch_remains_measurement_and_target_snapshot_stability_is_separate(self):
        live, logical, model, coordinator = live_weights(mismatch=True)
        before = {name: value.detach().clone() for name, value in model.named_parameters(remove_duplicate=False)}
        live.after_sync(logical)
        self.assertFalse(live.pre['all_parameter_values_match'])
        self.assertGreater(live.pre['different_elements'], 0)
        self.assertEqual([stage.removeprefix('original DTensor gather ') for stage in coordinator.stages
                          if stage.startswith('original DTensor gather ')], sorted(logical))
        self.assertTrue(all(value.full_tensor_calls == 1 for value in logical.values()))
        live.before_offload(); receipt = live.receipt()
        self.assertTrue(receipt['post']['target_bytes_unchanged_since_live_postsync_snapshot'])
        self.assertFalse(receipt['post']['source_cast_all_values_match'])
        self.assertTrue(receipt['post']['same_window_tie'])
        self.assertFalse(receipt['pre']['actual_device_is_cuda'])
        self.assertFalse(receipt['probability_accepted']); self.assertFalse(receipt['training_accepted'])
        for name, expected in before.items():
            self.assertTrue(torch.equal(expected, model.actual[name]))

    def test_exact_values_can_match_without_claiming_probability_or_historical_identity(self):
        live, logical, model, coordinator = live_weights()
        live.after_sync(logical); live.before_offload(); receipt = live.receipt()
        self.assertTrue(receipt['post']['source_cast_all_values_match'])
        self.assertFalse(receipt['post']['historical_identity_proven'])
        self.assertFalse(receipt['historical_3268_identity_proven'])
        self.assertFalse(receipt['probability_accepted'])

    def test_second_sync_refuses_before_another_gather(self):
        live, logical, model, coordinator = live_weights()
        live.after_sync(logical)
        with self.assertRaisesRegex(GlobalRefusal, 'second synchronization'):
            live.after_sync(logical)
        self.assertTrue(all(value.full_tensor_calls == 1 for value in logical.values()))

    def test_target_or_buffer_change_after_sync_refuses_stable_target_receipt(self):
        for kind in ('parameter', 'buffer', 'mode'):
            live, logical, model, coordinator = live_weights()
            live.after_sync(logical)
            with torch.no_grad():
                if kind == 'parameter':
                    model.actual['model.norm.weight'][0] += 1
                elif kind == 'buffer':
                    model.synthetic_static_buffer[0] += 1
                else:
                    model.train()
            with self.assertRaisesRegex(ValueError, 'weights/buffers/modes changed'):
                live.before_offload()
            with self.assertRaisesRegex(ValueError, 'Both live-window weight boundaries'):
                live.receipt()

    def test_equal_value_embedding_copy_is_not_same_window_tied_alias(self):
        live, logical, model, coordinator = live_weights()
        live.after_sync(logical)
        model.actual['lm_head.weight'] = torch.nn.Parameter(model.actual['model.embed_tokens.weight'].clone(),
                                                          requires_grad=False)
        with self.assertRaisesRegex(ValueError, 'tied embedding/head alias changed'):
            live.before_offload()
        with self.assertRaises(ValueError):
            live.receipt()

    def test_post_boundary_failure_cannot_publish_a_previous_success_receipt(self):
        live, logical, model, coordinator = live_weights()
        live.after_sync(logical); live.before_offload()
        with torch.no_grad():
            model.actual['model.norm.weight'][0] += 1
        with self.assertRaisesRegex(ValueError, 'weights/buffers/modes changed'):
            live.before_offload()
        with self.assertRaises(ValueError):
            live.receipt()


if __name__ == '__main__':
    torch.set_num_threads(2)
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    no_cuda = not torch.cuda.is_initialized()
    vllm_loaded = any(name == 'vllm' or name.startswith('vllm.') for name in sys.modules)
    distributed_initialized = torch.distributed.is_initialized()
    safe_cpu_only = no_cuda and not vllm_loaded and not distributed_initialized
    print(json.dumps(dict(status='WINDOW_RUNTIME_CPU_FIXTURES_PASS' if result.wasSuccessful() and safe_cpu_only else 'FAILED',
        tests_run=result.testsRun, runtime_sha256=R.file_sha(HERE / 'window_runtime.py'),
        test_sha256=R.file_sha(__file__), core_sha256=R.file_sha(CORE_FILE), cuda_initialized=not no_cuda,
        real_assets_read=False, vllm_loaded=vllm_loaded, real_distributed_collectives=distributed_initialized,
        core_existing_tests_rerun=False, acceptance=False), sort_keys=True))
    raise SystemExit(0 if result.wasSuccessful() and safe_cpu_only else 1)
