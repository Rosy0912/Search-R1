"""Captured-source CPU compatibility only; no vLLM import/model/GPU/assets."""
import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
import unittest


ROOT = Path(__file__).parent
CONFIG_SHA = '3a9d50709aa2f7ca3b14741b507232e2cd954936780d4a0365cabea12e6cc8b0'


def source_nodes():
    source = (ROOT/'fixtures/vllm-0.6.3-config-source.txt').read_bytes()
    if hashlib.sha256(source).hexdigest() != CONFIG_SHA:
        raise ValueError('Captured actual installed config source changed')
    configs = ast.parse(source.decode())
    runner = ast.parse((ROOT/'single_window_runner.py').read_text())
    function = next(n for n in runner.body if isinstance(n, ast.FunctionDef)
                    and n.name == 'actual_engine_metadata')
    properties = ast.literal_eval(next(n.value for n in function.body
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name)
            and t.id == 'properties' for t in n.targets)))
    return configs, function, properties


def actual_scheduler():
    configs, _, _ = source_nodes()
    cls = next(n for n in configs.body if isinstance(n, ast.ClassDef)
               and n.name == 'SchedulerConfig')
    # Execute only this exact captured class. The logger is a passive CPU stub;
    # constructors/verification/property methods and field spelling stay native.
    env = dict(Optional=Optional,
        logger=SimpleNamespace(info=lambda *args: None),
        envs=SimpleNamespace(VLLM_ALLOW_DEPRECATED_BLOCK_MANAGER_V1=False),
        _EMBEDDING_MODEL_MAX_NUM_BATCHED_TOKENS=32768,
        _MULTIMODAL_MODEL_MAX_NUM_BATCHED_TOKENS=5120)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), 'captured-config.py', 'exec'), env)
    return env['SchedulerConfig']


class ActualEngineMetadataTests(unittest.TestCase):
    def test_all_observed_fields_exist_in_captured_actual_classes_and_verl_path(self):
        configs, _, properties = source_nodes()
        classes = {n.name: n for n in configs.body if isinstance(n, ast.ClassDef)}
        count = 0
        for name, fields in properties.items():
            class_name = ''.join(word.title() for word in name.split('_'))
            cls = classes[class_name]
            stored = {n.attr for n in ast.walk(cls) if isinstance(n, ast.Attribute)
                and isinstance(n.ctx, ast.Store) and isinstance(n.value, ast.Name)
                and n.value.id == 'self'}
            self.assertTrue(set(fields).issubset(stored), (name, set(fields)-stored))
            count += len(fields)
        self.assertEqual(count, 18)
        self.assertNotIn('enable_chunked_prefill', properties['scheduler_config'])
        self.assertIn('chunked_prefill_enabled', properties['scheduler_config'])
        # VERL uses installed Cache/Parallel/Scheduler, and a ModelConfig subclass
        # that delegates to the installed constructor. No alternate schema.
        base = ROOT/'verl/third_party/vllm/vllm_v_0_6_3'
        custom = ast.parse((base/'config.py').read_text())
        observed_classes = {''.join(word.title() for word in name.split('_')) for name in properties}
        self.assertEqual([n.name for n in custom.body if isinstance(n, ast.ClassDef)
            and n.name in observed_classes], ['ModelConfig'])
        model = next(n for n in custom.body if isinstance(n, ast.ClassDef)
                     and n.name == 'ModelConfig')
        self.assertEqual([ast.unparse(n) for n in model.bases], ['ModelConfig'])
        self.assertTrue(any(isinstance(n, ast.Call) and ast.unparse(n.func) == 'super().__init__'
                            for n in ast.walk(model)))
        self.assertIn('from vllm.config import ModelConfig', (base/'config.py').read_text())
        self.assertIn('engine_config = super().create_engine_config()', (base/'arg_utils.py').read_text())
        engine_source = ast.parse((base/'llm_engine_sp.py').read_text())
        self.assertTrue(any(isinstance(n, ast.Attribute) and n.attr == 'chunked_prefill_enabled'
            and isinstance(n.value, ast.Name) and n.value.id == 'scheduler_config'
            for n in ast.walk(engine_source)))

    def test_actual_constructor_keyword_maps_to_observed_stored_field(self):
        scheduler = actual_scheduler()
        _, function, properties = source_nodes()
        # Run exact production metadata function without importing runner/vLLM.
        # Class-source/sampling helpers are explicitly out of this field test's scope.
        env = dict(plain_object=lambda x: x, class_source=lambda x: {'scope': 'CPU-field-fixture'},
                   sampling_params_metadata=lambda x: {'scope': 'CPU-field-fixture'})
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'actual-metadata.py', 'exec'), env)
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                engine = SimpleNamespace()
                for name, fields in properties.items():
                    setattr(engine, name, SimpleNamespace(**{f: None for f in fields}))
                engine.scheduler_config = scheduler(max_num_batched_tokens=None, max_num_seqs=256,
                    max_model_len=2325, enable_chunked_prefill=enabled)
                self.assertFalse(hasattr(engine.scheduler_config, 'enable_chunked_prefill'))
                model = SimpleNamespace(logits_processor=object(), sampler=object())
                engine.model_executor = SimpleNamespace(worker=SimpleNamespace(
                    model_runner=SimpleNamespace(model=model)))
                rollout = SimpleNamespace(inference_engine=SimpleNamespace(llm_engine=engine),
                                          sampling_params=object())
                actual = env['actual_engine_metadata'](rollout)['configs']['scheduler_config']
                self.assertEqual(actual['chunked_prefill_enabled'], enabled)
                self.assertEqual(actual['max_num_batched_tokens'], 512 if enabled else 2325)
                self.assertEqual(actual['max_num_seqs'], 256)
                self.assertEqual(actual['num_scheduler_steps'], 1)

    def test_missing_stored_field_refuses_instead_of_silent_none_fallback(self):
        _, function, properties = source_nodes()
        env = dict(plain_object=lambda x: x)
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'actual-metadata.py', 'exec'), env)
        engine = SimpleNamespace()
        for name, fields in properties.items():
            setattr(engine, name, SimpleNamespace(**{f: 1 for f in fields}))
        engine.model_executor = SimpleNamespace(worker=SimpleNamespace(model_runner=object()))
        del engine.scheduler_config.chunked_prefill_enabled
        with self.assertRaises(AttributeError):
            env['actual_engine_metadata'](SimpleNamespace(inference_engine=SimpleNamespace(llm_engine=engine)))


if __name__ == '__main__':
    unittest.main()
