"""Exact captured wrapper source + synthetic native body, CPU only."""
import ast
from functools import wraps
import hashlib
import inspect
from pathlib import Path
import tempfile
from typing import List, Optional
import unittest

from window_runtime import bind_method

BASE_SHA = '49fe0eedd1667889daa63d5aa572d00f5eefbb794ea4368497f5243b02a54173'


class ModelRunnerBaseBindingTests(unittest.TestCase):
    def fixture(self, directory):
        base = Path(__file__).parent/'fixtures/vllm-model-runner-base-source.txt'
        source = base.read_bytes()
        self.assertEqual(hashlib.sha256(source).hexdigest(), BASE_SHA)
        decorator = next(n for n in ast.parse(source.decode()).body
                         if isinstance(n, ast.FunctionDef) and n.name == 'dump_input_when_exception')
        namespace = dict(Optional=Optional, List=List, wraps=wraps)
        exec(compile(ast.Module(body=[decorator], type_ignores=[]), str(base), 'exec'), namespace)
        # Native body is deliberately a sentinel, never a real model invocation.
        native = Path(directory)/'synthetic_native.py'
        native.write_text('def execute_model(model_input, kv_caches, intermediate_tensors=None, num_steps=1):\n    return model_input\n')
        env = {}
        exec(compile(native.read_bytes(), str(native), 'exec'), env)
        original = env['execute_model']
        wrapped = namespace['dump_input_when_exception'](exclude_args=[0], exclude_kwargs=['self'])(original)
        return wrapped, original, dict(path=str(native), sha256=hashlib.sha256(native.read_bytes()).hexdigest()), dict(path=str(base), sha256=BASE_SHA)

    def test_captured_wrapper_requires_precise_extra_source_and_preserves_signature(self):
        with tempfile.TemporaryDirectory() as directory:
            wrapped, original, native, base = self.fixture(directory)
            receipt = bind_method(wrapped, native, [base])
            self.assertEqual([row['path'] for row in receipt], [base['path'], native['path']])
            self.assertEqual(inspect.signature(wrapped), inspect.signature(original))
            sentinel = object()
            self.assertIs(wrapped(model_input=sentinel, kv_caches=None), sentinel)

    def test_old_missing_wrapper_allowlist_refuses(self):
        with tempfile.TemporaryDirectory() as directory:
            wrapped, _, native, _ = self.fixture(directory)
            with self.assertRaisesRegex(ValueError, 'source path differs'):
                bind_method(wrapped, native)

    def test_wrong_wrapper_or_final_native_hash_refuses(self):
        with tempfile.TemporaryDirectory() as directory:
            wrapped, _, native, base = self.fixture(directory)
            for wrong in ('wrapper', 'native'):
                expected, wrappers = dict(native), [dict(base)]
                if wrong == 'wrapper':
                    wrappers[0]['sha256'] = '0'*64
                else:
                    expected['sha256'] = '0'*64
                with self.subTest(wrong=wrong), self.assertRaisesRegex(ValueError, 'source SHA changed'):
                    bind_method(wrapped, expected, wrappers)


if __name__ == '__main__':
    unittest.main()
