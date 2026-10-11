"""Execute real KV lifecycle branches with a CPU fake engine; no GPU timing claim."""
import ast
import copy
import pathlib
import types
import time
import unittest
from test_policy_weight_cache import Check as WeightCheck

SOURCE = pathlib.Path(__file__).resolve().parents[3] / 'verl/workers/rollout/vllm_rollout/vllm_rollout.py'
CLASS = next(n for n in ast.parse(SOURCE.read_text()).body
             if isinstance(n, ast.ClassDef) and n.name == 'vLLMRollout')
METHOD = next(n for n in CLASS.body if isinstance(n, ast.FunctionDef) and n.name == 'generate_sequences')
# Run the actual init/free statements around the omitted native GPU decode.
BRANCHES = [n for n in METHOD.body if
            (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'retain_kv' for t in n.targets)) or
            (isinstance(n, ast.If) and any(isinstance(x, ast.Call) and isinstance(x.func, ast.Attribute)
             and x.func.attr in ('init_cache_engine', 'free_cache_engine') for x in ast.walk(n)))]


class Check(unittest.TestCase):
    def make(self):
        manager, cuda = WeightCheck().make()
        engine = manager.inference_engine
        engine.kv_inits = engine.kv_frees = 0
        def init(): engine.kv_inits += 1
        def free(): engine.kv_frees += 1
        engine.init_cache_engine, engine.free_cache_engine = init, free
        method = ast.FunctionDef(name='generate', args=ast.arguments(posonlyargs=[], args=[ast.arg(arg='self')],
            kwonlyargs=[], kw_defaults=[], defaults=[]), body=copy.deepcopy(BRANCHES), decorator_list=[])
        namespace = dict(time=time)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(SOURCE), 'exec'), namespace)
        rollout = types.SimpleNamespace(inference_engine=engine, config=types.SimpleNamespace(free_cache_engine=True))
        return manager, rollout, namespace['generate']

    def generate(self, manager, rollout, generate, enabled=True, version=(80, 4)):
        manager.prepare_policy_cache(enabled, version)
        manager.__enter__()
        generate(rollout)
        manager.__exit__(None, None, None)

    def test_one_init_per_phase_release_at_hf(self):
        m, r, g = self.make()
        for _ in range(3): self.generate(m, r, g)
        self.assertEqual((r.inference_engine.kv_inits, r.inference_engine.kv_frees), (1, 0))
        m.release_model_weights()
        self.assertEqual((r.inference_engine.kv_inits, r.inference_engine.kv_frees), (1, 1))
        m.release_model_weights()
        self.assertEqual(r.inference_engine.kv_frees, 1)

    def test_version_change_frees_and_rebuilds(self):
        m, r, g = self.make()
        self.generate(m, r, g)
        self.generate(m, r, g, version=(100, 5))
        self.assertEqual((r.inference_engine.kv_inits, r.inference_engine.kv_frees), (2, 1))

    def test_exception_frees(self):
        m, r, g = self.make()
        m.prepare_policy_cache(True, (80, 4)); m.__enter__(); g(r)
        m.__exit__(RuntimeError, RuntimeError('decode failed'), None)
        self.assertEqual(r.inference_engine.kv_frees, 1)
        self.assertFalse(r.inference_engine._recursive_kv_cache_initialized)

    def test_non_opt_in_unchanged(self):
        m, r, g = self.make()
        for _ in range(3): self.generate(m, r, g, enabled=False)
        self.assertEqual((r.inference_engine.kv_inits, r.inference_engine.kv_frees), (3, 3))


if __name__ == '__main__': unittest.main()
