import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[2]
tree = ast.parse((ROOT / 'ts_probe_dump.py').read_text())
functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'gen_batch']
namespace = {'SamplingParams': lambda **kw: kw}
exec(compile(ast.Module(body=functions, type_ignores=[]), 'ts_probe_dump.py', 'exec'), namespace)


class CalibrationContext(unittest.TestCase):
    def engine(self, capacity=4608):
        engine = SimpleNamespace(
            _g1_calibration_prompt_limit=4096, _g1_calibration_context_stats=[],
            _g1_calibration_tokenizer=SimpleNamespace(encode=lambda p, **kw: list(map(ord, p))),
            llm_engine=SimpleNamespace(model_config=SimpleNamespace(max_model_len=capacity)))
        def generate(inputs, params, **kw):
            engine.inputs = inputs; engine.params = params
            return [SimpleNamespace(outputs=[SimpleNamespace(text='<answer>ok</answer>')]) for _ in inputs]
        engine.generate = generate
        return engine

    def test_long_last_turn_fits_without_decode_reencode_or_budget_overflow(self):
        engine = self.engine()
        result = namespace['gen_batch'](engine, ['a' * 4400 + 'tail'], 500, True, ['</answer>'])
        ids = engine.inputs[0]['prompt_token_ids']
        self.assertEqual(len(ids), 4096); self.assertEqual(ids[-4:], list(map(ord, 'tail')))
        self.assertLessEqual(len(ids) + engine.params['max_tokens'], 4608)
        self.assertEqual(engine._g1_calibration_context_stats[0]['truncated'], 1)
        self.assertEqual(result, ['<answer>ok</answer>'])

    def test_incompatible_capacity_refused_before_generation(self):
        with self.assertRaisesRegex(ValueError, 'exceeds engine context'):
            namespace['gen_batch'](self.engine(4096), ['prompt'], 500, True, [])

    def test_historical_default_still_passes_original_string_prompts(self):
        engine = self.engine(); del engine._g1_calibration_prompt_limit
        namespace['gen_batch'](engine, ['original'], 256, True, [])
        self.assertEqual(engine.inputs, ['original'])


if __name__ == '__main__':
    unittest.main()
