"""Execute the actual timing helper with its actual standard-library imports."""
import ast
import contextlib
import io
import json
from pathlib import Path
import unittest

SOURCE = Path(__file__).resolve().parents[3] / 'verl/trainer/ppo/ray_trainer.py'

class Check(unittest.TestCase):
    def helper(self):
        tree = ast.parse(SOURCE.read_text())
        nodes = [n for n in tree.body if
                 (isinstance(n, ast.Import) and all(a.name in ('time', 'json', 'os') for a in n.names)) or
                 (isinstance(n, ast.ImportFrom) and n.module == 'contextlib') or
                 (isinstance(n, ast.FunctionDef) and n.name == '_driver_cpu_stage')]
        scope = {}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), scope)
        return scope['_driver_cpu_stage']

    def test_success_flushes_start_and_complete(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.helper()('prepare_behavior_mu', 4):
            pass
        rows = [json.loads(x.split('=', 1)[1]) for x in output.getvalue().splitlines()]
        self.assertEqual([r['event'] for r in rows], ['start', 'complete'])
        self.assertTrue(rows[1]['successful'])
        self.assertGreaterEqual(rows[1]['wall_seconds'], 0)

    def test_programming_failure_is_preserved(self):
        output = io.StringIO()
        with self.assertRaisesRegex(RuntimeError, 'fault'), contextlib.redirect_stdout(output):
            with self.helper()('prepare_behavior_mu', 4):
                raise RuntimeError('fault')
        complete = json.loads(output.getvalue().splitlines()[-1].split('=', 1)[1])
        self.assertFalse(complete['successful'])

if __name__ == '__main__':
    unittest.main()
