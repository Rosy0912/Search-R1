"""Execute actual continuation callsites and fit initialization on CPU."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from omegaconf import OmegaConf
from verl.utils.weight_continuation import model_initialization_path

ROOT = Path(__file__).resolve().parents[2]


class ContinuationWiring(unittest.TestCase):
    def test_actual_actor_reference_calls(self):
        tree = ast.parse((ROOT/'verl/workers/fsdp_workers.py').read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == 'model_initialization_path']
        self.assertEqual(len(calls), 2)
        config = OmegaConf.create({'model': {'path': '/base', 'actor_checkpoint_path': '/checkpoint200'}})
        roles = set()
        for node in calls:
            role = ast.literal_eval(next(k.value for k in node.keywords if k.arg == 'actor'))
            roles.add(role)
            value = eval(compile(ast.Expression(node), 'actual-worker', 'eval'),
                         {'self': SimpleNamespace(config=config),
                          'model_initialization_path': model_initialization_path})
            self.assertEqual(value, '/checkpoint200' if role else '/base')
        self.assertEqual(roles, {False, True})

    def test_actual_fit_starts201(self):
        tree = ast.parse((ROOT/'verl/trainer/ppo/ray_trainer.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RayPPOTrainer')
        fit = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'fit')
        end = next(i for i, n in enumerate(fit.body) if isinstance(n, ast.AugAssign)
                   and ast.unparse(n.target) == 'self.global_steps')
        function = ast.FunctionDef(name='initialize', args=ast.arguments(posonlyargs=[],
            args=[ast.arg(arg='self')], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=fit.body[:end+1], decorator_list=[])
        ns = {}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                     'actual-fit-init', 'exec'), ns)
        config = OmegaConf.create({'trainer': {'weight_continuation_step': 200, 'acceptance_step': 205},
            'actor_rollout_ref': {'model': {'path': '/base', 'actor_checkpoint_path': '/checkpoint200'}}})
        trainer = SimpleNamespace(config=config, total_training_steps=301, logger=None, val_reward_fn=None)
        ns['initialize'](trainer)
        self.assertEqual(trainer.global_steps, 201)


if __name__ == '__main__':
    unittest.main()
