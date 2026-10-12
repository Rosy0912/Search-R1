"""Exercise the actual sparse-save branch without loading GPU dependencies."""
import ast
import contextlib
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

ROOT = Path(__file__).resolve().parents[3]


class RecoveryBoundary(unittest.TestCase):
    def test_sparse_hf_schedule_still_saves_native_boundary(self):
        source = ast.parse((ROOT / 'verl/trainer/ppo/ray_trainer.py').read_text())
        fit = next(n for n in ast.walk(source) if isinstance(n, ast.FunctionDef) and n.name == 'fit')
        assignment = next(n for n in ast.walk(fit) if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == 'rolling_state_due' for t in n.targets))
        branch = next(n for n in ast.walk(fit) if isinstance(n, ast.If)
                      and isinstance(n.test, ast.Name) and n.test.id == 'rolling_state_due')
        code = compile(ast.fix_missing_locations(ast.Module(body=[assignment, branch], type_ignores=[])),
                       'actual-training-save-branch', 'exec')
        for hf_due in (False, True):
            with self.subTest(hf_due=hf_due):
                save = Mock()
                trainer = SimpleNamespace(global_steps=4, _save_complete_training_state=save,
                    config=SimpleNamespace(actor_rollout_ref=SimpleNamespace(
                        actor={'recursive_document_objective': True}),
                        trainer=type('Config', (dict,), {'default_local_dir': '/remote-only'})
                            (training_state_enabled=True)))
                namespace = dict(self=trainer, save_due=hf_due, metrics={}, timing_raw={},
                                 _timer=lambda *args: contextlib.nullcontext())
                with patch('search_r1.baselines.recursive_rolling.prune', return_value=[]) as prune:
                    exec(code, namespace)
                if hf_due:
                    save.assert_not_called()  # _save_checkpoint already saved it.
                else:
                    save.assert_called_once_with(hf_checkpoint_saved=False)
                prune.assert_called_once_with('/remote-only', 4)

    def test_http_timeout_and_errors_are_visible(self):
        source = ast.parse((ROOT / 'search_r1/llm_agent/generation.py').read_text())
        method = next(n for n in ast.walk(source) if isinstance(n, ast.FunctionDef) and n.name == '_batch_search')
        requests = Mock()
        namespace = {'requests': requests}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                     'actual-retrieval-method', 'exec'), namespace)
        manager = SimpleNamespace(config=SimpleNamespace(topk=3, search_url='http://owned/retrieve'))
        response = requests.post.return_value
        response.raise_for_status.side_effect = RuntimeError('retrieval failed')
        with self.assertRaisesRegex(RuntimeError, 'retrieval failed'):
            namespace['_batch_search'](manager, ['question'])
        self.assertEqual(requests.post.call_args.kwargs['timeout'], (10, 300))
        response.json.assert_not_called()
        self.assertEqual(manager.retrieval_http_calls, 1)


if __name__ == '__main__':
    unittest.main()
