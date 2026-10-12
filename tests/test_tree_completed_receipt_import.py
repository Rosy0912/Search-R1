"""Regression for the post-Adam NameError, without model initialization."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1]/'search_r1/baselines/tree_guard_continue.py'

class CompletedReceiptImportTest(unittest.TestCase):
    def test_prevalidated_actor_result_is_written_after_real_update(self):
        spec = importlib.util.spec_from_file_location('search_r1.baselines.tree_guard_continue', SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        stub = ModuleType('search_r1.baselines.training_state')
        def atomic_save(value, path, json_format=False):
            self.assertTrue(json_format)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))
        stub.atomic_save = atomic_save
        trainer = SimpleNamespace(training_attempts=1, global_steps=1,
            config=SimpleNamespace(actor_rollout_ref=SimpleNamespace(actor={})))
        output = SimpleNamespace(meta_info={'tree_fresh_actor_result':{'ranks':[{'optimizer_calls':20}]}})
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, SEARCH_R1_BASELINE_OUTPUT=directory), \
                patch.dict(sys.modules, {'search_r1.baselines.training_state':stub}), \
                patch.object(module, 'validate_completed_result', return_value={'0':20}), \
                patch.object(module, 'round_progress_enabled', return_value=False):
            module.record_completed_result(trainer, output)
            receipt = json.loads((Path(directory)/'attempt-results/attempt-000001.json').read_text())
            self.assertEqual(receipt['actual_writes_this_round'], 20)
            self.assertEqual(receipt['accepted_outer'], 1)
            self.assertFalse(receipt['automatic_rollback'])
            self.assertEqual(trainer.completed_optimizer_calls_per_rank, {'0':20})

if __name__ == '__main__':
    unittest.main()
