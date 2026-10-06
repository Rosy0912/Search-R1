"""Reduced scalar count regressions; no fixture is native training evidence.

The gate fixture stops before worker/checkpoint acceptance. Its purpose is to
exercise the production metric reducer and the live/saved count boundary.
"""
import copy
import json
import os
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from reproduction.common8 import runtime
from reproduction.common8.update_evidence import METRICS


SUCCESS = 'actor/optimizer_steps_this_update'
OPPORTUNITIES = 'actor/update_opportunities_this_attempt'
SKIPPED = 'actor/skipped_update_opportunities'
COUNT_FIELDS = (SUCCESS, OPPORTUNITIES, SKIPPED)


class _PastCountValidation(Exception):
    """Sentinel: the count boundary passed, not the full acceptance gate."""


class ReducedCountScalarTests(unittest.TestCase):
    def test_builtin_and_numpy_real_scalar_integer_counts_are_valid(self):
        for scalar in (int, float, np.int32, np.int64, np.uint64,
                       np.float16, np.float32, np.float64):
            for count in (0, 1, 13, 20):
                with self.subTest(scalar=scalar.__name__, count=count):
                    actual = runtime._validate_reduced_count(scalar(count), minimum=0, maximum=20)
                    self.assertIs(type(actual), int)
                    self.assertEqual(actual, count)

    def test_noncounts_are_rejected_before_conversion_can_make_them_valid(self):
        invalid = (True, False, np.bool_(True), np.bool_(False), '13', None,
                   np.array(13), np.array([13]), torch.tensor(13), [13],
                   complex(13, 0), float('nan'), float('inf'), float('-inf'),
                   np.float64('nan'), np.float64('inf'), np.float64('-inf'),
                   1.5, np.float32(1.5), Fraction(10**20 + 1, 10**20),
                   -1, 21, 10**1000)
        for index, value in enumerate(invalid):
            with self.subTest(index=index, value_type=type(value).__name__), self.assertRaises(AssertionError):
                runtime._validate_reduced_count(value, minimum=0, maximum=20)


class FiveUpdateReducedCountTests(unittest.TestCase):
    def reduced_metrics(self, successes):
        # This is the actual actor reduction function used by the trainer at
        # ray_trainer.py:158, rather than a builtin-float stand-in for it.
        from verl.trainer.ppo.ray_trainer import reduce_metrics
        values = {key: .1 for key in METRICS}
        values.update({SUCCESS: successes, OPPORTUNITIES: 20, SKIPPED: 20 - successes})
        reduced = reduce_metrics({key: [value] * 8 for key, value in values.items()})
        for key in COUNT_FIELDS:
            self.assertIs(type(reduced[key]), np.float64)
        return reduced

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        (self.base / 'logs').mkdir()
        self.checkpoint = self.base / 'checkpoint5'
        self.checkpoint.mkdir()
        run = dict(run_id='constructed-reduced-counts', profile='common8',
                   completed_updates=0, acceptance_update=5,
                   initial_probe_version='constructed-probe')
        (self.base / 'logs/run-manifest.json').write_text(json.dumps(run))
        proof = dict(run_id=run['run_id'], step=5, finite_all_saved_tensors=True,
                     files_sha256={})
        (self.checkpoint / 'complete.json').write_text(json.dumps(proof))
        self.records = [dict(step=step, probe_version=run['initial_probe_version'],
                             metrics=self.reduced_metrics(successes))
                        for step, successes in enumerate((20, 20, 20, 20, 20), 1)]
        self.saved = [dict(run_id=run['run_id'], step=row['step'],
                           probe_version=row['probe_version'],
                           metrics={key: float(value) for key, value in row['metrics'].items()})
                      for row in self.records]
        for context in (patch.dict(os.environ, SEARCH_G1_BASE=str(self.base)),
                        patch.object(runtime, 'assert_formal_run'),
                        patch.object(runtime, 'verify', return_value={}),
                        patch('reproduction.common8.behavior_runtime.read_gate_receipt', return_value={}),
                        patch('reproduction.common8.behavior_runtime.verify_write_receipt_links'),
                        patch('reproduction.common8.worker_evidence.read_console_form', return_value='')):
            context.start()
            self.addCleanup(context.stop)
        self.boundary_context = patch.object(runtime, 'budget_rows', side_effect=_PastCountValidation)
        self.boundary = self.boundary_context.start()
        self.addCleanup(self.boundary_context.stop)

    def write_saved(self):
        (self.base / 'logs/actual-updates.jsonl').write_text(
            ''.join(json.dumps(row, allow_nan=False) + '\n' for row in self.saved))

    def assert_count_boundary_passes(self):
        self.write_saved()
        with self.assertRaises(_PastCountValidation):
            runtime.accept_five_updates(self.checkpoint, self.records)
        self.boundary.assert_called_once()
        self.assertFalse((self.base / 'logs/five-updates-acceptance.json').exists())

    def test_actual_numpy_mean_all_twenty_counts_reach_later_gate(self):
        self.assert_count_boundary_passes()

    def test_fewer_successes_keep_all_twenty_opportunities_and_exact_skips(self):
        writes = (1, 13, 20, 3, 17)
        for row, saved, successes in zip(self.records, self.saved, writes):
            row['metrics'] = self.reduced_metrics(successes)
            saved['metrics'] = {key: float(value) for key, value in row['metrics'].items()}
        self.assertEqual(sum(row['metrics'][OPPORTUNITIES] for row in self.records), 100)
        self.assertEqual(sum(row['metrics'][SUCCESS] for row in self.records), sum(writes))
        self.assertLess(sum(writes), 100)
        self.assert_count_boundary_passes()

    def test_each_live_count_field_rejects_invalid_scalar_types_and_values(self):
        original = copy.deepcopy(self.records)
        for key in COUNT_FIELDS:
            valid = original[0]['metrics'][key]
            invalid = (True, False, np.bool_(False), str(int(valid)),
                       np.array(valid), np.array([valid]), torch.tensor(float(valid)),
                       np.float64('nan'), np.float64('inf'), np.float64('-inf'),
                       np.float64(valid + .5), -1, 21)
            for index, value in enumerate(invalid):
                with self.subTest(key=key, index=index, value_type=type(value).__name__):
                    self.records = copy.deepcopy(original)
                    self.records[0]['metrics'][key] = value
                    self.write_saved()
                    self.boundary.reset_mock()
                    with self.assertRaises(AssertionError):
                        runtime.accept_five_updates(self.checkpoint, self.records)
                    self.boundary.assert_not_called()

    def test_zero_write_incomplete_opportunities_or_wrong_skip_arithmetic_reject(self):
        for key, value in ((SUCCESS, np.float64(0)),
                           (OPPORTUNITIES, np.float64(19)),
                           (SKIPPED, np.float64(1))):
            with self.subTest(key=key):
                records = copy.deepcopy(self.records)
                records[0]['metrics'][key] = value
                self.write_saved()
                self.boundary.reset_mock()
                with self.assertRaises(AssertionError):
                    runtime.accept_five_updates(self.checkpoint, records)
                self.boundary.assert_not_called()

    def test_serialized_actual_metrics_must_still_equal_live_reduced_values(self):
        original = copy.deepcopy(self.saved)
        for key in COUNT_FIELDS:
            with self.subTest(key=key):
                self.saved = copy.deepcopy(original)
                self.saved[0]['metrics'][key] += 1
                self.write_saved()
                self.boundary.reset_mock()
                with self.assertRaises(AssertionError):
                    runtime.accept_five_updates(self.checkpoint, self.records)
                self.boundary.assert_not_called()


if __name__ == '__main__':
    unittest.main()
