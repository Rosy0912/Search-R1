import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from .paired_evaluation import attach_question_ids, save_evaluation


class PairedCaptureTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.env = patch.dict(os.environ, dict(SEARCH_R1_BASELINE_OUTPUT=self.tmp.name,
            SEARCH_R1_BASELINE_RUN_ID='test', SLURM_JOB_ID='0', SEARCH_R1_BASELINE_COMMIT='test',
            SEARCH_R1_BASELINE_CONFIG_SHA256='test', SEARCH_R1_BASELINE_DATA_IDENTITY='test'))
        self.env.start(); self.addCleanup(self.env.stop)
        self.rows = [dict(source='test', em=float(i % 2)) for i in range(512)]
        attach_question_ids(self.rows, [f'dev:{i}' for i in range(512)])

    def test_roundtrip_and_duplicate_write_refusal(self):
        record = save_evaluation(self.rows, {'val/em/all': .5}, 150, 512)
        raw = (Path(self.tmp.name)/'evaluations/dev512-outer-000150.jsonl').read_bytes()
        self.assertEqual(record['rows_sha256'], hashlib.sha256(raw).hexdigest())
        self.assertEqual([json.loads(x) for x in raw.splitlines()], self.rows)
        self.assertEqual(record['completed_outer'], 150)
        with self.assertRaises(FileExistsError):
            save_evaluation(self.rows, {'val/em/all': .5}, 150, 512)

    def test_missing_or_duplicate_question_rejected(self):
        with self.assertRaises(ValueError):
            save_evaluation(self.rows[:-1], {'val/em/all': .5}, 150, 512)
        self.rows[-1]['question_id'] = self.rows[0]['question_id']
        with self.assertRaises(ValueError):
            save_evaluation(self.rows, {'val/em/all': .5}, 150, 512)

    def test_changed_aggregate_rejected(self):
        with self.assertRaises(ValueError):
            save_evaluation(self.rows, {'val/em/all': .51}, 150, 512)

    def test_id_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            attach_question_ids(self.rows, ['one'])


if __name__ == '__main__': unittest.main()
