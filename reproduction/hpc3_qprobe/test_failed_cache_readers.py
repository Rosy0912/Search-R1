"""Regression for the actual nested-creator cleanup failure, CPU only."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from reproduction.hpc3_qprobe.cleanup_failed_717827 import check_failed_readers


class FailedCacheReaderTests(unittest.TestCase):
    def check(self, actual, mutation=None):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);(base/'logs').mkdir()
            stage=dict(run_id='failed-run',slurm_job_id='717827',uid=14090,pid=99999999,start_ticks=1)
            corpus=dict(run_id='failed-run',slurm_job_id='717827',uid=14090,
                creator=dict(pid=99999998,start_ticks=2,uid=14090))
            if mutation:mutation(corpus)
            (base/'logs/retrieval-staging-claim.json').write_text(json.dumps(stage))
            (base/'logs/corpus-cache-claim.json').write_text(json.dumps(corpus))
            def identity(pid):
                if pid==stage['pid']:raise FileNotFoundError
                return actual
            with patch('reproduction.a800.owned_processes.identity',side_effect=identity):
                check_failed_readers(base)

    def test_actual_nested_creator_schema_and_expired_or_zombie_process_pass(self):
        self.check(dict(start_ticks=3,state='R'))
        self.check(dict(start_ticks=2,state='Z'))

    def test_matching_live_creator_is_rejected(self):
        with self.assertRaisesRegex(AssertionError,'Cache creator remains live'):
            self.check(dict(start_ticks=2,state='R'))

    def test_mismatched_run_or_creator_owner_is_rejected(self):
        for mutation in (lambda c:c.update(run_id='other-run'),
                         lambda c:c['creator'].update(uid=21930)):
            with self.assertRaises(AssertionError):self.check(dict(start_ticks=3,state='R'),mutation)


if __name__=='__main__':unittest.main()
