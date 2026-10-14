"""Reject unrelated/live failures before the specific storage-only recovery."""
import unittest
from unittest.mock import patch
from reproduction.hpc3_qprobe import storage_startup_failure as failure
from reproduction.hpc3_qprobe import recover_pretraining as recovery


class StorageFailureGateTests(unittest.TestCase):
    def test_only_exact_actual_terminal_accepted(self):
        raw='|'.join(failure.TERMINAL)+'\n'
        failure.validate_terminal(raw)
        for bad in ('',raw+raw,raw.replace('FAILED','RUNNING'),raw.replace('14090','21930'),raw.replace('ACD1-15','ACD1-12'),raw.replace('18:19:44','18:19:45'),raw.replace('1:0','0:0')):
            with self.assertRaises(ValueError):failure.validate_terminal(bad)

    def test_observer_cannot_archive_or_verify(self):
        with patch.object(failure.os,'getuid',return_value=21930),patch.object(failure.subprocess,'check_output') as read:
            with self.assertRaises(ValueError):failure.verify()
            read.assert_not_called()

    def test_storage_recovery_uses_new_exact_target_and_original_source(self):
        saved=(recovery.JOB,recovery.FAILED,recovery.TARGET,recovery.FAILED_COMMIT)
        try:
            recovery.select_failed_job('718288')
            self.assertEqual(recovery.FAILED,failure.BASE)
            self.assertEqual(recovery.FAILED_COMMIT,failure.SOURCE)
            self.assertNotEqual(recovery.FAILED,recovery.TARGET)
            self.assertTrue(str(recovery.TARGET).endswith('recovery718288-20261014'))
        finally:
            recovery.JOB,recovery.FAILED,recovery.TARGET,recovery.FAILED_COMMIT=saved


if __name__=='__main__':unittest.main()
