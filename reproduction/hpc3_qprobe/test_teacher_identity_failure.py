import unittest
from unittest.mock import patch
from reproduction.hpc3_qprobe import teacher_identity_failure as failure, recover_pretraining as recovery


class TeacherFailureTests(unittest.TestCase):
    def test_only_exact_actual_terminal_accepted(self):
        raw='|'.join(failure.TERMINAL)+'\n';failure.validate_terminal(raw)
        for bad in ('',raw+raw,raw.replace('FAILED','RUNNING'),raw.replace('1:0','0:0'),
                    raw.replace('14090','21930'),raw.replace('19:05:06','19:05:07')):
            with self.assertRaises(ValueError):failure.validate_terminal(bad)

    def test_observer_and_live_owner_fail_before_files(self):
        with patch.object(failure.os,'getuid',return_value=21930),patch.object(failure.subprocess,'check_output') as query:
            with self.assertRaises(ValueError):failure.verify()
            query.assert_not_called()
        with patch.object(failure.os,'getuid',return_value=14090),\
             patch.object(failure.subprocess,'check_output',return_value='|'.join(failure.TERMINAL).replace('FAILED','RUNNING')),\
             patch.object(failure.Path,'read_text') as read:
            with self.assertRaises(ValueError):failure.verify()
            read.assert_not_called()

    def test_recovery_uses_separate_actual_failed_base_and_fresh_target(self):
        saved=(recovery.JOB,recovery.FAILED,recovery.TARGET,recovery.FAILED_COMMIT)
        try:
            recovery.select_failed_job('718322')
            self.assertEqual(recovery.FAILED,failure.BASE);self.assertEqual(recovery.FAILED_COMMIT,failure.SOURCE)
            self.assertNotEqual(recovery.FAILED,recovery.TARGET)
        finally:recovery.JOB,recovery.FAILED,recovery.TARGET,recovery.FAILED_COMMIT=saved


if __name__=='__main__':unittest.main()
