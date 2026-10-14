import unittest
from unittest.mock import patch
from reproduction.hpc3_qprobe import telemetry_failure718509 as failure,recover_pretraining as recovery


class ActualFailureTests(unittest.TestCase):
    def test_exact_terminal_required(self):
        raw='|'.join(failure.TERMINAL)+'\n';failure.validate_terminal(raw)
        for bad in ('',raw+raw,raw.replace('FAILED','RUNNING'),raw.replace('1:0','0:0'),
                    raw.replace('14090','21930'),raw.replace('19:58:34','19:58:35'),raw.replace('ACD1-31','ACD1-1')):
            with self.assertRaises(ValueError):failure.validate_terminal(bad)

    def test_owner_and_live_job_fail_before_files(self):
        with patch.object(failure.os,'getuid',return_value=21930),patch.object(failure.subprocess,'check_output') as query:
            with self.assertRaises(ValueError):failure.verify()
            query.assert_not_called()
        with patch.object(failure.os,'getuid',return_value=14090),\
             patch.object(failure.subprocess,'check_output',return_value='|'.join(failure.TERMINAL).replace('FAILED','RUNNING')),\
             patch.object(failure.Path,'read_text') as read:
            with self.assertRaises(ValueError):failure.verify()
            read.assert_not_called()

    def test_specific_failure_and_fresh_target(self):
        saved=(recovery.JOB,recovery.FAILED,recovery.TARGET,recovery.FAILED_COMMIT)
        try:
            recovery.select_failed_job('718509')
            self.assertEqual(recovery.FAILED,failure.BASE);self.assertEqual(recovery.FAILED_COMMIT,failure.SOURCE)
            self.assertNotEqual(recovery.FAILED,recovery.TARGET)
        finally:recovery.JOB,recovery.FAILED,recovery.TARGET,recovery.FAILED_COMMIT=saved

    def test_cache_transport_fails_if_teacher_science_changed(self):
        identity=dict(scorer_sources={'scorer.py':'0'*64},rollout_sources={})
        with patch.object(failure,'preserved',return_value={}),\
             patch.object(failure.Path,'read_text',return_value=__import__('json').dumps(identity)),\
             patch.object(failure,'sha_file',return_value='1'*64),patch.object(failure.Path,'mkdir') as mkdir:
            with self.assertRaisesRegex(ValueError,'sampling/scoring changed'):failure.reuse_cache('/target','commit')
            mkdir.assert_not_called()


if __name__=='__main__':unittest.main()
