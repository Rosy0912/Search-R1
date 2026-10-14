"""CPU negative gates and emergency start-before-receipt regression."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from reproduction.hpc3_qprobe import observed_startup_failures as observed
from reproduction.hpc3_qprobe import cleanup_failed_717827 as cleanup


class TerminalAndOwnerTests(unittest.TestCase):
    row='717938|FAILED|1:0|jxu957|14090|jxu957|formal-user|emergency_acd|2026-10-14T15:36:25|2026-10-14T15:42:47|ACD1-12\n'

    def test_current_exact_failure_accepts_and_running_wrong_owner_node_or_duplicates_reject(self):
        with patch.object(observed.subprocess,'check_output',return_value=self.row):
            self.assertEqual(observed.actual_terminal('717938')['row'][0],'717938')
        for bad in (self.row.replace('FAILED','RUNNING'),self.row.replace('14090','21930'),
                    self.row.replace('ACD1-12','ACD1-67'),self.row+self.row,''):
            with patch.object(observed.subprocess,'check_output',return_value=bad),self.assertRaises(ValueError):
                observed.actual_terminal('717938')

    def test_observer_cannot_adopt_or_read_owner_recovery_or_submit_retry(self):
        with patch.object(observed.os,'getuid',return_value=21930),patch.object(observed,'verify') as verify:
            for operation in (lambda:observed.adopt('717938'),lambda:observed.owner_verified('717938'),lambda:cleanup.submit_retry('a'*40)):
                with self.assertRaises(ValueError):operation()
            verify.assert_not_called()

    def test_unknown_failure_cannot_be_generalized(self):
        with self.assertRaises(ValueError):observed.base_for('123456')


class EmergencyRunnerReceiptRaceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.base=Path(self.temp.name)/'failed';self.base.mkdir()
        (self.base/'preparation').mkdir();self.shadow=self.base/'new-shadow';self.shadow.mkdir()
        self.claim=dict(commit='a'*40,retry_of='717934',command=['sbatch'])
        (self.base/'preparation'/(cleanup.RETRY_STEM+'-submit-claim.json')).write_text(json.dumps(self.claim))
        entry=self.base.parent/'qprobe-teacher-warmup200-20261014/source'/self.claim['commit']/'reproduction/hpc3_qprobe/cleanup_failed_717827.sh'
        self.detail=('JobId=800000 JobState=RUNNING UserId=jxu957(14090) Account=jxu957 QOS=formal-user '
            'NodeList=ACD1-67 JobName=qprobe-cleanup717934 WorkDir='+str(self.base)+' Command='+str(entry))

    def tearDown(self):self.temp.cleanup()

    def bind(self,detail=None):
        with patch.object(cleanup,'FAILED',self.base),patch.object(cleanup,'RETRY_SHADOW',self.shadow),\
             patch.object(cleanup.os,'getuid',return_value=14090),patch.object(cleanup.subprocess,'check_output',return_value=detail or self.detail):
            cleanup.bind_native_retry_runner('800000')

    def test_real_job_can_bind_before_control_receipt_is_written(self):
        self.bind()
        proof=json.loads((self.shadow/'runner-job-binding.json').read_text())
        self.assertEqual(proof['job'],'800000')
        self.assertFalse(proof['submit_receipt_observed'])
        self.assertFalse((self.base/'preparation'/(cleanup.RETRY_STEM+'-submission.json')).exists())

    def test_wrong_native_owner_source_or_node_and_later_receipt_mismatch_reject(self):
        for bad in (self.detail.replace('jxu957(14090)','mou176(21930)'),
                    self.detail.replace('ACD1-67','ACD1-12'),self.detail.replace('cleanup_failed_717827.sh','other.sh')):
            with self.assertRaises(AssertionError):self.bind(bad)
            self.assertFalse((self.shadow/'runner-job-binding.json').exists())
        receipt=self.base/'preparation'/(cleanup.RETRY_STEM+'-submission.json')
        receipt.write_text(json.dumps(dict(job='other-job',commit=self.claim['commit'],returncode=0)))
        with self.assertRaises(AssertionError):self.bind()
        self.assertFalse((self.shadow/'runner-job-binding.json').exists())


if __name__=='__main__':unittest.main()
