import subprocess
import unittest
from unittest.mock import patch

from reproduction.hpc3_qprobe.recover_pretraining import read_terminal_job


class TerminalReadbackTests(unittest.TestCase):
    detail='JobId=717554 JobState=FAILED ExitCode=1:0 StartTime=2026-10-14T13:26:11 EndTime=2026-10-14T13:31:00 NodeList=ACD1-56'
    accounting='717554|FAILED|1:0|jxu957|14090|jxu957|formal-user|2026-10-14T13:26:11|2026-10-14T13:31:00|ACD1-56\n'

    def test_aged_out_job_requires_current_exact_accounting(self):
        missing=subprocess.CompletedProcess([],1,'','Invalid job id specified')
        with patch('subprocess.run',return_value=missing),patch('subprocess.check_output',return_value=self.accounting):
            detail,proof=read_terminal_job(self.detail)
        self.assertEqual(detail,self.detail)
        self.assertEqual(proof['kind'],'fresh_accounting_and_preserved_owner_birth')

    def test_missing_job_does_not_prove_failure_or_owner(self):
        missing=subprocess.CompletedProcess([],1,'','Invalid job id specified')
        for bad in ('',self.accounting.replace('FAILED','RUNNING'),self.accounting.replace('14090','21930'),
                    self.accounting+self.accounting,self.accounting.replace('ACD1-56','ACD1-4')):
            with self.subTest(bad=bad),patch('subprocess.run',return_value=missing),patch('subprocess.check_output',return_value=bad):
                with self.assertRaises(ValueError):read_terminal_job(self.detail)


if __name__=='__main__':unittest.main()
