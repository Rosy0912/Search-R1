import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from reproduction.hpc3_qprobe import recovery718509_controller as controller
from reproduction.hpc3_qprobe import telemetry_failure718509 as failure,recover_pretraining as recovery


class ControllerTests(unittest.TestCase):
    def test_complete_submission_records_success_without_repeating_any_action(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'launch-claim.json').write_text(json.dumps(dict(commit='fixture',failed_job='718509')))
            env=dict(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            born=dict(pid=os.getpid(),sid=os.getpid(),uid=14090,start_ticks=123)
            with patch.object(controller,'DIRECTORY',root),patch.object(controller,'source',return_value=root),\
                 patch.dict(os.environ,env),patch('reproduction.a800.owned_processes.identity',return_value=born),\
                 patch.object(failure,'archive',return_value=dict(teacher_cache=dict(raw_bytes=1351108800))) as archive,\
                 patch.object(failure,'submit_cleanup',return_value=dict(job='fixture-cleanup')) as cleanup,\
                 patch.object(recovery,'select_failed_job'),patch.object(recovery,'TARGET',root/'target'),\
                 patch.object(recovery,'fresh',return_value=dict(target=str(root/'target'))) as prepare,\
                 patch.object(recovery,'submit',return_value=dict(job='fixture-training')) as submit:
                value=controller.run('fixture')
            self.assertEqual(json.loads((root/'state.json').read_text())['stage'],'COMPLETED_ONCE_NATIVE4_SUBMISSION')
            self.assertEqual(json.loads((root/'complete.json').read_text()),value)
            self.assertFalse(value['training_progress_accepted'])
            for action in (archive,cleanup,prepare,submit):action.assert_called_once()

    def test_actual_failure_records_exit_and_never_submits_recovery(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'launch-claim.json').write_text(json.dumps(dict(commit='fixture',failed_job='718509')))
            env=dict(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            born=dict(pid=os.getpid(),sid=os.getpid(),uid=14090,start_ticks=123)
            with patch.object(controller,'DIRECTORY',root),patch.object(controller,'source',return_value=root),\
                 patch.dict(os.environ,env),patch('reproduction.a800.owned_processes.identity',return_value=born),\
                 patch.object(failure,'archive',side_effect=ValueError('source changed')),\
                 patch.object(recovery,'submit') as submit:
                with self.assertRaisesRegex(ValueError,'source changed'):controller.run('fixture')
            state=json.loads((root/'state.json').read_text());self.assertEqual(state['stage'],'FAILED')
            self.assertEqual(state['exit_code'],1);submit.assert_not_called()
            self.assertFalse((root/'complete.json').exists())


if __name__=='__main__':unittest.main()
