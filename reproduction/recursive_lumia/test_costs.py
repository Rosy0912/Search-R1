import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from .costs import action_costs,scheduler_costs,collect


class CostEvidenceTest(unittest.TestCase):
    def test_interleaved_record_preserves_valid_counts_and_completed_job_gap(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'log'
            valid='TREE_ACTION_TIMING='+json.dumps(dict(successful=True,cost_delta=dict(raw_emitted_tokens=100)))+'\n'
            path.write_text(valid+'TREE_ACTION_TIMING={"successful":true,"broken":"a\x1bb"}\n'+valid)
            entries=[dict(label='finished',job='1',log=str(path))]
            with patch('subprocess.check_output',return_value='1|COMPLETED|3600|gres/gpu=8|\n'):
                result=collect(entries)
            self.assertEqual(result['recorded_raw_emitted_tokens'],200)
            self.assertEqual(result['allocated_gpu_hours'],8)
            self.assertTrue(result['token_total_is_lower_bound'])
            record=result['entries'][0]['tokens']
            self.assertEqual(record['malformed_action_records'],1)
            self.assertEqual(record['malformed_record_examples'][0]['line'],2)

    def test_completed_truncated_ledger_is_not_exhaustive(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'log';path.write_text('TREE_ACTION_TIMING={')
            with patch('subprocess.check_output',return_value='1|COMPLETED|1|gres/gpu=8|\n'):
                result=collect([dict(job='1',log=str(path))])
            self.assertTrue(result['token_total_is_lower_bound'])
            self.assertEqual(result['entries'][0]['tokens']['incomplete_action_records'],1)

    def test_decodable_invalid_cost_still_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'log'
            for invalid in (-1,float('nan'),float('inf'),'100'):
                path.write_text('TREE_ACTION_TIMING='+json.dumps(dict(successful=True,cost_delta=dict(raw_emitted_tokens=invalid)))+'\n')
                with self.assertRaises(ValueError):action_costs(path)

    def test_counts_new_emissions_once_and_drops_unfinished_log_record(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'log'
            action=dict(successful=True,cost_delta=dict(raw_emitted_tokens=100,raw_retained_tokens=90))
            path.write_text('TREE_TIMING={"policy_tokens":500}\n'+
                'TREE_ACTION_TIMING='+json.dumps(action)+'\n'+
                'step:1 - rollout/raw_emitted_tokens:100\n'+
                'TREE_ACTION_TIMING={"unfinished":')
            result=action_costs(path)
            self.assertEqual(result['recorded_actions'],1)
            self.assertEqual(result['totals']['raw_emitted_tokens'],100)
            self.assertTrue(result['interrupted_or_inflight_action_may_be_unrecorded'])

    def test_scheduler_gpu_time_does_not_double_count_typed_gres_or_cpu_jobs(self):
        result=scheduler_costs('1|RUNNING|3600|cpu=64,gres/gpu=8,gres/gpu:a100=8|\n2|COMPLETED|7200|cpu=48,mem=240G|\n')
        self.assertEqual(result['1']['allocated_gpu_hours'],8.)
        self.assertEqual(result['2']['allocated_gpu_hours'],0.)

    def test_pending_dependency_without_stdout_is_zero_work(self):
        with tempfile.TemporaryDirectory() as directory:
            entries=[dict(label='queued',job='3',log=str(Path(directory)/'not-started.log'))]
            with patch('subprocess.check_output',return_value='3|PENDING|0||\n'):
                result=collect(entries)
            self.assertEqual(result['allocated_gpu_hours'],0)
            self.assertEqual(result['recorded_raw_emitted_tokens'],0)
            self.assertFalse(result['token_total_is_lower_bound'])
            for state in ('RUNNING','COMPLETED','FAILED'):
                with patch('subprocess.check_output',return_value=f'3|{state}|10|gres/gpu=8|\n'):
                    with self.assertRaises(FileNotFoundError):collect(entries)

    def test_cancelled_before_allocation_is_zero_but_missing_allocated_log_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            entries=[dict(label='cancelled',job='4',log=str(Path(directory)/'never-created.log'))]
            for state in ('CANCELLED','CANCELLED by 1066'):
                with patch('subprocess.check_output',return_value=f'4|{state}|0||\n'):
                    result=collect(entries)
                self.assertEqual(result['allocated_gpu_hours'],0)
                self.assertEqual(result['recorded_raw_emitted_tokens'],0)
                self.assertFalse(result['token_total_is_lower_bound'])
            for seconds,resources in ((0,'gres/gpu=8'),(0,'cpu=8'),(1,'')):
                with patch('subprocess.check_output',return_value=f'4|CANCELLED by 1066|{seconds}|{resources}|\n'):
                    with self.assertRaises(FileNotFoundError):collect(entries)


if __name__=='__main__':unittest.main()
