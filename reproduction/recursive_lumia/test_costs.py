import json
from pathlib import Path
import tempfile
import unittest
from .costs import action_costs,scheduler_costs


class CostEvidenceTest(unittest.TestCase):
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


if __name__=='__main__':unittest.main()
