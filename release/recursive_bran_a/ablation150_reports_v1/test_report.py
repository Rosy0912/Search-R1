"""Meaningful aggregation checks; no production dependencies or model imports."""
import json
from pathlib import Path
import tempfile
import unittest

import report


def metrics(hits=246):
    return {'val/qa_count/all': 512., 'val/qa_count/nq': 256.,
            'val/qa_count/hotpotqa': 256., 'val/em/all': hits/512,
            'val/em/nq': 108/256, 'val/em/hotpotqa': (hits-108)/256}


class ReportTest(unittest.TestCase):
    def test_full_aggregate_is_integer_and_balanced(self):
        value = report.evaluation(metrics())
        self.assertEqual(value['all']['hits'], 246)
        self.assertEqual(value['nq']['hits'], 108)
        self.assertEqual(value['hotpotqa']['hits'], 138)
        self.assertIsNone(report.evaluation({}))
        partial = metrics()
        partial['val/qa_count/all'] = 511
        with self.assertRaises(ValueError):
            report.evaluation(partial)

    def test_native_cost_and_real_skips(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'attempt-results').mkdir()
            lines = []
            cumulative = 0
            for outer in range(1, 6):
                writes = 19 if outer == 3 else 20
                cumulative += writes
                ranks = []
                for rank in range(8):
                    ledger = [{'index': index, 'actual_writes': int(index <= writes)}
                              for index in range(1, 21)]
                    ranks.append(dict(rank=rank, optimizer_calls=writes, opportunities_skipped=20-writes,
                                      opportunity_ledger=ledger))
                record = dict(accepted_outer=outer, all_rank_result=dict(
                    identity=dict(run_id='run', config_sha256='config'), decision='COMPLETED_OUTER', ranks=ranks),
                    actual_completed_optimizer_calls_per_rank={str(r): cumulative for r in range(8)})
                (root/'attempt-results'/f'attempt-{outer:06d}.json').write_text(json.dumps(record))
                row = {'timing_s/step': 10., 'timing_s/gen': 6., 'rollout/raw_emitted_tokens': 100.,
                       'training/actual_adam_calls_per_rank': float(cumulative)}
                if outer == 5:
                    row.update(metrics())
                lines.append('step:'+str(outer)+' - '+' - '.join(f'{key}:{value}' for key, value in row.items()))
            log = root/'training.log'
            log.write_text('\n'.join(lines)+'\n')
            condition = dict(id='C1', run_id='run', config_sha256='config', world_size=8,
                             logs=[str(log)], output_roots=[str(root)])
            result = report.build(dict(conditions=[condition], milestones=[5]))
            row = result['conditions']['C1']['milestones']['5']
            self.assertTrue(result['analysis_complete'])
            self.assertEqual(row['receipt_adam_each_rank'], 99)
            self.assertEqual(row['receipt_skips_each_rank'], 1)
            self.assertEqual(row['generated_tokens_recorded_sum'], 500)
            self.assertAlmostEqual(row['allocated_gpu_hours_from_recorded_outer_wall'], 50*8/3600)
            (root/'attempt-results'/'attempt-000003.json').unlink()
            row = report.build(dict(conditions=[condition], milestones=[5]))['conditions']['C1']['milestones']['5']
            self.assertIsNone(row['receipt_adam_each_rank'])
            self.assertEqual(row['native_actual_adam_each_rank'], 99)

    def test_conflicting_native_replay_is_not_last_wins(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'log'
            path.write_text('step:5 - timing_s/step:10\nstep:5 - timing_s/step:11\n')
            with self.assertRaises(ValueError):
                report.native_log([path])

    def test_existing_published_A150_shape(self):
        root = Path(__file__).resolve().parents[3]
        path = root/'reports/h100-outer150-20261014/aggregate.json'
        condition = dict(id='A', run_id='history', config_sha256='history', world_size=8,
                         aggregate_files={'150': str(path)})
        summary = report.build(dict(conditions=[condition], milestones=[150]))
        row = summary['conditions']['A']['milestones']['150']
        self.assertEqual(row['dev512']['all']['hits'], 246)
        self.assertEqual(row['dev512']['nq']['hits'], 108)
        self.assertIsNone(row['generated_tokens_recorded_sum'])
        self.assertFalse(row['receipts_complete_through_outer'])


if __name__ == '__main__':
    unittest.main()
