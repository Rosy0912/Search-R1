import json
import tempfile
import unittest
from pathlib import Path
from accept import collect_rows, collect_gpu_records


class EvidenceTests(unittest.TestCase):
    def test_coverage_and_aggregates(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'rows.jsonl'
            rows = [dict(final_index=i, source='nq', f1=.5, em=i,
                         answer_format=1, format_rate=1, generated_tokens=10, search_calls=2)
                    for i in range(2)]
            path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            result = collect_rows(path, 2)
            self.assertEqual(result['all']['count'], 2)
            self.assertEqual(result['nq']['em'], 1)
            path.write_text(json.dumps(rows[0]) + '\n' + json.dumps(rows[0]) + '\n')
            with self.assertRaises(AssertionError):
                collect_rows(path, 2)

    def test_nonfinite_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'rows.jsonl'
            row = dict(final_index=0, source='nq', f1=float('nan'), em=0,
                       answer_format=1, format_rate=1, generated_tokens=10, search_calls=2)
            path.write_text(json.dumps(row) + '\n')
            with self.assertRaises(AssertionError):
                collect_rows(path, 1)

    def test_worker_duplicate_and_conflict(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'worker.out'
            row = dict(event='complete', rank=0, pid=12, call=1, generated_tokens=10)
            line = 'prefix FINAL51713_GPU=' + json.dumps(row) + '\n'
            path.write_text(line + line)
            self.assertEqual(len(collect_gpu_records([path])), 1)
            row['generated_tokens'] = 11
            path.write_text(line + 'FINAL51713_GPU=' + json.dumps(row) + '\n')
            with self.assertRaises(AssertionError):
                collect_gpu_records([path])


if __name__ == '__main__':
    unittest.main()
