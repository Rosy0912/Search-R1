import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from .compare import compare


class PairedComparisonTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def fixture(self, name, reverse=False, outer=150):
        # Four paired outcomes each occur128times; reordering must not change pairing.
        rows = [dict(question_id=f'dev:{i}', source='nq', em=float(i%4 in
            ((0,1) if name == 'a' else (0,2)))) for i in range(512)]
        if reverse: rows.reverse()
        raw = ''.join(json.dumps(row)+'\n' for row in rows).encode()
        path = Path(self.tmp.name)/(name+'.jsonl'); path.write_bytes(raw)
        meta = dict(split='dev512', count=512, completed_outer=outer, data_identity='fixed', em=.5,
            rows_sha256=hashlib.sha256(raw).hexdigest(), ordered_ids_sha256=hashlib.sha256(
                json.dumps([row['question_id'] for row in rows],ensure_ascii=False).encode()).hexdigest())
        path.with_suffix('.json').write_text(json.dumps(meta))
        return path

    def test_exact_pairs_ignore_row_order(self):
        result = compare(self.fixture('a'),self.fixture('b',reverse=True))['overall']
        for key in ('both_correct','both_incorrect','baseline_only_correct','candidate_only_correct'):
            self.assertEqual(result[key],128)
        self.assertEqual(result['candidate_minus_baseline_percentage_points'],0.)

    def test_changed_round_rejected(self):
        with self.assertRaisesRegex(ValueError,'completed_outer'):
            compare(self.fixture('a'),self.fixture('b',outer=100))

    def test_modified_rows_rejected(self):
        a,b=self.fixture('a'),self.fixture('b')
        b.write_bytes(b.read_bytes()+b'\n')
        with self.assertRaisesRegex(ValueError,'bytes'):
            compare(a,b)

    def test_different_question_set_rejected_even_with_matching_aggregate(self):
        a,b=self.fixture('a'),self.fixture('b')
        rows=[json.loads(line) for line in b.read_text().splitlines()]
        rows[0]['question_id']='different-question'
        raw=''.join(json.dumps(row)+'\n' for row in rows).encode();b.write_bytes(raw)
        meta=json.loads(b.with_suffix('.json').read_text())
        meta['rows_sha256']=hashlib.sha256(raw).hexdigest()
        meta['ordered_ids_sha256']=hashlib.sha256(json.dumps([r['question_id'] for r in rows],ensure_ascii=False).encode()).hexdigest()
        b.with_suffix('.json').write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError,'same questions'):
            compare(a,b)


if __name__ == '__main__': unittest.main()
