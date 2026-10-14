import json
import tempfile
import unittest
from pathlib import Path
from reproduction.common8.final_resume_journal import (
    FINAL_COUNT, make_binding, append_batch, load_prefix, aggregate_final)


class FinalResumeJournalTest(unittest.TestCase):
    def setUp(self):
        # Identical IDs deliberately occur at distinct canonical ordinals.
        self.ids = ['same-question' if i < 3 else 'q'+str(i) for i in range(FINAL_COUNT)]
        self.sources = ['nq' if i % 2 else 'hotpotqa' for i in range(FINAL_COUNT)]
        self.binding = make_binding('run', 'a'*64, 'b'*64, self.ids, self.sources)
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def row(self, i):
        return dict(ordinal=i, question_id=self.ids[i], data_source=self.sources[i],
            em=i % 2, f1=1. if i % 2 else .25, actual_calls=1, valid_action=2,
            turns=3, answer_present=True, answer_sha256='c'*64,
            response_token_count=25, model_token_count=20)

    def append(self, first, last):
        return append_batch(self.path, self.binding, self.ids, self.sources,
                            [self.row(i) for i in range(first, last)])

    def test_batch_size_change_keeps_exact_occurrences_and_cursor(self):
        self.assertEqual(self.append(0, 64)['cursor'], 64)
        self.assertEqual(self.append(64, 320)['cursor'], 320)
        rows = load_prefix(self.path, self.binding, self.ids, self.sources)
        self.assertEqual([r['ordinal'] for r in rows], list(range(320)))
        self.assertEqual([r['question_id'] for r in rows[:3]], ['same-question']*3)
        with self.assertRaisesRegex(ValueError, 'ordinal'):
            self.append(64, 320)  # An acknowledged batch cannot be replayed.

    def test_crash_before_atomic_publication_is_not_a_durable_batch(self):
        self.append(0, 64)
        (self.path/'.journal-dead.tmp').write_text('{partial')
        self.assertEqual(len(load_prefix(self.path, self.binding, self.ids, self.sources)), 64)

    def test_missing_prefix_and_corrupted_score_are_rejected(self):
        self.append(0, 64); self.append(64, 320)
        first = self.path/'batch-00000000-00000064.json'
        raw = first.read_text(); first.unlink()
        with self.assertRaisesRegex(ValueError, 'Gap/overlap'):
            load_prefix(self.path, self.binding, self.ids, self.sources)
        first.write_text(raw)
        value = json.loads(raw); value['rows'][0]['f1'] = .5
        first.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'scores changed'):
            load_prefix(self.path, self.binding, self.ids, self.sources)

    def test_changed_checkpoint_order_and_source_binding_rejected(self):
        self.append(0, 64)
        other = dict(self.binding, checkpoint_manifest_sha256='d'*64)
        with self.assertRaisesRegex(ValueError, 'different evaluation'):
            load_prefix(self.path, other, self.ids, self.sources)
        ids = list(self.ids); ids[10], ids[11] = ids[11], ids[10]
        with self.assertRaisesRegex(ValueError, 'binding changed'):
            load_prefix(self.path, self.binding, ids, self.sources)
        wrong = self.row(64); wrong['data_source'] = 'other'
        with self.assertRaisesRegex(ValueError, 'question/source'):
            append_batch(self.path, self.binding, self.ids, self.sources, [wrong])

    def test_nonfinite_and_inconsistent_scores_refused_before_persisting(self):
        for changes in ({'f1':float('nan')}, {'em':1,'f1':.5},
                        {'actual_calls':4}, {'model_token_count':26}):
            row = dict(self.row(0), **changes)
            with self.assertRaises(ValueError):
                append_batch(self.path, self.binding, self.ids, self.sources, [row])
        self.assertEqual(load_prefix(self.path, self.binding, self.ids, self.sources), [])

    def test_full_51713_aggregation_counts_actions_and_duplicate_occurrences(self):
        with self.assertRaisesRegex(ValueError, 'complete 51713'):
            aggregate_final(self.binding, self.ids, self.sources, [self.row(0)])
        rows = [self.row(i) for i in range(FINAL_COUNT)]
        proof = aggregate_final(self.binding, self.ids, self.sources, rows)
        self.assertEqual(proof['qa_count'], FINAL_COUNT)
        self.assertEqual(sum(proof['counts'].values()), FINAL_COUNT)
        self.assertEqual(proof['correct'], FINAL_COUNT//2)
        self.assertEqual(proof['valid_action_count'], FINAL_COUNT*2)
        self.assertEqual(proof['action_opportunity_count'], FINAL_COUNT*3)
        self.assertEqual(proof['valid_action_rate'], 2/3)
        self.assertEqual(proof['question_ids'][:3], ['same-question']*3)

    def test_last_partial_batch_and_complete_resume(self):
        # A realistic end boundary crossing 64 -> 256 does not require padding
        # or fake rows for the canonical odd final count.
        self.append(0, 64)
        self.append(64, FINAL_COUNT-1)
        self.append(FINAL_COUNT-1, FINAL_COUNT)
        rows = load_prefix(self.path, self.binding, self.ids, self.sources)
        self.assertEqual(len(rows), FINAL_COUNT)
        self.assertEqual(rows[-1]['ordinal'], FINAL_COUNT-1)
        self.assertEqual(aggregate_final(self.binding, self.ids, self.sources, rows)['qa_count'], FINAL_COUNT)


if __name__ == '__main__':
    unittest.main()
