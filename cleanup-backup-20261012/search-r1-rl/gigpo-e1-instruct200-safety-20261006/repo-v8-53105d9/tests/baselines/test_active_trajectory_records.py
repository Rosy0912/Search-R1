"""Exercise shrinking batches through the production rollout and environment parser."""
import gzip
import json
from pathlib import Path
import tempfile
import unittest
import torch
from tests.baselines.test_algorithms import manager, batch, padded
from verl import DataProto
from search_r1.baselines.rollout import SearchRollout
from search_r1.baselines.trajectory_records import RolloutTrace

class ActiveTraceTests(unittest.TestCase):
    def test_shrinking_rollout_keeps_full_rows_without_changing_model_inputs(self):
        def run(root):
            obj, worker, queries = manager(gpus=2)
            def generate(data):
                worker.calls.append(data.batch['input_ids'].clone())
                texts = ['<answer>Paris</answer>' if (len(worker.calls) == 1 and i == 0)
                         or '<information>' in obj.tokenizer.decode(row)
                         else '<search>capital</search>'
                         for i, row in enumerate(data.batch['input_ids'])]
                return DataProto.from_dict({'responses': padded([obj.tokenizer.encode(t) for t in texts], 0)})
            worker.generate_sequences = generate
            if root is not None:
                obj.rollout_trace = RolloutTrace(root, 26, ['q'] * 5, 5)
            output, sources = SearchRollout(obj).run(batch(obj.tokenizer, 5))
            return output, sources, worker.calls, queries
        with tempfile.TemporaryDirectory() as root:
            rng = torch.get_rng_state().clone()
            plain, logged = run(None), run(root)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertEqual(plain[3], logged[3])
            self.assertEqual(plain[1].tolist(), logged[1].tolist())
            self.assertEqual(len(plain[2]), len(logged[2]))
            for a, b in zip(plain[2], logged[2]):
                torch.testing.assert_close(a, b)
            for key in plain[0].batch.keys():
                torch.testing.assert_close(plain[0].batch[key], logged[0].batch[key])
            files = sorted((Path(root) / 'iter_000026').glob('turn_*.gz'))
            self.assertEqual(len(files), 2)
            with gzip.open(files[1], 'rt') as stream:
                rows = [json.loads(line) for line in stream]
            self.assertEqual([r['trajectory_row'] for r in rows], list(range(5)))
            self.assertEqual([r['active'] for r in rows], [False, True, True, True, True])
            self.assertEqual(rows[0]['model_action'], '')
            self.assertTrue(all(r['done'] for r in rows))

    def test_reordered_subset_and_invalid_identity_fail_closed(self):
        with tempfile.TemporaryDirectory() as root:
            trace = RolloutTrace(root, 26, ['q'] * 5, 5)
            trace.record_active_turn([4, 1], ['four', 'one'], ['raw4', 'raw1'], [0, 1], [1, 1], [1, 0])
            with gzip.open(trace.root / 'turn_01.jsonl.gz', 'rt') as stream:
                rows = [json.loads(line) for line in stream]
            self.assertEqual(rows[4]['raw_tool_observation_before_token_limit'], 'raw4')
            self.assertEqual(rows[1]['model_action'], 'one')
            for indices in ([1, 1], [-1], [5]):
                with self.assertRaises(ValueError):
                    n = len(indices)
                    trace.record_active_turn(indices, ['x'] * n, ['y'] * n, [0] * n, [1] * n, [1] * n)
            with self.assertRaises(ValueError):
                trace.record_active_turn([1], [], ['y'], [0], [1], [1])
            self.assertEqual(trace.turn, 1)
