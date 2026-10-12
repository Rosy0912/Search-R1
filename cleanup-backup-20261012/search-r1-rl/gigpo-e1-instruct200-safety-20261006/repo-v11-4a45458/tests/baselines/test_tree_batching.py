"""Cross-root continuation batching preserves the real search provenance/budget."""
from types import SimpleNamespace
import unittest
import torch

from tests.baselines.test_algorithms import manager, batch
from search_r1.baselines.rollout import SearchRollout


class TreeBatching(unittest.TestCase):
    def test_cross_root_children_share_one_generation_batch(self):
        obj, worker, queries = manager(gpus=8)
        engine = SearchRollout(obj)
        # Resume from the post-search state. Reusing that observation must not
        # replay its retrieval, even when several roots continue in one call.
        engine.rng.choice = lambda nodes: nodes[-1]
        output, sources = engine.run(batch(obj.tokenizer, 2),
                                     SimpleNamespace(rounds=1, expansions=2, leaves=3))
        self.assertEqual(len(worker.calls), 3)  # two initial turns, one expansion
        self.assertEqual([len(call) for call in worker.calls], [8, 8, 8])
        self.assertEqual(engine.cost['padded_rows'], 16)  # formerly 24
        self.assertEqual(engine.cost['trajectories'], 6)
        self.assertEqual(engine.cost['retrieval_calls'], 2)
        self.assertEqual(len(queries), 2)
        self.assertEqual(sources.tolist(), [0, 0, 0, 1, 1, 1])
        self.assertEqual(len(set(output.non_tensor_batch['tree_uid'][:3])), 1)
        self.assertEqual(len(set(output.non_tensor_batch['tree_uid'][3:])), 1)
        self.assertNotEqual(output.non_tensor_batch['tree_uid'][0], output.non_tensor_batch['tree_uid'][3])
        self.assertEqual(len(set(output.non_tensor_batch['trajectory_uid'])), 6)
        for state in engine.last_states:
            self.assertEqual(state.turns, 2)
            self.assertEqual(state.valid_searches, 1)
            self.assertEqual(state.records[0]['text'], '<search>capital</search>')
            self.assertEqual(state.records[1]['text'], '<answer>Paris</answer>')
        width = output.batch['responses'].shape[1]
        action_mask = output.batch['info_mask'][:, -width:].bool()
        self.assertTrue(torch.all(output.batch['step_ids'][~action_mask] == -1))
        self.assertEqual(engine.cost['generated_tokens'],
                         8 * (len(worker.tokenizer.encode('<search>capital</search>'))
                              + 2 * len(worker.tokenizer.encode('<answer>Paris</answer>'))))

    def test_multi_round_deduplication_and_remaining_turn_budget(self):
        obj, worker, queries = manager(gpus=8)
        engine = SearchRollout(obj)
        engine.rng.choice = lambda nodes: nodes[-1]
        output, sources = engine.run(batch(obj.tokenizer, 3),
                                     SimpleNamespace(rounds=2, expansions=2, leaves=3))
        self.assertEqual(len(worker.calls), 4)
        self.assertEqual(engine.tree_candidate_counts, [2, 2, 2, 2, 2, 2])
        self.assertEqual(engine.cost['trajectories'], 15)
        self.assertEqual(engine.cost['padded_rows'], 14)  # 5+5 initial, 2+2 expansion
        self.assertEqual(len(queries), 3)
        self.assertEqual(sources.tolist(), [0, 0, 0, 1, 1, 1, 2, 2, 2])
        self.assertTrue(all(s.turns == 2 and len(s.anchors) == 2 for s in engine.last_states))
        self.assertEqual(len(output), 9)


if __name__ == '__main__':
    unittest.main()
