"""Small CPU tests for actual ablation semantics; no model/GPU fixtures."""
import random
from types import SimpleNamespace
import unittest

from search_r1.baselines.recursive_ablation import Planner, Options, position_distribution
from search_r1.baselines.recursive_branch import Planner as OriginalPlanner, Record


def root(identity='root'):
    return Record(identity, None, identity, 0, 0, SimpleNamespace(source=0),
                  [0, 1, 2, 3], [('root', n) for n in range(4)], 'initial')


def result(parent, index, attempt, reward):
    return Record('child-%d' % attempt, parent.trajectory_id, parent.root_id,
                  parent.generation + 1, reward, SimpleNamespace(source=0),
                  [0, 1, 2, 3], parent.steps[:index] + [('new', attempt, n)
                  for n in range(index, 4)], 'initial')


def drive(cls=Planner, *, options=None, budget=10, seed=42, rewards=None, identity='root'):
    rng, events, requests = random.Random(seed), [], []
    kwargs = (dict(options=options, question_id='nq:128', outer=50) if cls is Planner else {})
    planner = cls(root(identity), rng, lambda event, **fields: events.append((event, fields)),
                  target=5, budget=budget, **kwargs)
    while True:
        pending = planner.next_request()
        if pending is None:
            break
        parent, index, attempt = pending
        requests.append((parent.trajectory_id, index, attempt, parent.generation))
        reward = (rewards[attempt - 1] if rewards is not None else 1 - parent.reward)
        planner.accept_result(result(parent, index, attempt, reward))
    selected = planner.finish()
    return planner, requests, selected, rng.getstate(), events


class Checks(unittest.TestCase):
    def test_root_only_never_reuses_a_child_even_after_acceptance(self):
        planner, requests, selected, _, _ = drive(options={'variant': 'root_only'})
        self.assertEqual(len(selected), 5)
        self.assertEqual(planner.attempts, 4)
        self.assertTrue(all(identity == 'root' and generation == 0
                            for identity, _, _, generation in requests))
        self.assertEqual([record.generation for record in selected], [0, 1, 1, 1, 1])
        # The scripted contrast children really are eligible parents in A.
        _, original_requests, _, _, _ = drive(OriginalPlanner)
        self.assertTrue(any(generation > 0 for _, _, _, generation in original_requests))

    def test_random_final_has_same_tree_and_consumes_exact_original_rng(self):
        rewards = [0, 0, 1, 1, 0, 1, 0, 1, 0, 1]
        original, requests, original_final, rng_state, _ = drive(OriginalPlanner, rewards=rewards)
        ablation, new_requests, selected, new_rng_state, events = drive(
            options={'variant': 'random_final'}, rewards=rewards)
        self.assertEqual(new_requests, requests)
        self.assertEqual(new_rng_state, rng_state)
        self.assertEqual([(r.trajectory_id, r.parent_id, r.generation, r.reward, r.steps)
                          for r in ablation.accepted + ablation.pool],
                         [(r.trajectory_id, r.parent_id, r.generation, r.reward, r.steps)
                          for r in original.accepted + original.pool])
        self.assertEqual(ablation.original_final_ids, [r.trajectory_id for r in original_final])
        self.assertEqual(selected[0].trajectory_id, 'root')
        self.assertEqual(len({r.trajectory_id for r in selected}), 5)
        self.assertTrue(any(r.trajectory_id not in ablation.original_final_ids for r in selected))
        self.assertTrue(any(r in ablation.pool for r in selected))
        self.assertEqual(sum(event == 'question_end' for event, _ in events), 1)
        self.assertEqual(sum(event == 'original_selection_end' for event, _ in events), 1)
        self.assertEqual(ablation.finish(), selected)
        self.assertEqual(ablation.rng.getstate(), rng_state)

    def test_random_final_independent_of_nondeterministic_root_uuid(self):
        rewards = [0] * 10
        first = drive(options={'variant': 'random_final'}, rewards=rewards, identity='uuid-one')
        second = drive(options={'variant': 'random_final'}, rewards=rewards, identity='uuid-two')
        self.assertEqual([r.trajectory_id for r in first[2][1:]],
                         [r.trajectory_id for r in second[2][1:]])

    def test_short_group_does_not_fabricate_or_add_attempts(self):
        planner, requests, selected, _, events = drive(
            options={'variant': 'random_final'}, budget=2, rewards=[0, 0])
        self.assertEqual(planner.attempts, 2)
        self.assertEqual(len(requests), 2)
        self.assertEqual(len(selected), 3)
        end = [fields for event, fields in events if event == 'question_end'][0]
        self.assertFalse(end['complete_group'])
        self.assertEqual(end['stop_reason'], 'budget_exhausted')

    def test_depth_one_is_same_root_only_control(self):
        depth = drive(options={'max_generation': 1})
        c1 = drive(options={'variant': 'root_only'})
        self.assertEqual(depth[1], c1[1])
        self.assertEqual([r.trajectory_id for r in depth[2]], [r.trajectory_id for r in c1[2]])
        self.assertEqual(depth[3], c1[3])

    def test_depth_limit_is_genealogy_not_attempt_number(self):
        planner, requests, selected, _, _ = drive(options={'max_generation': 2}, seed=7)
        self.assertEqual(planner.attempts, 4)
        self.assertTrue(all(generation < 2 for _, _, _, generation in requests))
        self.assertTrue(all(r.generation <= 2 for r in selected))
        self.assertTrue(any(r.generation == 2 for r in selected))

    def test_biased_position_keeps_lcp_boundary_and_original_backoff(self):
        class LastChoice:
            def choice(self, values):
                return values[-1]
            def choices(self, values, **kwargs):
                return [values[-1]]
            def sample(self, values, count):
                return values[:count]
        initial = root()
        parent = result(initial, 2, 1, 1)
        events = []
        planner = Planner(initial, LastChoice(), lambda event, **fields: events.append((event, fields)),
                          options={'position': 'late', 'max_generation': 2})
        planner.accepted.append(parent)
        first = planner.next_request()
        self.assertEqual((first[0].trajectory_id, first[1], planner.boundary), ('child-1', 3, 2))
        planner.accept_result(result(parent, 3, 2, 1))
        second = planner.next_request()
        self.assertEqual((second[0].trajectory_id, second[1]), ('child-1', 2))
        start = [fields for event, fields in events if event == 'search_start'][0]
        self.assertEqual(start['legal_positions'], [2, 3])
        self.assertEqual(start['position_probabilities'], [1 / 3, 2 / 3])

    def test_probabilities_frozen_and_configuration_rejects_typos(self):
        self.assertEqual(position_distribution([2, 3, 4], 'early')[0], [3, 2, 1])
        self.assertEqual(position_distribution([2, 3, 4], 'late')[0], [1, 2, 3])
        with self.assertRaises(TypeError):
            Options.parse({'max_generations': 1})
        with self.assertRaises(ValueError):
            Options(max_generation=True)
        with self.assertRaisesRegex(ValueError, 'question ID'):
            Planner(root(), random.Random(42), lambda *args, **kwargs: None,
                    options={'variant': 'random_final'})


if __name__ == '__main__':
    unittest.main()
