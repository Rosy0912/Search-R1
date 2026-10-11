import importlib.util
import pathlib
import random
import sys
import types
import unittest

spec = importlib.util.spec_from_file_location(
    'recursive_branch_b', pathlib.Path(__file__).parents[2]/'search_r1/baselines/recursive_branch.py')
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


def record(name='root', generation=0, reward=0, steps=None, parent=None):
    steps = ['a', 'b', 'c'] if steps is None else steps
    return m.Record(name, parent, 'root', generation, reward, None,
                    [types.SimpleNamespace(done=False) for _ in steps], steps, 'initial')


class LastPositionRng(random.Random):
    def choice(self, values):
        return values[-1]


class RecursiveBTests(unittest.TestCase):
    def test_exact_inverse_generation_probabilities(self):
        eligible = [(record(str(i), i), ([0], 0, {})) for i in range(3)]
        w, p = m.parent_probabilities(eligible, 'inverse_generation')
        self.assertEqual(w, [1, .5, 1/3])
        for actual, expected in zip(p, [6/11, 3/11, 2/11]):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(m.parent_probabilities(eligible, 'uniform')[1], [1/3]*3)

    def test_lower_generation_only_and_complete_observations(self):
        parent = record('p', 2, steps=[('a', 'obs-a'), ('b', 'obs-b'), ('c', 'obs-c')])
        lower = record('l', 1, steps=[('a', 'obs-a'), ('b', 'DIFFERENT'), ('c', 'obs-c')])
        same = record('s', 2, steps=parent.steps)
        self.assertEqual(m.legal_positions(parent, [lower, same])[0], [1, 2])
        self.assertEqual(m.legal_positions(record(), [same])[0], [0, 1, 2])

    def test_nonrestorable_and_terminal_positions_excluded(self):
        p = record()
        p.snapshots[0] = None
        p.snapshots[1].done = True
        self.assertEqual(m.legal_positions(p, [])[0], [2])

    def test_budget_mid_backoff_closes_search_and_fills_without_resampling(self):
        events, calls = [], []
        def cont(parent, position, attempt):
            calls.append(position)
            return record(str(attempt), 1, steps=parent.steps[:position]+[str(attempt)],
                          parent=parent.trajectory_id)
        final, n, reason = m.search(record(), cont, LastPositionRng(42),
                                   lambda name, **x: events.append((name, x)), budget=2)
        self.assertEqual((calls, n, reason), ([2, 1], 2, 'budget_exhausted'))
        self.assertEqual(len(final), 3)
        ends = [x for name, x in events if name=='search_end']
        self.assertEqual(len(ends), 1)
        self.assertEqual(ends[0]['reason'], 'budget_exhausted')
        self.assertEqual(ends[0]['attempts'], 2)
        self.assertEqual(len({r.trajectory_id for r in final}), 3)

    def test_error_preserves_existing_pool_and_closes_all_events(self):
        events = []
        planner = m.Planner(record(), LastPositionRng(42),
                            lambda name, **x: events.append((name, x)))
        parent, d, a = planner.next_request()
        planner.accept_result(record('valid', 1, steps=parent.steps[:d]+['x'], parent='root'))
        planner.next_request()
        planner.accept_result(error=RuntimeError('executor failed'))
        self.assertIsNone(planner.next_request())
        final = planner.finish()
        self.assertEqual((planner.attempts, planner.errors, len(final)), (2, 1, 2))
        self.assertEqual(planner.reason, 'execution_error')
        self.assertEqual([n for n, _ in events][-3:], ['search_end', 'fill_end', 'question_end'])
        self.assertEqual(events[-3][1]['reason'], 'execution_error')
        self.assertEqual(final[-1].trajectory_id, 'valid')
        before = len(events)
        self.assertIs(planner.finish(), final)
        self.assertEqual(len(events), before)

    def test_same_reward_pool_never_becomes_search_parent(self):
        events = []
        def cont(parent, d, a):
            self.assertEqual(parent.trajectory_id, 'root')
            return record(str(a), 1, steps=parent.steps[:d]+[str(a)], parent='root')
        final, n, reason = m.search(record(), cont, random.Random(42),
                                   lambda name, **x: events.append((name, x)))
        self.assertEqual((len(final), n, reason), (5, 10, 'budget_exhausted'))
        self.assertEqual(len({r.trajectory_id for r in final}), 5)
        for name, x in events:
            if name=='search_start':
                self.assertEqual([v['id'] for v in x['eligible']], ['root'])
                self.assertEqual(x['eligible'][0]['probability'], 1)

    def test_invalid_child_is_error_not_reward_zero(self):
        p = m.Planner(record(), random.Random(42), lambda *a, **k: None)
        parent, d, a = p.next_request()
        p.accept_result(record('bad', 1, reward=1, steps=parent.steps, parent='WRONG'))
        self.assertIsNone(p.next_request())
        self.assertEqual((p.errors, len(p.pool), len(p.finish())), (1, 0, 1))

    def test_zero_budget_and_single_target_do_not_generate(self):
        def impossible(*a):
            raise AssertionError('Unexpected generation')
        final, n, reason = m.search(record(), impossible, random.Random(1),
                                   lambda *a, **k: None, budget=0)
        self.assertEqual((len(final), n, reason), (1, 0, 'budget_exhausted'))
        final, n, reason = m.search(record(), impossible, random.Random(1),
                                   lambda *a, **k: None, target=1)
        self.assertEqual((len(final), n, reason), (1, 0, 'target_reached'))


if __name__ == '__main__':
    unittest.main()
