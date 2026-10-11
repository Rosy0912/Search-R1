"""Actual recursive executor with scripted CPU actions; no model/speed claim."""
import copy
import os
import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from search_r1.baselines.rollout import SearchRollout, State
from search_r1.baselines.recursive_branch import Planner


def append_action(state, token):
    state.response.append(token)
    state.model_mask.append(1)
    state.steps.append(len(state.anchors))
    state.anchors.append(str(token))
    state.records.append(dict(text=str(token), action_tokens=1, observation_tokens=0))
    state.raw_ledger.append(dict(action=dict(action_token_ids=[token],
        native_emission=dict(true_emitted_length=1)), observation_token_ids=[]))
    state.rolling.append(token)
    state.turns += 1


class Scripted(SearchRollout):
    def __init__(self, durations, fail_at=None):
        self.config = SimpleNamespace(max_turns=4, force_final_answer=False)
        self.rng = random.Random(42)
        self.record_details = False
        self.question_uids = list(range(len(durations)))
        self.rollout_outer = 6
        self.cost = dict(trajectories=0)
        self.durations = durations
        self.calls = []
        self.fail_at = fail_at
        self.reward_callback = lambda states: [0] * len(states)

    def advance(self, states, meta):
        self.calls.append([(s.source, s.trajectory, s.turns) for s in states])
        if len(self.calls) == self.fail_at:
            raise RuntimeError('scripted backend failure')
        self.assert_unique = len({s.source for s in states}) == len(states)
        if not self.assert_unique:
            raise AssertionError('More than one pending branch for a question')
        for state in states:
            append_action(state, 100 + state.source * 10 + state.turns)
            state.done = state.turns >= self.durations[state.source]

    def execute(self, target=2, budget=2):
        roots, snapshots = [], []
        for source in range(len(self.durations)):
            before = State(source, [source], [source])
            root = copy.deepcopy(before)
            append_action(root, 90 + source)
            root.done = True
            roots.append(root)
            snapshots.append([before])
        tree = SimpleNamespace(roots=1, leaves=target, continuation_budget_per_question=budget)
        with patch.dict(os.environ, SEARCH_R1_BASELINE_OUTPUT='/unused-cpu-fixture'):
            return self._recursive_branches(roots, snapshots, tree, {})


class Checks(unittest.TestCase):
    def test_short_question_starts_next_branch_before_long_one_finishes(self):
        runner = Scripted([1, 3])
        final = runner.execute()
        # First two calls both questions are active, but short question already
        # starts a NEW branch while long question continues its first branch.
        first, second = runner.calls[:2]
        self.assertEqual([x[0] for x in first], [0, 1])
        self.assertEqual([x[0] for x in second], [0, 1])
        self.assertNotEqual(first[0][1], second[0][1])
        self.assertEqual(first[1][1], second[1][1])
        self.assertEqual(second[1][2], 1)
        self.assertEqual([s.source for s in final], [0, 0, 1, 1])
        self.assertEqual(runner.tree_candidate_counts, [2, 2])
        self.assertEqual(runner.cost['trajectories'], 4)

    def test_all_attempts_counted_and_groups_stay_question_major(self):
        runner = Scripted([1, 3, 2])
        final = runner.execute(target=5, budget=10)
        self.assertEqual(runner.tree_candidate_counts, [10, 10, 10])
        self.assertEqual([s.source for s in final], [0]*5 + [1]*5 + [2]*5)
        self.assertEqual(runner.cost['trajectories'], 30)
        self.assertTrue(all(x['selected'] == 5 and x['reason'] == 'budget_exhausted'
                            for x in runner.recursive_summaries))
        # No later call revives a completed trajectory ID.
        seen, terminal = {}, set()
        for call in runner.calls:
            self.assertEqual([x[0] for x in call], sorted(x[0] for x in call))
            for source, identity, turn in call:
                self.assertNotIn(identity, terminal)
                if identity in seen:
                    self.assertEqual(turn, seen[identity] + 1)
                seen[identity] = turn
                if turn + 1 == runner.durations[source]:
                    terminal.add(identity)

    def test_backend_error_refuses_outer_and_closes_every_planner(self):
        instances = []
        class TrackingPlanner(Planner):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                instances.append(self)
        runner = Scripted([1, 3], fail_at=2)
        with patch('search_r1.baselines.recursive_branch.Planner', TrackingPlanner):
            with self.assertRaisesRegex(RuntimeError, 'scripted backend failure'):
                runner.execute()
        self.assertEqual(len(instances), 2)
        self.assertTrue(all(p.pending is None and p.reason == 'execution_error' for p in instances))
        self.assertFalse(hasattr(runner, 'recursive_summaries'))


if __name__ == '__main__':
    unittest.main()
