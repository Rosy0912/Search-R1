"""Root-anchored recursion invariants; CPU only, with a real executor smoke test."""
import copy
import os
from pathlib import Path
import random
import subprocess
import sys
import textwrap
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from search_r1.baselines.recursive_ablation import Options, Planner, root_prefix_diagnostics
from search_r1.baselines.recursive_branch import (
    Planner as OriginalPlanner, Record, legal_positions,
)


VARIANT = {'variant': 'root_anchored_recursive'}


def root(reward=0):
    steps = [((n,), (100 + n,)) for n in range(4)]
    snapshots = [dict(turns=n, ledger=copy.deepcopy(steps[:n])) for n in range(4)]
    # A tree ID need not equal its original trajectory ID in the real executor.
    return Record('initial', None, 'tree-id', 0, reward, SimpleNamespace(source=0),
                  snapshots, steps, 'initial')


def child(parent, index, attempt, reward):
    steps = copy.deepcopy(parent.steps[:index]) + [
        ((attempt * 10 + n,), (attempt * 100 + n,)) for n in range(index, 4)]
    snapshots = copy.deepcopy(parent.snapshots[:index]) + [
        dict(turns=n, ledger=copy.deepcopy(steps[:n])) for n in range(index, 4)]
    return Record('child-%d' % attempt, parent.trajectory_id, parent.root_id,
                  parent.generation + 1, reward, SimpleNamespace(source=0),
                  snapshots, steps, 'initial', {'model_calls': 4 - index})


class ForcedParentRandom:
    """Force a genealogy while preserving an explicit matched random stream."""
    def __init__(self, last_parent=True, position='last', seed=42):
        self.last_parent = last_parent
        self.position = position
        self.rng = random.Random(seed)
        self.calls = []

    def choice(self, values):
        if isinstance(values[0], tuple):
            self.calls.append(('parent', self.rng.random()))
            return values[-1] if self.last_parent else values[0]
        self.calls.append(('position', tuple(values)))
        if self.position == 'random':
            return self.rng.choice(values)
        return values[-1] if self.position == 'last' else values[0]

    def sample(self, values, count):
        self.calls.append(('fill', tuple(r.trajectory_id for r in values), count))
        return self.rng.sample(values, count)


def drive(*, initial=None, rng=None, options=None, rewards=None, budget=10,
          target=5, planner_class=Planner):
    initial = initial if initial is not None else root()
    rng = rng if rng is not None else random.Random(42)
    events, requests = [], []
    kwargs = dict(options=options, question_id='q:1', outer=50) if planner_class is Planner else {}
    planner = planner_class(initial, rng, lambda event, **fields: events.append((event, fields)),
                            target=target, budget=budget, **kwargs)
    while (request := planner.next_request()) is not None:
        parent, index, attempt = request
        requests.append((parent.trajectory_id, parent.generation, index, attempt))
        reward = rewards[attempt - 1] if rewards is not None else 1 - initial.reward
        planner.accept_result(child(parent, index, attempt, reward))
    return planner, requests, planner.finish(), events


class RootAnchoredTests(unittest.TestCase):
    def test_both_root_rewards_accept_same_parent_reward_and_recurse(self):
        for reward in (0, 1):
            with self.subTest(root_reward=reward):
                initial = root(reward)
                planner, requests, selected, events = drive(
                    initial=initial, rng=ForcedParentRandom(), options=VARIANT)
                self.assertIs(selected[0], initial)
                self.assertEqual([r.generation for r in selected], [0, 1, 2, 3, 4])
                self.assertEqual([r.reward for r in selected], [reward] + [1 - reward] * 4)
                self.assertEqual((planner.attempts, planner.reason), (4, 'target_reached'))
                self.assertTrue(any(generation >= 1 for _, generation, _, _ in requests))
                branches = [f for e, f in events if e == 'branch_end']
                self.assertEqual(branches[1]['parent_reward'], 1 - reward)
                self.assertEqual(branches[1]['reward'], 1 - reward)
                self.assertTrue(all(f['source'] == 'contrast_accept' for f in branches))
                for _, fields in events:
                    self.assertEqual(fields['acceptance_reference'], 'initial_root')
                    self.assertEqual(fields['reference_reward'], reward)

    def test_descendant_can_revisit_before_old_lcp_boundary(self):
        initial = root()
        descendant = child(initial, 2, 99, 1)
        self.assertEqual(legal_positions(descendant, [initial, descendant])[0], [2, 3])
        planner = Planner(initial, ForcedParentRandom(position='first'), lambda *a, **kw: None,
                          options=VARIANT)
        planner.accepted.append(descendant)
        parent, index, attempt = planner.next_request()
        self.assertIs(parent, descendant)
        self.assertEqual((index, planner.boundary), (0, 0))
        self.assertIs(parent.snapshots[index], descendant.snapshots[0])
        planner.accept_result(child(parent, index, attempt, 1))
        self.assertEqual(planner.accepted[-1].generation, 2)

    def test_root_reward_candidate_backoffs_same_parent_to_zero(self):
        initial = root()
        descendant = child(initial, 2, 99, 1)
        rng = ForcedParentRandom()
        planner = Planner(initial, rng, lambda *a, **kw: None, options=VARIANT, budget=4)
        planner.accepted.append(descendant)
        positions = []
        while (request := planner.next_request()) is not None:
            parent, index, attempt = request
            self.assertIs(parent, descendant)
            positions.append(index)
            planner.accept_result(child(parent, index, attempt, initial.reward))
        self.assertEqual(positions, [3, 2, 1, 0])
        self.assertEqual(len(planner.accepted), 2)
        self.assertEqual([r.reward for r in planner.pool], [0] * 4)
        self.assertEqual(sum(call[0] == 'parent' for call in rng.calls), 1)
        self.assertEqual(planner.reason, 'budget_exhausted')
        selected = planner.finish()
        self.assertEqual(len(selected), 5)
        self.assertEqual(sum(r.source == 'failure_fill' for r in selected), 3)
        self.assertEqual(len({r.trajectory_id for r in selected}), 5)

    def test_budget_short_group_and_zero_budget_are_truthful(self):
        for budget in (0, 2, 10):
            with self.subTest(budget=budget):
                planner, requests, selected, events = drive(
                    options=VARIANT, budget=budget, rewards=[0] * budget)
                self.assertEqual(planner.attempts, budget)
                self.assertEqual(len(requests), budget)
                self.assertEqual(len(selected), min(5, 1 + budget))
                self.assertEqual(len({r.trajectory_id for r in selected}), len(selected))
                self.assertEqual(events[-1][1]['complete_group'], budget >= 4)
                self.assertEqual(events[-1][1]['stop_reason'], 'budget_exhausted')

    def test_execution_error_consumes_attempt_but_never_fabricates_record(self):
        events = []
        planner = Planner(root(), ForcedParentRandom(),
                          lambda event, **fields: events.append((event, fields)), options=VARIANT)
        parent, index, attempt = planner.next_request()
        planner.accept_result(child(parent, index, attempt, 0))
        planner.next_request()
        planner.accept_result(error=RuntimeError('backend failed'))
        self.assertIsNone(planner.next_request())
        selected = planner.finish()
        self.assertEqual((planner.attempts, len(selected)), (2, 2))
        self.assertEqual(planner.reason, 'execution_error')
        self.assertEqual(selected[1].source, 'failure_fill')
        self.assertEqual(sum(e == 'search_end' for e, _ in events), 1)
        self.assertEqual([e for e, _ in events][-3:], ['search_end', 'fill_end', 'question_end'])

    def test_root_validation_and_empty_real_snapshots(self):
        for field, value in [('generation', 1), ('parent_id', 'other'), ('reward', 2)]:
            initial = root()
            setattr(initial, field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                Planner(initial, random.Random(42), lambda *a, **kw: None, options=VARIANT)
        initial = root()
        initial.snapshots = []
        planner, requests, selected, _ = drive(initial=initial, options=VARIANT)
        self.assertEqual((requests, planner.reason), ([], 'no_expandable_record'))
        self.assertEqual(selected, [initial])
        for kwargs in ({'budget': -1}, {'target': 0}):
            with self.assertRaises(ValueError):
                Planner(root(), random.Random(42), lambda *a, **kw: None, options=VARIANT, **kwargs)
        for extra in ({'position': 'late'}, {'max_generation': 2}):
            with self.assertRaises(ValueError):
                Options(**VARIANT, **extra)

    def test_invalid_identity_reward_or_actual_prefix_is_rejected(self):
        mutations = {
            'parent': lambda c: setattr(c, 'parent_id', 'wrong'),
            'root': lambda c: setattr(c, 'root_id', 'wrong'),
            'generation': lambda c: setattr(c, 'generation', 0),
            'reward': lambda c: setattr(c, 'reward', 0.5),
            'duplicate': lambda c: setattr(c, 'trajectory_id', 'initial'),
            'observation_prefix': lambda c: c.steps.__setitem__(0, ((0,), (999,))),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                planner = Planner(root(), ForcedParentRandom(), lambda *a, **kw: None,
                                  options=VARIANT)
                parent, index, attempt = planner.next_request()
                result = child(parent, index, attempt, 1)
                mutate(result)
                with self.assertRaisesRegex(ValueError, 'identity, generation or prefix'):
                    planner.accept_result(result)
                self.assertIsNotNone(planner.pending)
                planner.accept_result(error=ValueError('rejected continuation'))
                self.assertIsNone(planner.next_request())
                self.assertEqual(len(planner.finish()), 1)

    def test_child_prefix_and_snapshots_are_independent_of_parent(self):
        initial = root()
        saved = copy.deepcopy(initial)
        planner = Planner(initial, ForcedParentRandom(), lambda *a, **kw: None, options=VARIANT)
        parent, index, attempt = planner.next_request()
        result = child(parent, index, attempt, 1)
        planner.accept_result(result)
        result.snapshots[0]['ledger'].append('mutation after completion')
        result.steps[0] = ((999,), (999,))
        self.assertEqual(initial.steps, saved.steps)
        self.assertEqual(initial.snapshots, saved.snapshots)

    def test_effective_recursive_prefix_counts_inherited_actions_not_genealogy(self):
        initial = root()
        descendant = child(initial, 2, 99, 1)
        for index in (0, 2):
            fields = root_prefix_diagnostics(initial, descendant, index)
            self.assertTrue(fields['genealogical_recursive'])
            self.assertFalse(fields['effective_recursive_prefix'])
            self.assertEqual(fields['inherited_novel_action_count'], 0)
            self.assertEqual(fields['parent_prefix_sha256'], fields['root_prefix_sha256'])
        fields = root_prefix_diagnostics(initial, descendant, 3)
        self.assertTrue(fields['effective_recursive_prefix'])
        self.assertEqual(fields['inherited_novel_action_count'], 1)
        self.assertEqual(fields['inherited_novel_action_positions'], [2])
        self.assertEqual(fields['inherited_semantic_difference_positions'], [2])
        self.assertNotEqual(fields['parent_prefix_sha256'], fields['root_prefix_sha256'])
        self.assertEqual(fields['inherited_action_count'], 3)

    def test_observation_only_change_is_not_counted_as_novel_action(self):
        initial = root()
        descendant = copy.deepcopy(initial)
        descendant.trajectory_id, descendant.parent_id, descendant.generation = 'descendant', 'initial', 1
        descendant.steps[1] = (initial.steps[1][0], (999,))
        fields = root_prefix_diagnostics(initial, descendant, 3)
        self.assertTrue(fields['observation_only_recursive_prefix'])
        self.assertFalse(fields['effective_recursive_prefix'])
        self.assertEqual(fields['inherited_novel_action_count'], 0)
        self.assertEqual(fields['inherited_observation_only_difference_positions'], [1])
        self.assertEqual(fields['inherited_semantic_difference_positions'], [1])
        # A new action beyond a shorter original trajectory also counts.
        initial.steps = initial.steps[:2]
        fields = root_prefix_diagnostics(initial, descendant, 3)
        self.assertTrue(fields['effective_recursive_prefix'])
        self.assertFalse(fields['observation_only_recursive_prefix'])
        self.assertEqual(fields['inherited_novel_action_positions'], [2])
        self.assertEqual(fields['inherited_semantic_difference_positions'], [1, 2])

    def test_passive_diagnostics_do_not_change_requests_or_rng(self):
        full_rng, disabled_rng = random.Random(29), random.Random(29)
        rewards = [0, 1, 0, 1, 1, 0, 0, 1, 0, 1]
        full = drive(rng=full_rng, options=VARIANT, rewards=rewards)
        with patch('search_r1.baselines.recursive_ablation.root_prefix_diagnostics', return_value={}):
            disabled = drive(rng=disabled_rng, options=VARIANT, rewards=rewards)
        self.assertEqual(full[1:3], disabled[1:3])
        self.assertEqual(full_rng.getstate(), disabled_rng.getstate())
        starts = [fields for event, fields in full[3] if event == 'branch_start']
        self.assertTrue(starts)
        self.assertTrue(all('effective_recursive_prefix' in fields for fields in starts))

    def test_forced_root_parent_is_c1_with_same_requests_and_random_calls(self):
        for reward in (0, 1):
            first_rng = ForcedParentRandom(last_parent=False, position='random')
            second_rng = ForcedParentRandom(last_parent=False, position='random')
            rewards = [reward, 1 - reward, reward, reward, 1 - reward,
                       reward, reward, 1 - reward, reward, reward]
            c1 = drive(initial=root(reward), rng=first_rng,
                       options={'variant': 'root_only'}, rewards=rewards)
            new = drive(initial=root(reward), rng=second_rng, options=VARIANT, rewards=rewards)
            self.assertEqual(c1[1], new[1])
            self.assertEqual(c1[2], new[2])
            self.assertEqual(first_rng.calls, second_rng.calls)
            self.assertEqual(first_rng.rng.getstate(), second_rng.rng.getstate())

    def test_baseline_still_matches_original_requests_events_and_rng_exactly(self):
        old_rng, new_rng = random.Random(17), random.Random(17)
        rewards = [0, 0, 1, 1, 0, 1, 0, 1, 0, 1]
        original = drive(planner_class=OriginalPlanner, rng=old_rng, rewards=rewards)
        baseline = drive(options={'variant': 'baseline'}, rng=new_rng, rewards=rewards)
        self.assertEqual(original[1:], baseline[1:])
        self.assertEqual(old_rng.getstate(), new_rng.getstate())

    def test_install_uses_real_executor_snapshots_preserves_prefix_and_context(self):
        # Isolate process-local installation. Only missing tensor packages are
        # stubbed: these CPU methods never call numpy or torch operations.
        script = r'''
import copy, importlib.util, os, random, sys, types
from types import SimpleNamespace
for dependency in ('numpy', 'torch'):
    if importlib.util.find_spec(dependency) is None:
        sys.modules[dependency] = types.ModuleType(dependency)
from search_r1.baselines import recursive_ablation, recursive_branch
from search_r1.baselines.rollout import SearchRollout, State

def append(state, token):
    state.response.append(token)
    state.model_mask.append(1)
    state.raw_ledger.append(dict(action=dict(action_token_ids=[token],
        native_emission=dict(true_emitted_length=1)), observation_token_ids=[]))
    state.rolling.append(token)
    state.turns += 1

class ChooseLast:
    def choice(self, values): return values[-1]
    def sample(self, values, count): return values[:count]

class Runner(SearchRollout):
    def __init__(self):
        self.config = SimpleNamespace(max_turns=4, force_final_answer=False)
        self.rng = ChooseLast()
        self.record_details = False
        self.question_uids = ['real-q']
        self.rollout_outer = 50
        self.cost = dict(trajectories=0)
        self.reward_callback = lambda states: [int(s.response[-1] == 99) for s in states]
        self.calls = []
    def advance(self, states, meta):
        for state in states:
            self.calls.append((state.turns, list(state.response)))
            assert state.turns == 3
            assert state.response == [10, 20, 30]
            append(state, 99)
            state.done = True
    def timing(self, *args, **kwargs): pass

initial = State(0, [1], [1])
snapshots = []
for token in (10, 20, 30, 40):
    snapshots.append(copy.deepcopy(initial))
    append(initial, token)
initial.done = True
saved_initial, saved_snapshots = copy.deepcopy(initial), copy.deepcopy(snapshots)
os.environ['SEARCH_R1_BASELINE_OUTPUT'] = '/unused-r1-cpu-fixture'
options = {'variant': 'root_anchored_recursive'}
recursive_ablation.install(options)
recursive_ablation.install(options)
runner = Runner()
tree = SimpleNamespace(roots=1, leaves=5, continuation_budget_per_question=10)
selected = runner._recursive_branches([initial], [snapshots], tree, {})
assert selected[0] is initial
assert [s.branch_metadata['generation'] for s in selected] == [0, 1, 2, 3, 4]
assert [s.branch_metadata['sampled_reward'] for s in selected] == [0, 1, 1, 1, 1]
assert [s.turns for s in selected] == [4] * 5
assert len({s.trajectory for s in selected}) == 5
assert runner.cost['trajectories'] == 4
assert len(runner.calls) == 4
assert initial.response == saved_initial.response and snapshots == saved_snapshots
selected[1].response[0] = -1
assert initial.response[0] == 10 and selected[2].response[0] == 10
assert recursive_ablation._rollout_context.get() is None
try:
    recursive_ablation.install({'variant': 'root_only'})
except RuntimeError:
    pass
else:
    raise AssertionError('Live configuration was silently changed')
try:
    recursive_branch.Planner(None, random.Random(1), lambda *a, **k: None)
except RuntimeError as exc:
    assert 'outside identified rollout context' in str(exc)
else:
    raise AssertionError('Configured planner accepted missing rollout context')
print('installed real executor: generation 4, snapshots/prefix/turns preserved')
'''
        result = subprocess.run([sys.executable, '-c', textwrap.dedent(script)],
                                cwd=Path(__file__).resolve().parents[2],
                                env=os.environ.copy(), text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('installed real executor', result.stdout)


if __name__ == '__main__':
    unittest.main()
