import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from release.recursive_bran_a.ablation150_v1.test_runner import minimal_config
from search_r1.baselines.recursive_ablation import Planner, Options
from search_r1.baselines.recursive_branch import Record
from .prepare_ablation import condition_for
from .r1_evidence import GROUP, load_question, validate_question, verify_events


class R1Evidence(unittest.TestCase):
    def record(self, base):
        condition = condition_for(GROUP)
        config = base / 'resolved.json'
        config.write_text(json.dumps(minimal_config(condition)))
        return dict(group=GROUP, condition=condition, config=str(config),
                    config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
                    run_id='run', source_commit='source')

    def fixture(self, mode='contrast', root_reward=0, directory=None, record=None, question='nq:128'):
        events, files = [], {}
        identity = dict(question_id=question, outer=5, run_id='run', source_commit='source',
                        config_sha256=record['config_sha256'] if record else 'config')

        def state(key, steps):
            return dict(trajectory=key, tree='tree', source=0,
                raw_ledger=[dict(action=dict(action_token_ids=list(a)), observation_token_ids=list(o))
                            for a, o in steps])

        def snapshots(key, steps, begin=0):
            return [state(key, steps[:i]) for i in range(begin, len(steps))]

        def emit(event, **fields):
            if event == 'branch_start' and directory is not None:
                snapshot = copy.deepcopy(planner.parent.snapshots[fields['action_index']])
                path = directory / ('snapshot-%d.json' % fields['attempt'])
                path.write_text(json.dumps(snapshot))
                fields.update(snapshot=str(path), snapshot_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            events.append(dict(event=event, event_seq=len(events) + 1, **identity, **fields))

        class Choices:
            positions = 0

            def choice(self, values):
                if isinstance(values[0], tuple):
                    return values[0] if mode == 'shallow' else values[-1]
                self.positions += 1
                if mode == 'early_ancestor' and self.positions == 2:
                    return values[0]
                return values[min(1, len(values) - 1)] if self.positions == 1 else values[-1]

            def sample(self, values, count):
                return values[:count]

        steps = [] if mode == 'empty' else [((100 + i,), (200 + i,)) for i in range(4)]
        initial = state('root', steps)
        files['root'] = dict(state=initial, source='initial')
        root = Record('root', None, 'tree', 0, root_reward, initial, snapshots('root', steps), steps, 'initial')
        planner = Planner(root, Choices(), emit, options=Options(variant='root_anchored_recursive'))
        while True:
            request = planner.next_request()
            if request is None:
                break
            parent, index, attempt = request
            suffix = [((100 + i if mode == 'observation' else 1000 * attempt + i,),
                       (2000 * attempt + i,)) for i in range(index, 4)]
            steps = parent.steps[:index] + suffix
            reward = root_reward if mode == 'same' or (mode == 'mixed' and attempt % 2) else 1 - root_reward
            key = 'child-%d' % attempt
            value = dict(state=state(key, steps), cost=dict(new_generated_tokens=4 - index))
            files[key] = value
            planner.accept_result(Record(key, parent.trajectory_id, 'tree', parent.generation + 1,
                reward, value['state'], parent.snapshots[:index] + snapshots(key, steps, index),
                steps, 'initial', value['cost']))
        planner.finish()
        if directory is not None:
            for key, value in files.items():
                (directory / (key + '.json')).write_text(json.dumps(value))
            (directory / 'events.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in events))
        return events, files

    def test_binary_roots_same_parent_acceptance_and_real_recursive_prefix(self):
        for reward in (0, 1):
            events, files = self.fixture(root_reward=reward)
            result = validate_question(events, files)
            self.assertEqual(result['accepted'], 4)
            self.assertEqual(result['accepted_same_reward_as_parent'], 3)
            self.assertEqual(result['max_generation'], 4)
            self.assertEqual(result['effective_recursive_branches'], 3)

    def test_reports_zero_effective_recursion_without_forcing_it(self):
        for mode in ('same', 'shallow', 'empty', 'observation'):
            result = validate_question(*self.fixture(mode))
            self.assertEqual(result['effective_recursive_branches'], 0)
            if mode == 'observation':
                self.assertEqual(result['genealogical_recursive_branches'], 3)
                self.assertEqual(result['observation_only_recursive_branches'], 3)
            if mode == 'same':
                self.assertEqual(result['branches'], 10)
                self.assertEqual(result['filled'], 4)
        validate_question(*self.fixture('mixed'))

    def test_rejects_parent_reference_budget_prefix_and_diagnostic_corruption(self):
        for mutation in ('reference', 'budget', 'parent_accept', 'prefix', 'novelty', 'generation', 'fill'):
            events, files = self.fixture()
            if mutation == 'reference':
                events[0]['acceptance_reference'] = 'parent'
            elif mutation == 'budget':
                events[0]['budget'] = 20
            elif mutation == 'parent_accept':
                end = next(e for e in events if e['event'] == 'branch_end' and e['generation'] == 2)
                end['source'] = 'same_reward'
            elif mutation == 'prefix':
                files['child-2']['state']['raw_ledger'][1]['action']['action_token_ids'] = [99999]
            elif mutation == 'novelty':
                next(e for e in events if e['event'] == 'branch_start' and e['parent_generation'] == 1)[
                    'effective_recursive_prefix'] = False
            elif mutation == 'generation':
                next(e for e in events if e['event'] == 'branch_end')['generation'] = 9
            else:
                events[-1]['final_ids'] = events[-1]['final_ids'][1:]
            with self.assertRaises(ValueError, msg=mutation):
                validate_question(events, files)

    def test_snapshot_bytes_receipt_identity_and_config_must_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            record = self.record(base)
            self.fixture(directory=base, record=record)
            path = base / 'events.jsonl'
            self.assertEqual(load_question(path, record, 5)['max_generation'], 4)
            with self.assertRaises(ValueError):
                load_question(path, dict(record, run_id='wrong'), 5)
            snapshot = base / 'snapshot-1.json'
            snapshot.write_bytes(snapshot.read_bytes() + b' ')
            with self.assertRaises(ValueError):
                load_question(path, record, 5)
            with self.assertRaises(ValueError):
                verify_events(base, record, 5)  # no complete512 evidence
            Path(record['config']).write_text('{}')
            with self.assertRaises(ValueError):
                verify_events(base, record, 5)

    def test_nonroot_parent_can_revisit_an_actual_ancestor_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            record = self.record(base)
            events, _ = self.fixture('early_ancestor', directory=base, record=record)
            branch = next(e for e in events if e['event'] == 'branch_start' and e['attempt'] == 2)
            self.assertEqual(branch['parent_id'], 'child-1')
            snapshot = json.loads((base / 'snapshot-2.json').read_text())
            self.assertEqual(snapshot['trajectory'], 'root')
            self.assertEqual(snapshot['raw_ledger'], [])
            result = load_question(base / 'events.jsonl', record, 5)
            self.assertEqual(result['genealogical_recursive_branches'], 3)

    def test_complete512_coverage_zero_recursion_and_duplicate_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            record = self.record(base)
            for i in range(512):
                directory = base / 'attempt-one/recursive-events/batch' / str(i)
                directory.mkdir(parents=True)
                self.fixture('shallow', directory=directory, record=record, question='nq:%d' % i)
            result = verify_events(base, record, 5)
            self.assertEqual(result['questions'], 512)
            self.assertEqual(result['branches'], 2048)
            self.assertFalse(result['effective_recursion_observed'])
            self.assertIsNone(result['effective_fraction_of_nonroot_parent_branches'])
            duplicate = base / 'attempt-two/recursive-events/batch/0'
            duplicate.mkdir(parents=True)
            self.fixture('shallow', directory=duplicate, record=record, question='nq:0')
            with self.assertRaises(ValueError):
                verify_events(base, record, 5)


if __name__ == '__main__':
    unittest.main()
