import copy
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest

from search_r1.baselines.recursive_ablation import Planner, Options
from search_r1.baselines.recursive_branch import Record
from .c2_evidence import validate_question, load_question, verify_events
from .prepare_ablation import condition_for


class C2Evidence(unittest.TestCase):
    def fixture(self, mode='mixed', directory=None):
        events = []
        files = {}
        snapshots = {}
        identity = dict(question_id='nq:128', outer=5, run_id='run',
                        source_commit='source', config_sha256='config')

        def state(key, steps):
            return dict(trajectory=key, tree='tree', source=0,
                raw_ledger=[dict(action=dict(action_token_ids=list(action)),
                    observation_token_ids=list(observation)) for action, observation in steps])

        def emit(event, **fields):
            if event == 'branch_start' and directory is not None:
                snapshot = copy.deepcopy(planner.parent.state)
                snapshot['raw_ledger'] = snapshot['raw_ledger'][:fields['action_index']]
                path = directory / ('snapshot-%d.json' % fields['attempt'])
                data = json.dumps(snapshot).encode()
                snapshots[path] = data
                fields.update(snapshot=str(path), snapshot_sha256=hashlib.sha256(data).hexdigest())
            events.append(dict(event=event, event_seq=len(events) + 1, **identity, **fields))

        steps = [] if mode == 'empty' else [((100 + i,), (200 + i,)) for i in range(4)]
        initial = state('root', steps)
        files['root'] = dict(state=initial, source='initial')
        root = Record('root', None, 'tree', 0, 0, initial, [None] * len(steps), steps, 'initial')
        planner = Planner(root, random.Random(42), emit, options=Options(variant='random_final'),
                          question_id=identity['question_id'], outer=5)
        while True:
            request = planner.next_request()
            if request is None:
                break
            parent, index, attempt = request
            steps = parent.steps[:index] + [((attempt * 1000 + i,), (400 + i,)) for i in range(index, 4)]
            reward = (1 - parent.reward if mode == 'contrast' else parent.reward if mode == 'same'
                      else [0, 0, 1, 1, 0, 1, 0, 1, 0, 1][attempt - 1])
            key = 'child-%d' % attempt
            value = dict(state=state(key, steps), cost=dict(new_generated_tokens=4 - index))
            files[key] = value
            planner.accept_result(Record(key, parent.trajectory_id, 'tree', parent.generation + 1,
                reward, value['state'], [None] * len(steps), steps, 'pending', value['cost']))
        planner.finish()
        if directory is not None:
            for key, value in files.items():
                (directory / (key + '.json')).write_text(json.dumps(value))
            for path, data in snapshots.items():
                path.write_bytes(data)
            (directory / 'events.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in events))
        return events, files

    def test_replays_real_planner_mixed_contrast_same_and_no_expandable_paths(self):
        for mode in ('mixed', 'contrast', 'same', 'empty'):
            events, files = self.fixture(mode)
            result = validate_question(events, files)
            self.assertEqual(result['candidates'], len(files) - 1)
            self.assertEqual(result['selected'], min(4, len(files) - 1))
        mixed, files = self.fixture()
        result = validate_question(mixed, files)
        self.assertGreater(result['selected_same_reward'], 0)
        # Actual parent generations include recursive extensions, not C1.
        self.assertTrue(any(e.get('generation', 0) > 1 for e in mixed))

    def test_rejects_filtered_pool_duplicate_sample_and_relabelled_training_group(self):
        for mutation in ('filtered', 'duplicate', 'old_group', 'wrong_order'):
            events, files = self.fixture()
            selection, final = events[-2], events[-1]
            if mutation == 'filtered':
                selection['candidate_ids'] = selection['original_final_ids'][1:]
            if mutation == 'duplicate':
                selection['selected_nonroot_ids'][1] = selection['selected_nonroot_ids'][0]
            if mutation == 'old_group':
                final['final_ids'] = selection['original_final_ids']
            if mutation == 'wrong_order':
                selection['selected_nonroot_ids'].reverse()
            with self.assertRaises(ValueError):
                validate_question(events, files)

    def test_rejects_drift_in_original_tree_and_real_trajectory_prefix(self):
        for mutation in ('parent', 'position', 'reward_acceptance', 'file_prefix', 'identity'):
            events, files = self.fixture()
            if mutation == 'parent':
                next(e for e in events if e['event'] == 'search_start')['parent_id'] = 'invented'
            if mutation == 'position':
                next(e for e in events if e['event'] == 'branch_start')['action_index'] = 99
            if mutation == 'reward_acceptance':
                next(e for e in events if e['event'] == 'branch_end')['source'] = 'contrast_accept'
            if mutation == 'file_prefix':
                start = next(e for e in events if e['event'] == 'branch_start' and e['action_index'] > 0)
                end = next(e for e in events if e['event'] == 'branch_end' and e['attempt'] == start['attempt'])
                files[end['id']]['state']['raw_ledger'][0]['action']['action_token_ids'] = [99999]
            if mutation == 'identity':
                files['child-1']['state']['trajectory'] = 'fake'
            with self.assertRaises(ValueError):
                validate_question(events, files)

    def record(self):
        return dict(group='C2-random-final', condition=condition_for('C2-random-final'),
                    run_id='run', source_commit='source', config_sha256='config')

    def test_reads_actual_files_and_rejects_missing_candidate_or_changed_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.fixture(directory=directory)
            path = directory / 'events.jsonl'
            result = load_question(path, self.record(), 5)
            self.assertGreater(len(result['candidate_and_snapshot_file_sha256']), result['candidates'])
            snapshot = directory / 'snapshot-1.json'
            snapshot.write_bytes(snapshot.read_bytes() + b' ')
            with self.assertRaises(ValueError):
                load_question(path, self.record(), 5)
            self.fixture(directory=directory)
            (directory / 'child-1.json').unlink()
            with self.assertRaises(FileNotFoundError):
                load_question(path, self.record(), 5)

    def test_incomplete_round_or_wrong_identity_cannot_be_audited_as_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            directory = output / 'recursive-events/batch/0'
            directory.mkdir(parents=True)
            self.fixture(directory=directory)
            with self.assertRaises(ValueError):
                verify_events(output, self.record(), 5)  # one question is not512
            wrong = dict(self.record(), run_id='other')
            with self.assertRaises(ValueError):
                load_question(directory / 'events.jsonl', wrong, 5)


if __name__ == '__main__':
    unittest.main()
