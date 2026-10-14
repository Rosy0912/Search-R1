"""Replay C2's persisted candidates through the unchanged original planner.

The replay consumes recorded choices, never calls the model, and independently
reconstructs the final sample from the question/outer selection RNG. It checks
actual candidate files; event labels alone cannot establish the mechanism.
"""
import hashlib
import json
from pathlib import Path

from search_r1.baselines.recursive_branch import Planner, Record
from search_r1.baselines.recursive_ablation import selection_rng
from .prepare_ablation import condition_for


_CONTEXT = {'event_seq', 'question_id', 'outer', 'run_id', 'source_commit',
            'config_sha256', 'snapshot', 'snapshot_sha256'}


def _steps(state):
    return [(tuple(row['action']['action_token_ids']), tuple(row['observation_token_ids']))
            for row in state['raw_ledger']]


def validate_question(events, trajectories, seed=42):
    if (len(events) < 5 or events[0]['event'] != 'question_start'
            or [e['event_seq'] for e in events] != list(range(1, len(events) + 1))
            or [e['event'] for e in events[-3:]] !=
                ['original_selection_end', 'final_selection', 'question_end']):
        raise ValueError('Incomplete C2 original-tree and final-selection stream')
    first, original_end, selection, final = events[0], events[-3], events[-2], events[-1]
    if first['budget'] != 10 or first['target'] != 5 or first['initial_reward'] not in (0, 1):
        raise ValueError('C2 changed the original budget/target/reward')
    root_id = original_end['final_ids'][0]
    root_file = trajectories[root_id]
    root_state = root_file['state']
    if (root_file.get('source') != 'initial' or root_state['trajectory'] != root_id
            or root_state['tree'] != first['root_id']):
        raise ValueError('C2 root does not identify the saved initial trajectory')
    original_events = [{k: v for k, v in e.items() if k not in _CONTEXT} for e in events[:-2]]
    original_events[-1]['event'] = 'question_end'
    cursor = 0

    def emit(event, **fields):
        nonlocal cursor
        if cursor >= len(original_events) or original_events[cursor] != dict(event=event, **fields):
            raise ValueError('C2 candidate construction differs from original planner at event %d' % cursor)
        cursor += 1

    class RecordedChoices:
        def choice(self, values):
            event = original_events[cursor]
            if event['event'] != 'search_start' or not values:
                raise ValueError('Unexpected original-tree random choice')
            if isinstance(values[0], tuple):
                matches = [v for v in values if v[0].trajectory_id == event['parent_id']]
                if len(matches) != 1:
                    raise ValueError('C2 selected an ineligible parent')
                return matches[0]
            if event['start'] not in values:
                raise ValueError('C2 selected an illegal original branch position')
            return event['start']

        def sample(self, values, count):
            event = original_events[cursor]
            ids = [r.trajectory_id for r in values]
            chosen = event['selected_ids']
            if (event['event'] != 'fill_end' or len(chosen) != count
                    or len(set(chosen)) != count or not set(chosen).issubset(ids)):
                raise ValueError('C2 changed the original fill rule')
            return [values[ids.index(key)] for key in chosen]

    root_steps = _steps(root_state)
    root = Record(root_id, None, first['root_id'], 0, first['initial_reward'],
                  root_state, [None] * len(root_steps), root_steps, 'initial')
    planner = Planner(root, RecordedChoices(), emit, target=5, budget=10)
    used = {root_id}
    while True:
        request = planner.next_request()
        if request is None:
            break
        parent, index, attempt = request
        event = original_events[cursor]
        if event['event'] != 'branch_end' or event['status'] != 'complete' or event['attempt'] != attempt:
            raise ValueError('C2 candidate lacks a completed real continuation')
        key = event['id']
        value = trajectories[key]
        state = value['state']
        if (key in used or state['trajectory'] != key or state['tree'] != root_state['tree']
                or state['source'] != root_state['source']):
            raise ValueError('C2 continuation file has a different trajectory/question identity')
        used.add(key)
        steps = _steps(state)
        planner.accept_result(Record(key, parent.trajectory_id, first['root_id'], event['generation'],
            event['reward'], state, [None] * len(steps), steps, 'pending', value['cost']))
    original = planner.finish()
    if cursor != len(original_events) or set(trajectories) != used:
        raise ValueError('C2 candidate replay did not consume the exact persisted candidate set')
    candidates = [r for r in planner.accepted + planner.pool if r is not root]
    ids = [r.trajectory_id for r in candidates]
    count = min(4, len(ids))
    expected = selection_rng(seed, first['question_id'], first['outer']).sample(ids, count)
    if {k: v for k, v in selection.items() if k not in _CONTEXT} != dict(event='final_selection',
            selection='uniform_without_replacement',
            original_final_ids=[r.trajectory_id for r in original], candidate_ids=ids,
            selected_nonroot_ids=expected, marginal_probability=(count / len(ids) if ids else 0.),
            selection_rng_protocol='sha256_seed_question_outer_v1'):
        raise ValueError('C2 final sample omits real candidates or differs from independent uniform RNG replay')
    chosen = [r for r in candidates if r.trajectory_id in expected]
    expected_final = dict(event='question_end', final_ids=[root_id] + expected,
        attempts=planner.attempts, accepted_branches=len(planner.accepted) - 1,
        same_reward_candidates=len(planner.pool), fill_count=sum(r.source == 'failure_fill' for r in chosen),
        complete_group=count == 4, stop_reason=planner.reason, ablation_variant='random_final')
    if {k: v for k, v in final.items() if k not in _CONTEXT} != expected_final:
        raise ValueError('C2 reported training group differs from the actual uniform final sample')
    return dict(root=root_id, branches=planner.attempts, candidates=len(ids), selected=count,
        contrast_candidates=len(planner.accepted) - 1, same_reward_candidates=len(planner.pool),
        selected_contrast=sum(r in planner.accepted for r in chosen),
        selected_same_reward=sum(r in planner.pool for r in chosen))


def load_question(path, record, outer):
    path = Path(path)
    raw = path.read_bytes()
    first_line = raw.split(b'\n', 1)[0]
    if first_line and json.loads(first_line)['outer'] != outer:
        return None
    if not raw.endswith(b'\n'):
        raise ValueError('C2 event stream is unfinished')
    rows = [json.loads(line) for line in raw.splitlines()]
    if not rows or rows[0]['outer'] != outer:
        return None
    question = rows[0]['question_id']
    expected = dict(question_id=question, outer=outer, run_id=record['run_id'],
                    source_commit=record['source_commit'], config_sha256=record['config_sha256'])
    if any(any(row[k] != v for k, v in expected.items()) for row in rows):
        raise ValueError('C2 event identity changed within a question')
    original = next(row for row in rows if row['event'] == 'original_selection_end')
    ids = [original['final_ids'][0]] + [row['id'] for row in rows if row['event'] == 'branch_end']
    if len(set(ids)) != len(ids):
        raise ValueError('C2 repeated a real candidate identity')
    trajectories = {}
    files = {}
    for key in ids:
        candidate = path.parent / (key + '.json')
        if candidate.parent != path.parent or candidate.is_symlink():
            raise ValueError('C2 candidate must be a local persisted trajectory file')
        data = candidate.read_bytes()
        files[candidate.name] = hashlib.sha256(data).hexdigest()
        trajectories[key] = json.loads(data)
    for row in rows:
        if row['event'] == 'branch_start':
            snapshot = path.parent / ('snapshot-%d.json' % row['attempt'])
            data = snapshot.read_bytes()
            if str(snapshot) != row['snapshot'] or snapshot.is_symlink() or hashlib.sha256(data).hexdigest() != row['snapshot_sha256']:
                raise ValueError('C2 pre-action snapshot bytes differ')
            files[snapshot.name] = row['snapshot_sha256']
            state = json.loads(data)
            parent = trajectories[row['parent_id']]['state']
            if (state['trajectory'] != row['parent_id'] or state['tree'] != parent['tree']
                    or _steps(state) != _steps(parent)[:row['action_index']]):
                raise ValueError('C2 pre-action snapshot does not match the selected parent prefix')
    result = validate_question(rows, trajectories, record['condition']['selection_seed'])
    return dict(question_id=question, **result, events_sha256=hashlib.sha256(raw).hexdigest(),
                candidate_and_snapshot_file_sha256=files)


def verify_events(output, record, outer=5):
    if record['group'] != 'C2-random-final' or record['condition'] != condition_for('C2-random-final'):
        raise ValueError('Need the exact C2 random-final configuration')
    questions = {}
    for path in sorted((Path(output) / 'recursive-events').glob('*/*/events.jsonl')):
        result = load_question(path, record, outer)
        if result is None:
            continue
        key = result['question_id']
        if key in questions:
            raise ValueError('Duplicate C2 question in the requested complete round')
        questions[key] = result
    if len(questions) != 512:
        raise ValueError('Need complete C2 candidate evidence for all512 actual training questions')
    return dict(outer=outer, questions=len(questions),
        original_candidate_tree_replayed=True, final_uniform_selection_replayed=True,
        branches=sum(q['branches'] for q in questions.values()),
        selected_same_reward=sum(q['selected_same_reward'] for q in questions.values()),
        question_evidence=questions,
        limits='Verifies persisted tree construction and final sample, not optimizer use or training completion; those require separate native/round audits.')
