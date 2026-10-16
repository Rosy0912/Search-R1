"""Read-only R1 event and saved-prefix audit, independent of planner diagnostics.

This does not generate trajectories, reconstruct rewards from answers, or assert
that an optimizer consumed the selected group. Sparse recorded rounds alone are
the denominator; zero observed effective recursion is a valid reported result.
"""
import argparse
import hashlib
import json
from pathlib import Path

from release.recursive_bran_a.ablation150_v1 import EVALUATION_STEPS
from release.recursive_bran_a.ablation150_v1.contracts import read_receipt, sha, validate_config_dict
from .prepare_ablation import condition_for

GROUP = 'R1-root-anchored-recursive'


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False).encode()).hexdigest()


def _steps(state):
    return [(tuple(row['action']['action_token_ids']), tuple(row['observation_token_ids']))
            for row in state['raw_ledger']]


def _prefix_expected(root, parent, index, generation):
    """Recompute from persisted ledgers, never call root_prefix_diagnostics."""
    left, right = _steps(parent)[:index], _steps(root)[:index]
    novel = [i for i, step in enumerate(left) if i >= len(right) or step[0] != right[i][0]]
    semantic = [i for i, step in enumerate(left) if i >= len(right) or step != right[i]]
    obs = [i for i, step in enumerate(left) if i < len(right)
           and step[0] == right[i][0] and step[1] != right[i][1]]
    recursive = generation > 0
    return dict(parent_generation=generation, genealogical_recursive=recursive,
        inherited_action_count=len(left), inherited_novel_action_count=len(novel),
        inherited_novel_action_positions=novel, inherited_semantic_difference_count=len(semantic),
        inherited_semantic_difference_positions=semantic,
        inherited_observation_only_difference_count=len(obs),
        inherited_observation_only_difference_positions=obs,
        effective_recursive_prefix=recursive and bool(novel),
        observation_only_recursive_prefix=recursive and not novel and bool(obs),
        parent_prefix_sha256=_digest(left), root_prefix_sha256=_digest(right),
        prefix_comparison='positionwise_action_and_observation_token_ids_v1')


def validate_question(events, trajectories):
    _require(len(events) >= 3 and events[0]['event'] == 'question_start'
             and events[-1]['event'] == 'question_end'
             and [e['event_seq'] for e in events] == list(range(1, len(events) + 1)),
             'Incomplete R1 event sequence')
    first, last = events[0], events[-1]
    _require(first['budget'] == 10 and first['target'] == 5
             and first['initial_reward'] in (0, 1), 'R1 budget/target/reward changed')
    root = first['reference_trajectory_id']
    state = trajectories[root]['state']
    _require(trajectories[root].get('source') == 'initial' and state['trajectory'] == root
             and state['tree'] == first['root_id'], 'Saved R1 root identity differs')
    context = dict(acceptance_reference='initial_root', reference_reward=first['initial_reward'],
                   reference_trajectory_id=root, root_id=first['root_id'],
                   ablation_variant='root_anchored_recursive')
    _require(all(all(e.get(k) == v for k, v in context.items()) for e in events),
             'R1 acceptance reference differs within question')
    accepted, pool, known = [root], [], {root: (first['initial_reward'], 0)}
    attempts = 0
    pending = search = fill = None
    backoff = []
    last_contrast = False
    counts = dict(genealogical_recursive_branches=0, effective_recursive_branches=0,
                  observation_only_recursive_branches=0, accepted_same_reward_as_parent=0)
    maximum_generation = 0
    for event in events[1:-1]:
        kind = event['event']
        _require(fill is None, 'R1 event follows final fill')
        if kind == 'search_start':
            _require(search is None and pending is None and attempts < 10 and len(accepted) < 5,
                     'R1 search opened after termination or during another search')
            eligible = [key for key in accepted if _steps(trajectories[key]['state'])]
            expected = [dict(id=key, generation=known[key][1], weight=1,
                             probability=1 / len(eligible)) for key in eligible]
            _require(event['eligible'] == expected and event['parent_id'] in eligible,
                     'R1 parent distribution differs from uniform accepted trajectories')
            positions = list(range(len(_steps(trajectories[event['parent_id']]['state']))))
            _require(event['legal_positions'] == positions and event['start'] in positions
                     and event['boundary'] == 0 and event['lcps'] == {}
                     and event['position_distribution'] == 'uniform'
                     and event['position_probabilities'] == [1 / len(positions)] * len(positions)
                     and event['max_generation'] is None, 'R1 restored an LCP/position/depth restriction')
            search, backoff, last_contrast = event, list(range(event['start'], -1, -1)), False
        elif kind == 'branch_start':
            _require(search is not None and pending is None and backoff and not last_contrast
                     and attempts < 10 and len(accepted) < 5, 'R1 continuation exceeds search budget')
            index = backoff.pop(0)
            attempts += 1
            parent = search['parent_id']
            _require(event['attempt'] == attempts and event['parent_id'] == parent
                     and event['action_index'] == index and event['boundary'] == 0
                     and event['backoff'] == (index != search['start']), 'R1 changed same-parent backoff')
            diagnostic = _prefix_expected(state, trajectories[parent]['state'], index, known[parent][1])
            _require(all(event.get(k) == v for k, v in diagnostic.items()),
                     'R1 prefix diagnostic disagrees with actual saved action/observation tokens')
            counts['genealogical_recursive_branches'] += diagnostic['genealogical_recursive']
            counts['effective_recursive_branches'] += diagnostic['effective_recursive_prefix']
            counts['observation_only_recursive_branches'] += diagnostic['observation_only_recursive_prefix']
            pending = event
        elif kind == 'branch_end':
            _require(pending is not None and event['status'] == 'complete'
                     and event['attempt'] == attempts, 'R1 lacks complete continuation evidence')
            parent, index = pending['parent_id'], pending['action_index']
            key, reward, generation = event['id'], event['reward'], event['generation']
            parent_reward, parent_generation = known[parent]
            contrast = reward != first['initial_reward']
            _require(key not in known and reward in (0, 1) and generation == parent_generation + 1
                     and event['parent_id'] == parent and event['parent_generation'] == parent_generation
                     and event['parent_reward'] == parent_reward
                     and event['same_reward_as_parent'] == (reward == parent_reward)
                     and event['different_reward_from_root'] == contrast
                     and event['source'] == ('contrast_accept' if contrast else 'same_reward')
                     and event['acceptance_reason'] == ('different_reward_from_initial_root' if contrast
                                                       else 'same_reward_as_initial_root'),
                     'R1 child identity, generation or root-relative acceptance differs')
            child = trajectories[key]['state']
            _require(child['trajectory'] == key and child['tree'] == state['tree']
                     and child['source'] == state['source']
                     and child['raw_ledger'][:index] == trajectories[parent]['state']['raw_ledger'][:index]
                     and len(child['raw_ledger']) >= index,
                     'R1 saved child changed inherited raw prefix or question identity')
            known[key] = reward, generation
            (accepted if contrast else pool).append(key)
            counts['accepted_same_reward_as_parent'] += contrast and reward == parent_reward
            maximum_generation = max(maximum_generation, generation)
            pending, last_contrast = None, contrast
        elif kind == 'search_end':
            _require(search is not None and pending is None and event['attempts'] == attempts
                     and event['accepted'] == len(accepted) and event['pool'] == len(pool),
                     'R1 search-end accounting differs')
            expected_reason = ('contrast_accept' if last_contrast else
                               'boundary_reached' if not backoff else 'budget_exhausted')
            _require(event['reason'] == expected_reason
                     and (last_contrast or not backoff or attempts == 10), 'R1 ended search prematurely')
            search, backoff = None, []
        elif kind == 'fill_end':
            _require(search is None and pending is None, 'R1 fills while a search is open')
            fill = event
        else:
            raise ValueError('Unexpected R1 event: ' + kind)
    _require(fill is not None and search is None and pending is None, 'R1 final fill missing')
    chosen = fill['selected_ids']
    _require(fill['pool_ids'] == pool and len(set(chosen)) == len(chosen)
             and set(chosen).issubset(pool) and len(chosen) == min(5 - len(accepted), len(pool))
             and fill['fill_count'] == len(chosen) and last['final_ids'] == accepted + chosen
             and last['attempts'] == attempts and last['accepted_branches'] == len(accepted) - 1
             and last['same_reward_candidates'] == len(pool) and last['fill_count'] == len(chosen)
             and last['complete_group'] == (len(accepted) + len(chosen) == 5),
             'R1 final group or pool fill differs')
    reason = ('target_reached' if len(accepted) == 5 else 'budget_exhausted' if attempts == 10
              else 'no_expandable_record')
    _require(last['stop_reason'] == reason and (reason != 'no_expandable_record'
             or not any(_steps(trajectories[k]['state']) for k in accepted)),
             'R1 search terminated before its fixed budget/target')
    _require(set(trajectories) == set(known), 'Saved R1 candidate set differs from event set')
    return dict(branches=attempts, accepted=len(accepted) - 1, same_root_reward=len(pool),
                filled=len(chosen), final_trajectories=len(last['final_ids']),
                complete_group=last['complete_group'], max_generation=maximum_generation, **counts)


def load_question(path, record, outer):
    path = Path(path)
    _require(not path.is_symlink(), 'R1 event file must not be a symlink')
    raw = path.read_bytes()
    first = json.loads(raw.split(b'\n', 1)[0])
    if first['outer'] != outer:
        return None
    _require(raw.endswith(b'\n'), 'R1 event stream is unfinished')
    events = [json.loads(line) for line in raw.splitlines()]
    identity = dict(question_id=first['question_id'], outer=outer, run_id=record['run_id'],
                    source_commit=record['source_commit'], config_sha256=record['config_sha256'])
    _require(all(all(e.get(k) == v for k, v in identity.items()) for e in events),
             'R1 event identity differs from frozen receipt')
    ids = [first['reference_trajectory_id']] + [e['id'] for e in events
                                               if e['event'] == 'branch_end' and e['status'] == 'complete']
    _require(len(set(ids)) == len(ids), 'R1 candidate identity is repeated')
    trajectories, files = {}, {}
    for key in ids:
        _require(isinstance(key, str) and Path(key).name == key and key not in ('', '.', '..'),
                 'R1 trajectory ID is not a local filename')
        candidate = path.parent / (key + '.json')
        _require(not candidate.is_symlink(), 'R1 candidate must not be a symlink')
        data = candidate.read_bytes()
        files[candidate.name] = hashlib.sha256(data).hexdigest()
        trajectories[key] = json.loads(data)
    result = validate_question(events, trajectories)
    lineage = {first['reference_trajectory_id']: None}
    lineage.update({e['id']: e['parent_id'] for e in events if e['event'] == 'branch_end'})
    for event in events:
        if event['event'] != 'branch_start':
            continue
        snapshot = path.parent / ('snapshot-%d.json' % event['attempt'])
        _require(not snapshot.is_symlink() and snapshot.resolve() == Path(event['snapshot']).resolve(),
                 'R1 snapshot path differs from persisted local file')
        data = snapshot.read_bytes()
        _require(hashlib.sha256(data).hexdigest() == event['snapshot_sha256'], 'R1 snapshot hash differs')
        value, parent = json.loads(data), trajectories[event['parent_id']]['state']
        # Earlier pre-action snapshots are deliberately inherited by descendants;
        # revisiting before their fork uses a real ancestor snapshot identity.
        ancestors, key = set(), event['parent_id']
        while key is not None:
            ancestors.add(key)
            key = lineage[key]
        _require(value['trajectory'] in ancestors and value['tree'] == parent['tree']
                 and value['source'] == parent['source']
                 and value['raw_ledger'] == parent['raw_ledger'][:event['action_index']],
                 'R1 saved pre-action snapshot differs from selected parent prefix')
        files[snapshot.name] = event['snapshot_sha256']
    return dict(question_id=first['question_id'], **result, events_path=str(path),
                events_sha256=hashlib.sha256(raw).hexdigest(), saved_file_count=len(files),
                saved_file_hashes_digest=_digest(files))


def verify_events(output, record, outer=5):
    _require(outer in EVALUATION_STEPS, 'R1 mechanism audit requires a recorded 5/50/100/150 milestone')
    _require(record['group'] == GROUP and record['condition'] == condition_for(GROUP),
             'Need the exact frozen R1 configuration')
    _require(sha(record['config']) == record['config_sha256'], 'R1 frozen configuration hash changed')
    raw_config = Path(record['config']).read_text()
    try:
        config = json.loads(raw_config)
    except json.JSONDecodeError:
        from omegaconf import OmegaConf
        config = OmegaConf.to_container(OmegaConf.create(raw_config), resolve=True)
    _require(validate_config_dict(config) == record['condition'], 'R1 receipt and runtime variant disagree')
    output = Path(output).resolve()
    paths = sorted(list(output.glob('recursive-events/*/*/events.jsonl'))
                   + list(output.glob('attempt-*/recursive-events/*/*/events.jsonl')))
    questions = {}
    for path in paths:
        _require(path.resolve().is_relative_to(output), 'R1 event path escapes declared output')
        result = load_question(path, record, outer)
        if result is None:
            continue
        key = result['question_id']
        _require(key not in questions, 'Duplicate R1 question at the requested milestone')
        questions[key] = result
    _require(len(questions) == 512, 'Need complete R1 saved events for all 512 questions')
    keys = ('branches', 'accepted', 'same_root_reward', 'filled', 'final_trajectories',
            'genealogical_recursive_branches', 'effective_recursive_branches',
            'observation_only_recursive_branches', 'accepted_same_reward_as_parent')
    totals = {k: sum(q[k] for q in questions.values()) for k in keys}
    recursive, effective, branches = (totals[k] for k in (
        'genealogical_recursive_branches', 'effective_recursive_branches', 'branches'))
    return dict(status='R1_RECORDED_ROUND_MECHANISM_VERIFIED', outer=outer, questions=len(questions),
        **totals, incomplete_groups=sum(not q['complete_group'] for q in questions.values()),
        max_generation=max(q['max_generation'] for q in questions.values()),
        effective_fraction_of_nonroot_parent_branches=effective / recursive if recursive else None,
        effective_fraction_of_all_branches=effective / branches if branches else None,
        genealogical_recursion_observed=recursive > 0, effective_recursion_observed=effective > 0,
        question_evidence=questions,
        limits='Only this sparse recorded milestone; zero recursion is reported, never forced. '
               'Verifies raw-ledger prefixes, snapshot hashes/lineage and selection evidence; '
               'does not independently replay all executor state fields or answer rewards, '
               'optimizer consumption, training completion, generalization or statistical uniformity.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', required=True)
    parser.add_argument('--outer', type=int, required=True)
    parser.add_argument('--output', required=True, help='New audit JSON; existing files are never overwritten')
    args = parser.parse_args()
    record = read_receipt(args.receipt)
    result = verify_events(record['output'], record, args.outer)
    result.update(configuration_receipt=str(Path(args.receipt).resolve()),
                  configuration_receipt_sha256=sha(args.receipt))
    with Path(args.output).open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'question_evidence'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
