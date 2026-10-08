"""Small fixtures for actual own-record prefix classification."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from analyze_tree_retained_actions import analyze_rows, load_bytes, prefix_length


def action(tokens, raw, context, observation=None, equal=True):
    observation = observation or []
    return dict(action_token_ids=tokens, raw_generation_token_ids=raw,
                generation_context_token_ids=context, observation_token_ids=observation,
                action_tokens=len(tokens), observation_tokens=len(observation),
                scoring_context_matches_generation=equal)


def leaf(actions):
    timeline = [t for a in actions for t in a['action_token_ids'] + a['observation_token_ids']]
    return dict(row=0, identity={'source': 'fixture'}, token_ids=[10] + timeline,
                metadata={'action_records': actions})


def result(row):
    return analyze_rows([row], {'source': 'fixture'}, 1)


def rejected(row, message):
    try:
        result(row)
    except AssertionError:
        return
    raise AssertionError(message)


def main():
    checks = []
    r = result(leaf([action([1, 2], [1, 2, 99], [10])]))
    assert r['own_record_exact_token_coverage'] == 1 and r['counts']['raw_tail_discarded_after_exact_prefix_records'] == 1
    checks.append('exact_retained_prefix_does_not_require_raw_tail_equality')
    r = result(leaf([action([1, 8, 3], [1, 2, 3], [10])]))
    assert r['counts']['own_record_exact_causal_prefix_token_occurrences'] == 1 and r['counts']['own_record_nonmatching_token_occurrences'] == 2
    checks.append('tokens_after_first_mismatch_remain_context_nonmatches')
    r = result(leaf([action([1, 2, 3], [1, 2], [10])]))
    assert r['counts']['action_extends_past_raw_records'] == 1 and r['counts']['own_record_nonmatching_token_occurrences'] == 1
    assert r['counts']['first_difference_is_final_action_token_records'] == 0
    checks.append('retained_extension_beyond_raw_is_not_false_match')
    row = leaf([action([1, 8], [1, 2], [10], [50]), action([3, 4], [3, 4], [10, 1, 8, 50])])
    r = result(row)
    assert r['counts']['own_record_exact_causal_prefix_token_occurrences'] == 3 and r['counts']['generation_vs_final_prefix_equal_records'] == 2
    assert r['own_record_equal_leaf_weight_coverage'] == .75
    checks.append('prior_retokenized_action_and_observation_enter_next_turn_context')
    row = leaf([action([1, 2], [1, 2], [90], equal=False)])
    r = result(row)
    assert r['own_record_exact_token_coverage'] == 0
    checks.append('different_generation_context_prevents_token_only_match')
    row['metadata']['action_records'][0]['scoring_context_matches_generation'] = True
    rejected(row, 'false stored context claim accepted')
    checks.append('stored_context_flag_must_equal_actual_prefix_ids')
    row = leaf([action([1], [1], [10])])
    for changed in ('identity', 'row', 'token_ids', 'action_tokens', 'bool_token', 'bool_row', 'bool_count'):
        q = copy.deepcopy(row)
        if changed == 'identity': q['identity'] = {'source': 'wrong'}
        elif changed == 'row': q['row'] = 1
        elif changed == 'token_ids': q['token_ids'] = [10, 2]
        elif changed == 'action_tokens': q['metadata']['action_records'][0]['action_tokens'] = 2
        elif changed == 'bool_token': q['metadata']['action_records'][0]['raw_generation_token_ids'] = [True]
        elif changed == 'bool_row': q['row'] = False
        else: q['metadata']['action_records'][0]['action_tokens'] = True
        rejected(q, changed + ' tamper accepted')
    checks.append('identity_row_timeline_length_and_bool_token_tamper_rejected')
    try:
        analyze_rows([row], {'source': 'fixture'}, 1, 2)
    except AssertionError:
        pass
    else:
        raise AssertionError('false expected effective token count accepted')
    checks.append('full_token_count_bound_required')
    with tempfile.TemporaryDirectory() as temp:
        p = Path(temp) / 'bytes'
        p.write_bytes(b'fixture')
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        assert load_bytes(p, digest) == b'fixture'
        p.write_bytes(b'changed')
        try:
            load_bytes(p, digest)
        except AssertionError:
            pass
        else:
            raise AssertionError('changed payload accepted')
    checks.append('same_read_file_hash_binding')
    assert prefix_length([1, 2], [1, 2]) == 2
    assert prefix_length([], [1]) == 0
    checks.append('prefix_end_boundaries')
    print(json.dumps(dict(status='PASS_CPU_RETAINED_ACTION_PREFIX_FIXTURES', checks=checks,
                         count=len(checks), actual_assets_loaded=False, gpu_calls=0)))


if __name__ == '__main__':
    main()
