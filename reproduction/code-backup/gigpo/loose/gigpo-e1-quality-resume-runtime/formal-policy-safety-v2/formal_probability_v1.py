"""Pure controls for every actual outer's sampler/scoring identity.

    Structural identity has no tolerance. Numerical tolerances must come from the
same accepted source's actual probability evidence, separately from PPO guards.
This module never repairs a prefix, token, mask, log-prob, advantage or reward.
"""
import hashlib
import json
import math
import re


def require(value, reason):
    if not value:
        raise ValueError(reason)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def validate_tolerances(value):
    names = {'actor_recompute_abs_tolerance', 'actor_vllm_abs_tolerance',
             'actor_recompute_joint_abs_tolerance', 'actor_vllm_joint_abs_tolerance'}
    require(isinstance(value, dict) and set(value) == names,
            'Missing separately accepted probability tolerances')
    require(all(isinstance(n, (int, float)) and not isinstance(n, bool)
                and math.isfinite(n) and n >= 0 for n in value.values()),
            'Probability tolerances are null/invalid; PPO KL limits cannot replace them')


def validate_action(action, tolerances, pad_token_id):
    """Validate one complete processed action against unchanged raw sampling.

    Keep the engine's exact token buffer, including any tensor right-padding.
    Frozen get_eos_mask does not prove sampled length when no EOS is present.
    Only the complete effective action prefix has probability mass: neither
    deleting an interior pad nor using discarded tails to set tolerance is valid.
    """
    require(action['step_id'] >= 0 and action['step_id'] == int(action['step_id']),
            'Invalid actual action ID')
    generation = action['generation_prefix']
    scoring = action['scoring_prefix']
    final_scoring = action['final_scoring_prefix']
    require(generation == scoring == final_scoring == action['engine_prompt_token_ids']
            == action['worker_prompt_token_ids'], 'Actual sampler/scoring prefix differs')
    require(pad_token_id not in generation, 'Effective sampler prefix contains an interior pad')
    complete = action['complete_processed_action_tokens']
    retained = action['retained_action_tokens']
    effective = action['effective_scored_tokens']
    require(complete == retained == effective,
            'Scored action is not the complete processed action; cap/timeline differs')
    raw = action['raw_sampled_tokens']
    raw_mask = action['raw_sampled_response_mask']
    require(len(raw) == len(raw_mask) and all(v in (0, 1) for v in raw_mask),
            'Missing raw sampled response mask')
    require(len(complete) <= len(raw) and raw_mask[:len(complete)] == [1] * len(complete),
            'Sampling response mask has a hole inside the complete effective action')
    require(pad_token_id not in raw[:len(complete)], 'Interior sampled action pad cannot be removed')
    require(raw[:len(complete)] == complete,
            'Processed action is not the exact raw chosen-token prefix')
    require(action['worker_response_tokens'][:len(complete)] == complete,
            'Per-turn actor scored different response tokens')
    length = len(complete)
    require(action['sampler_runner_positions_observed'] is False
            and action['engine_position_evidence_scope'] == 'DERIVED_BACKEND_CONTRACT_EXPECTATION',
            'This observer does not capture actual vLLM model-runner position tensors')
    prefix_positions = list(range(len(generation)))
    action_positions = list(range(len(generation), len(generation) + length))
    require(action['worker_prompt_position_ids'] == prefix_positions
            and action['worker_action_position_ids'] == action_positions
            and action['scoring_action_position_ids'] == action_positions,
            'Per-turn/full HF positions disagree with the default causal-position expectation')
    require(action['worker_action_mask'] == action['scoring_action_mask'] == [1] * length,
            'Actual sampling/scoring effective mask differs')
    parameters = action['sampling_parameters']
    require(parameters['temperature'] == 1 and parameters['top_p'] == 1
            and parameters['top_k'] == -1 and parameters['n'] == 1
            and parameters['min_p'] == 0 and parameters['best_of'] == 1
            and parameters['logits_processors_count'] == 0
            and parameters['repetition_penalty'] == 1
            and parameters['presence_penalty'] == parameters['frequency_penalty'] == 0,
            'Actual sampler distribution differs from common experiment')
    require(action['full_scoring_temperature'] == action['per_turn_scoring_temperature'] == 1,
            'HF scoring temperature differs from actual sampler')
    names = ('full_old_log_probs', 'per_turn_old_log_probs', 'raw_sampled_log_probs')
    values = [action[key] for key in names]
    require(all(len(v) == length and all(math.isfinite(x) for x in v) for v in values),
            'Missing/nonfinite complete action probabilities')
    full, turn, sampled = values
    full_turn = [a - b for a, b in zip(full, turn)]
    turn_sampled = [a - b for a, b in zip(turn, sampled)]
    full_sampled = [a - b for a, b in zip(full, sampled)]
    maximum = lambda rows: max(map(abs, rows), default=0.)
    require(maximum(full_turn) <= tolerances['actor_recompute_abs_tolerance'],
            'Full/per-turn actor token probability differs')
    require(maximum(turn_sampled) <= tolerances['actor_vllm_abs_tolerance']
            and maximum(full_sampled) <= tolerances['actor_vllm_abs_tolerance'],
            'Actual sampler/actor token probability differs')
    require(abs(sum(full_turn)) <= tolerances['actor_recompute_joint_abs_tolerance'],
            'Full/per-turn complete joint action probability differs')
    require(abs(sum(turn_sampled)) <= tolerances['actor_vllm_joint_abs_tolerance']
            and abs(sum(full_sampled)) <= tolerances['actor_vllm_joint_abs_tolerance'],
            'Actual sampler/actor complete joint action probability differs')
    return dict(step_id=action['step_id'], effective_tokens=length,
                full_vs_turn_token_max=maximum(full_turn), turn_vs_sampled_token_max=maximum(turn_sampled),
                full_vs_sampled_token_max=maximum(full_sampled), full_vs_turn_joint=sum(full_turn),
                turn_vs_sampled_joint=sum(turn_sampled), full_vs_sampled_joint=sum(full_sampled))


def validate_outer(records, tolerances, pad_token_id, expected_rows=2560):
    validate_tolerances(tolerances)
    require(len(records) == expected_rows and {r['source'] for r in records} == set(range(expected_rows)),
            'Missing/duplicate actual full rollout trajectories')
    require(len({r['trajectory_uid'] for r in records}) == expected_rows,
            'Missing/duplicate actual trajectory identity')
    require(all(r['actions'] for r in records), 'Missing actual action timeline')
    results = []
    for row in records:
        actions = row['actions']
        require([a['step_id'] for a in actions] == list(range(len(actions))),
                'Missing/duplicate/reordered actual action IDs')
        for action in actions:
            results.append(dict(source=row['source'], trajectory_uid=row['trajectory_uid'],
                                **validate_action(action, tolerances, pad_token_id)))
    require(sum(r['effective_tokens'] for r in results) > 0,
            'Every actual effective action is empty')
    return dict(status='STRUCTURAL_AND_PROBABILITY_CHECKS_ONLY_NOT_INDEPENDENT_ACCEPTANCE',
                trajectories=expected_rows, actions=len(results), effective_tokens=sum(r['effective_tokens'] for r in results),
                prefix_mismatches=0, action_token_mismatches=0, hf_position_mask_mismatches=0,
                sampler_runner_positions_observed=False,
                engine_position_evidence_scope='DERIVED_BACKEND_CONTRACT_EXPECTATION',
                action_checks=results)


def validate_live_arms(rows, outer_step, manifest_sha256):
    """Actual all-worker post-sampling state identity; never infer it from tokens."""
    require(len(rows) == 8 and {row['rank'] for row in rows} == set(range(8)),
            'Missing/duplicate actual eight worker arms')
    require(len({(row['pid'], row['start_ticks']) for row in rows}) == 8,
            'Duplicate/replaced actual worker arm identities')
    for row in rows:
        require(row['outer_step'] == outer_step and row['manifest_sha256'] == manifest_sha256
                and type(row['pid']) is int and row['pid'] > 0 and str(row['start_ticks']).isdecimal()
                and re.fullmatch(r'[0-9a-f]{64}', str(row['complete_pause_state_sha256']))
                and isinstance(row['arm_receipt'], dict), 'Actual arm state/provenance incomplete')
    return rows


def collective_gate(records, expected_world=8):
    """Every worker reaches this decision before invoking the original actor RPC."""
    require(len(records) == expected_world and {r['rank'] for r in records} == set(range(expected_world)),
            'Missing real distributed admission ranks')
    require(len({(r['pid'], r['start_ticks']) for r in records}) == expected_world,
            'Duplicate/replaced worker identities')
    require(all(isinstance(r['pid'], int) and r['pid'] > 0 and str(r['start_ticks']).isdecimal()
                and isinstance(r['outer_step'], int) and 26 <= r['outer_step'] <= 300
                and re.fullmatch(r'[0-9a-f]{40}', str(r['source_commit']))
                and str(r['job_id']).isdecimal() and isinstance(r['run_id'], str) and r['run_id']
                for r in records), 'Incomplete actual process/source/outer identity')
    require(all(r['local_ok'] is True and r['error'] is None for r in records),
            'Distributed probability admission rejected; zero Adam writes')
    for key in ('outer_step', 'source_commit', 'job_id', 'run_id', 'global_manifest_sha256',
                'global_batch_digest', 'production_artifact_sha256', 'probability_acceptance_sha256',
                'global_arm_sha256'):
        require(len({r[key] for r in records}) == 1, 'Ranks disagree before any Adam: ' + key)
    require(all(re.fullmatch(r'[0-9a-f]{64}', r['local_batch_digest']) for r in records),
            'Missing actual local scored-batch digest')
    require(all(re.fullmatch(r'[0-9a-f]{64}', str(r[key])) for r in records
                for key in ('global_manifest_sha256', 'global_batch_digest',
                            'production_artifact_sha256', 'probability_acceptance_sha256', 'global_arm_sha256')),
            'Missing actual source/global/probability acceptance SHA')
    return dict(status='ALL_RANKS_MATCH_BEFORE_ORIGINAL_UPDATE_RPC', outer_step=records[0]['outer_step'],
                global_manifest_sha256=records[0]['global_manifest_sha256'],
                global_batch_digest=records[0]['global_batch_digest'], global_arm_sha256=records[0]['global_arm_sha256'],
                actual_adam_writes_at_gate=0)


def collective_rejection_gate(records, expected_world=8):
    """A driver refusal is acknowledged only after all rank/journal errors agree."""
    require(len(records) == expected_world and {r['rank'] for r in records} == set(range(expected_world)),
            'Missing actual distributed rejection ranks')
    require(len({(r['pid'], r['start_ticks']) for r in records}) == expected_world,
            'Duplicate/replaced actual rejection workers')
    require(all(type(r['pid']) is int and r['pid'] > 0 and str(r['start_ticks']).isdecimal()
                and type(r['outer_step']) is int and 26 <= r['outer_step'] <= 300
                and re.fullmatch(r'[0-9a-f]{40}', str(r['source_commit']))
                and str(r['job_id']).isdecimal() and isinstance(r['run_id'], str) and r['run_id']
                and re.fullmatch(r'[0-9a-f]{64}', str(r['failure_sha256']))
                and isinstance(r['failure_path'], str) and r['failure_path'].startswith('/')
                for r in records), 'Incomplete actual rejection provenance')
    require(all(r['local_ok'] is True and r['error'] is None for r in records),
            'Rejection boundary/evidence unreadable on an actual rank; zero original update RPC')
    for key in ('outer_step', 'source_commit', 'job_id', 'run_id', 'failure_path', 'failure_sha256'):
        require(len({r[key] for r in records}) == 1, 'Ranks disagree on actual refusal: ' + key)
    return dict(status='ALL_RANKS_REJECTED_BEFORE_ORIGINAL_UPDATE_RPC',
                outer_step=records[0]['outer_step'], failure_path=records[0]['failure_path'],
                failure_sha256=records[0]['failure_sha256'], actual_adam_writes_at_gate=0,
                all_ranks=records)


def processed_tokens_without_right_padding(tokens, pad_token_id):
    """Preserve complete processed token IDs; an interior PAD is structural failure."""
    end = len(tokens)
    while end and tokens[end - 1] == pad_token_id:
        end -= 1
    result = tokens[:end]
    require(pad_token_id not in result, 'Interior processed PAD cannot be silently deleted')
    return result
