"""Fail-closed controls for isolated healthy25 calibration, never formal acceptance."""
import math
import re
from pathlib import PurePosixPath

SCOPE = 'GIGPO_POLICY_SAFETY_HEALTHY25_CALIBRATION'
NATIVE_SCOPE = 'GIGPO_POLICY_SAFETY_NATIVE_ONLY'
MANIFEST = '3a073aba77147db7d581cd514ee5c4ae34965e5ff9c163716bd25b47aeb478e4'
CHECKPOINT = '/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004/runs/gigpo-e1-instruct300-job3185-1791112239036063948/checkpoints/actor/global_step_25'
DATA = '0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'


def validate_guard_binding(binding, native_identity):
    """The spawned guard must be the exact guard actually accepted by new native."""
    identity = {}
    for key, filename in (('resource_guard', 'resource_guard_v2.py'),
                          ('resource_ownership', 'resource_ownership_v2.py')):
        item = binding[key]
        path = PurePosixPath(item['path'])
        require(path.is_absolute() and path.name == filename, 'Unbound/wrong resource guard path: ' + key)
        require(bool(re.fullmatch(r'[0-9a-f]{64}', item['sha256'])), 'Invalid resource guard SHA: ' + key)
        identity[key] = dict(path=str(path), sha256=item['sha256'])
        files = [record for record in binding['files'] if record['path'] == str(path)]
        require(len(files) == 1 and files[0]['sha256'] == item['sha256'],
                'Guard/module must have unique full binding file coverage: ' + key)
    require(PurePosixPath(identity['resource_guard']['path']).parent ==
            PurePosixPath(identity['resource_ownership']['path']).parent,
            'Guard must import its bound same-directory ownership module')
    require(native_identity == identity, 'Calibration guard differs from actual full native acceptance')
    return identity


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def validate_binding(binding, actual_source):
    # Frozen source authorizer deliberately only admits this explicit isolated scope.
    require(binding['status'] == 'REVIEWED_FOR_NATIVE_ONLY', 'No isolated independent review')
    require(binding['validation_scope'] == NATIVE_SCOPE, 'Wrong source admission scope')
    require(binding['calibration_scope'] == SCOPE, 'Missing distinct calibration scope')
    require(binding['allowed_safety_modes'] == ['calibration'], 'Wrong execution mode')
    require(binding['code_commit'] == actual_source and re.fullmatch(r'[0-9a-f]{40}', actual_source),
            'Unfrozen source identity')
    require(binding['restore_step'] == 25 and binding['restore_checkpoint'] == CHECKPOINT
            and binding['restore_manifest_sha256'] == MANIFEST, 'Wrong healthy full restore')
    require(binding['common_manifest_sha256'] == DATA, 'Wrong common data identity')
    phase = binding['calibration_phase']
    require(phase in ('probability_probe_only', 'probability_probe_then_wait', 'one_outer'), 'Unknown isolated calibration phase')
    require(binding['calibration_actual_outer_budget'] == (0 if phase == 'probability_probe_only' else 1)
            and binding['retained_training_updates_contributed'] == 0, 'Wrong isolated calibration budget')
    require(binding['optimizer_steps_per_outer'] == 10, 'Wrong optimizer budget')
    require(binding['source_native_acceptance']['status'] == 'PASS'
            and binding['source_native_acceptance']['code_commit'] == actual_source,
            'Missing exact source eight-GPU acceptance')
    validate_guard_binding(binding, binding['source_native_acceptance']['guard_identity'])
    require(binding['target_old_policy_kl'] is None and binding['negative_tail_limit'] is None,
            'Calibration cannot invent production thresholds')
    require(binding['healthy_baseline_f1_floor'] == .3231979640745351 / 2,
            'Health floor must be the authorized half of selected actual fixed512 F1')
    require(binding['prefix_mismatch_limit'] == 0 and binding['action_token_mismatch_limit'] == 0,
            'Prefix/token identity cannot be relaxed')
    for key in ('actor_recompute_abs_tolerance', 'actor_vllm_abs_tolerance',
                'actor_recompute_joint_abs_tolerance', 'actor_vllm_joint_abs_tolerance'):
        value = binding[key]
        if phase == 'one_outer':
            require(isinstance(value, (float, int)) and math.isfinite(value) and value > 0,
                    'Missing reviewed probability comparison tolerance: ' + key)
        else:
            require(value is None, 'Probability probe only observes and cannot fabricate tolerances')
    if phase == 'one_outer':
        require(bool(binding['probability_tolerance_evidence']),
                'Probability tolerances require bound GPU evidence, not KL threshold absorption')
    else:
        require(binding['probability_tolerance_evidence'] == [], 'Probe cannot claim prior acceptance')
    if phase == 'probability_probe_then_wait':
        require(isinstance(binding['probability_review_wait_seconds'], int)
                and 0 < binding['probability_review_wait_seconds'] <= 43200, 'Invalid bounded probability review wait')


def validate_probability_approval(approval, identity, binding):
    require(approval['status'] == 'PASS'
            and approval['scope'] == 'GIGPO_HEALTHY25_PROBABILITY_IDENTITY', 'Missing separate probability identity acceptance')
    for key in ('code_commit', 'job_id', 'run_id', 'material_manifest_sha256', 'native_binding_sha256',
                'post_sampling_worker_identities', 'post_sampling_worker_pause_sha256',
                'post_sampling_driver_complete_state_sha256', 'post_sampling_pause_raw_sha256'):
        require(approval[key] == identity[key], 'Probability approval binds different actual evidence: ' + key)
    from calibration_state_v1 import validate_pause_workers, state_digest
    workers = validate_pause_workers(identity['post_sampling_worker_identities'])
    require(state_digest(workers) == identity['post_sampling_worker_pause_sha256'],
            'Approval omitted actual eight-worker post-sampling identities')
    require(approval['restore_manifest_sha256'] == MANIFEST and approval['common_manifest_sha256'] == DATA,
            'Probability review checkpoint/data identity differs')
    require(approval['rank_count'] == 8 and approval['sampled_trajectory_count'] == 2560
            and approval['prefix_mismatch_count'] == 0 and approval['action_token_mismatch_count'] == 0
            and approval['same_position_ids_and_mask'] is True, 'Structural probability identity not proven')
    for key in ('actor_recompute_abs_tolerance', 'actor_vllm_abs_tolerance',
                'actor_recompute_joint_abs_tolerance', 'actor_vllm_joint_abs_tolerance'):
        value = approval[key]
        require(isinstance(value, (float, int)) and math.isfinite(value) and value > 0,
                'Independent probability review omitted actual numerical tolerance')
    require(approval['production_thresholds_selected'] is False
            and approval['target_old_policy_kl'] is None and approval['negative_tail_limit'] is None,
            'Probability acceptance cannot select production KL/tail thresholds')
    admitted = dict(binding)
    admitted['actor_recompute_abs_tolerance'] = approval['actor_recompute_abs_tolerance']
    admitted['actor_vllm_abs_tolerance'] = approval['actor_vllm_abs_tolerance']
    admitted['actor_recompute_joint_abs_tolerance'] = approval['actor_recompute_joint_abs_tolerance']
    admitted['actor_vllm_joint_abs_tolerance'] = approval['actor_vllm_joint_abs_tolerance']
    return admitted


def validate_alignment(records, expected_rows=2560):
    """Every retained action has its sampled prefix and exact sampled token prefix."""
    require(len(records) == expected_rows, 'Missing actual rollout row coverage')
    require({r['source'] for r in records} == set(range(expected_rows)), 'Duplicate/missing rollout IDs')
    prefix_bad = token_bad = actions = 0
    for record in records:
        require(record['actions'], 'Missing complete generated action records')
        for action in record['actions']:
            actions += 1
            ids = action['retained_action_tokens']
            if not ids:
                continue  # Empty actor action contributes no probability mass, same as the objective.
            prefix_bad += action['generation_prefix'] != action['scoring_prefix']
            sampled = action['sampled_tokens']
            token_bad += sampled[:len(ids)] != ids
            require(len(sampled) >= len(ids), 'Retokenized action longer than sampled sequence')
    return dict(trajectory_rows=len(records), recorded_actions=actions,
                prefix_mismatch_count=prefix_bad, action_token_mismatch_count=token_bad)


def admit_alignment(alignment):
    require(alignment['prefix_mismatch_count'] == 0,
            'Generation prefix differs from full-trajectory old-policy prefix; preserve and investigate')
    require(alignment['action_token_mismatch_count'] == 0,
            'Retokenized action differs from sampled tokens; preserve and investigate')
    for key in ('position_ids_mismatch_count', 'effective_mask_mismatch_count', 'engine_prefix_mismatch_count'):
        require(key in alignment and alignment[key] == 0, 'Missing/different actual sampling structural identity: ' + key)


def validate_probability_receipts(records, binding):
    require(records and {r['rank'] for r in records} == set(range(8)), 'Missing real eight-rank probability checks')
    require({r['serial'] for r in records} == set(range(max(r['serial'] for r in records) + 1)),
            'Incomplete sampled turn serials')
    require(len({(r['serial'], r['rank']) for r in records}) == len(records)
            and len(records) == 8 * (max(r['serial'] for r in records) + 1),
            'Incomplete per-turn eight-rank probability coverage')
    for record in records:
        require(record['source_commit'] == binding['code_commit'] and record['comparison_scope'] == 'SAME_PREFIX_SAME_TOKENS',
                'Wrong probability identity')
        require(record['effective_probability_scope'] == 'RETAINED_ACTOR_ACTION_PREFIX_EXCLUDES_DISCARDED_AND_PADDED_ROWS',
                'Discarded generated tails or padding cannot define actor admission')
        require(record['temperature'] == 1 and record['top_p'] == 1 and record['top_k'] == -1
                and record['n'] == 1, 'Sampling distribution changed')
        require(record['valid_tokens'] >= 0 and record['nonfinite_effective_count'] == 0,
                'No finite effective probability comparison')
        require(record['repeated_probability_probe_rng_unchanged'], 'Probability observation changed RNG')
        require(record['repeated_probability_probe_complete_state_unchanged']
                and record['repeated_probe_identity_before'] == record['repeated_probe_identity_after'],
                'Probability observation changed model/internal/Adam/scheduler/RNG; zero Adam')
        for value in (record['actor_old_max_abs_delta'], record['actor_vllm_max_abs_delta']):
            require(math.isfinite(value) and value >= 0, 'Nonfinite probability error')
        require(record['actor_old_max_abs_delta'] <= binding['actor_recompute_abs_tolerance'],
                'Repeated current/old actor probabilities differ')
        require(record['actor_vllm_max_abs_delta'] <= binding['actor_vllm_abs_tolerance'],
                'Actor versus actual sampling log-prob differs; do not absorb with KL threshold')
    require(sum(r['valid_tokens'] for r in records) > 0, 'Global effective sampled tokens empty')


def validate_action_probability_checks(records, binding):
    require(records and sum(r['effective_tokens'] for r in records) > 0, 'Empty globally scored actions')
    for row in records:
        require(row['same_position_ids'] and row['same_effective_mask'] and row['engine_prefix_matches_generation'],
                'Actual position/mask/engine prefix identity differs')
        values = (row['full_trajectory_vs_sample_prefix_old_max_abs'], row['full_vs_prefix_old_joint_sum_delta'],
                  row['full_vs_vllm_joint_sum_delta'], row['prefix_old_vs_vllm_joint_sum_delta'])
        require(all(math.isfinite(v) for v in values), 'Nonfinite complete action probability difference')
        require(row['full_trajectory_vs_sample_prefix_old_max_abs'] <= binding['actor_recompute_abs_tolerance'],
                'Full trajectory actor token probability differs from per-turn prefix')
        require(abs(row['full_vs_prefix_old_joint_sum_delta']) <= binding['actor_recompute_joint_abs_tolerance'],
                'Full trajectory actor joint probability differs from per-turn prefix')
        require(abs(row['full_vs_vllm_joint_sum_delta']) <= binding['actor_vllm_joint_abs_tolerance']
                and abs(row['prefix_old_vs_vllm_joint_sum_delta']) <= binding['actor_vllm_joint_abs_tolerance'],
                'Actual sampler versus scored complete joint action probability differs')


def summarize_guards(rank_records):
    require(len(rank_records) == 8, 'Missing eight-rank calibration monitor logs')
    for rank, rows in enumerate(rank_records):
        require(len(rows) == 20, 'Exactly pre/post for ten actual Adam writes required')
        require([r['actual_optimizer_writes'] for r in rows] == [v for n in range(10) for v in (n, n + 1)],
                'Incomplete actual optimizer accounting')
        require([r['phase'] for r in rows] == [v for _ in range(10) for v in ('before', 'after')],
                'Wrong monitor placement')
        require(all(r['rank'] == rank and r['execution_mode'] == 'calibration'
                    and r['admission_scope'] == 'numeric_only' and r['allowed']
                    and not r['monitor_identity_changed'] for r in rows), 'Wrong/invalid monitor scope')
        require(all(r['fixed_monitor_hashes'] == rows[0]['fixed_monitor_hashes'] for r in rows),
                'Monitor/old changed during rollout reuse')
        for row in rows:
            if row['phase'] == 'after':
                norm = row['gradient_norm_after_clip']
                require(math.isfinite(norm) and 0 <= norm <= 1.000001, 'Actual clipped gradient exceeds original bound')
    for index in range(20):
        require(all(rank_records[r][index]['statistics'] == rank_records[0][index]['statistics']
                    for r in range(8)), 'Global statistics differ across ranks')
    observed = [r['statistics'] for r in rank_records[0]]
    for s in observed:
        require(s['weight_mass'] > 0 and s['valid_token_count'] > 0 and s['rejected_numeric_count'] == 0,
                'Invalid numerical monitor')
        require(s['sampled_old_policy_kl'] is not None and math.isfinite(s['sampled_old_policy_kl'])
                and s['sampled_old_policy_kl'] >= 0, 'Invalid token k3 estimate')
    return dict(status='OBSERVATIONS_ONLY_NOT_PRODUCTION_ACCEPTANCE', guard_count=20,
                max_observed_token_kl=max(s['sampled_old_policy_kl'] for s in observed),
                max_observed_negative_tail_fraction=max(s['negative_tail_fraction_count'] for s in observed),
                candidate_kl_0_01_would_reject=any(s['sampled_old_policy_kl'] > .01 for s in observed),
                target_old_policy_kl=None, negative_tail_limit=None, observations=observed)
