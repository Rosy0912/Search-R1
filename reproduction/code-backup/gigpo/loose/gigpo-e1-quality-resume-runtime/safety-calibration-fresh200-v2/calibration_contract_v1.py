"""Fresh official two-outer diagnostic authorization; never formal acceptance."""
import math
import re
from pathlib import PurePosixPath

SCOPE = 'GIGPO_POLICY_SAFETY_OFFICIAL_INSTRUCT_FRESH200_CALIBRATION_V1'
PROBABILITY_SCOPE = 'GIGPO_OFFICIAL_INSTRUCT_FRESH200_RAW_ACTION_PROBABILITY_IDENTITY_V1'
DATA = '0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
REVISION = 'aa8e72537993ba99e69dfaafa59ed015b17504d1'


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def initial_fields():
    return dict(origin_kind='OFFICIAL_MODEL_INITIALIZATION', official_revision=REVISION,
        common_manifest_sha256=DATA, initial_outer_step=0, initial_adam_steps=0,
        initial_scheduler_step=0, initial_data_cursor=0, restore_checkpoint=None,
        restore_manifest_sha256=None, ppo_mini_batch_size=128, expected_adam_per_rank=20,
        expected_gradient_micro_per_rank=160, reference_kl_coefficient=0.0)


def validate_initial_fields(proof):
    for key, value in initial_fields().items():
        require(key in proof and proof[key] == value and type(proof[key]) is type(value),
                'Fresh official initialization differs: ' + key)


def validate_guard_binding(binding):
    """Static independent review and exact physical bytes bind the unchanged guard.

    The real diagnostic job is its own eight-GPU acceptance; it is NOT gated on
    unexecuted/failed native-v2 fixtures or a historical ten-Adam result.
    """
    identity = {}
    for key, filename in (('resource_guard', 'resource_guard_v2.py'),
                          ('resource_ownership', 'resource_ownership_v2.py')):
        item = binding[key]; path = PurePosixPath(item['path'])
        require(path.is_absolute() and path.name == filename, 'Unbound guard path: ' + key)
        require(bool(re.fullmatch(r'[0-9a-f]{64}', item['sha256'])), 'Invalid guard SHA: ' + key)
        files = [record for record in binding['files'] if record['path'] == str(path)]
        require(len(files) == 1 and files[0]['sha256'] == item['sha256'], 'Missing unique guard coverage')
        identity[key] = dict(path=str(path), sha256=item['sha256'])
    require(PurePosixPath(identity['resource_guard']['path']).parent ==
            PurePosixPath(identity['resource_ownership']['path']).parent, 'Guard ownership module directory differs')
    return identity


def validate_binding(binding, actual_source):
    require(binding['status'] == 'REVIEWED_FOR_CALIBRATION_ONLY'
            and binding['validation_scope'] == binding['calibration_scope'] == SCOPE,
            'No isolated fresh calibration independent review')
    require(binding['allowed_safety_modes'] == ['calibration'], 'Wrong execution mode')
    require(binding['code_commit'] == binding['source_commit'] == actual_source
            and bool(re.fullmatch(r'[0-9a-f]{40}', actual_source)), 'Wrong frozen source')
    validate_initial_fields(binding)
    require(binding['actor_token_identity'] == 'sampled_action_prefix_v1'
            and binding['capture_actual_runner_inputs'] is True
            and binding['capture_actual_normalizer_inputs'] is True,
            'Missing decisive actual probability observations')
    require(binding['calibration_phase'] == 'probability_probe_then_wait'
            and binding['calibration_actual_outer_budget'] == 2
            and binding['optimizer_steps_per_outer'] == 20
            and binding['gradient_micro_per_outer_per_rank'] == 160
            and binding['retained_training_updates_contributed'] == 0, 'Wrong two-outer diagnostic budget')
    require(binding.get('source_native_acceptance') is None,
            'Historical/unexecuted native cannot substitute this diagnostic actual GPU acceptance')
    validate_guard_binding(binding)
    require(binding['candidate_old_policy_kl'] == .01
            and binding['target_old_policy_kl'] is None and binding['negative_tail_limit'] is None,
            'Candidate is not production threshold acceptance')
    require(binding['debug_evaluation_rows'] == 512 and binding['extra_stage64_evaluation'] is False,
            'Fresh diagnostic requires common512 before/after, no extra64')
    require(binding['prefix_mismatch_limit'] == binding['action_token_mismatch_limit'] == 0,
            'Structural identity cannot be relaxed')
    for key in ('actor_recompute_abs_tolerance', 'actor_vllm_abs_tolerance',
                'actor_recompute_joint_abs_tolerance', 'actor_vllm_joint_abs_tolerance'):
        require(binding[key] is None, 'Unknown actual tolerance cannot be pre-filled')
    require(binding['probability_tolerance_evidence'] == [], 'No borrowed historical probability acceptance')
    require(type(binding['probability_review_wait_seconds']) is int
            and 0 < binding['probability_review_wait_seconds'] <= 43200, 'Invalid bounded review wait')


def validate_probability_approval(approval, identity, binding):
    require(approval['status'] == 'PASS' and approval['scope'] == PROBABILITY_SCOPE,
            'Missing separate fresh probability acceptance')
    validate_initial_fields(approval)
    require(approval.get('raw_alignment_schema') == 'RAW_ACTION_PREFIX_ALIGNMENT_V1'
            and approval.get('actual_complete_hf_inputs_observed') is True
            and approval.get('actual_hf_roles') == ['old', 'current', 'reference']
            and approval.get('actual_vllm_runner_positions_observed') is True
            and approval.get('actual_normalizer_inputs_observed') is True
            and approval.get('actual_adam_writes') == 0, 'Probability identity precedes every Adam write')
    for key in ('code_commit', 'job_id', 'run_id', 'material_manifest_sha256', 'native_binding_sha256',
                'post_sampling_worker_identities', 'post_sampling_worker_pause_sha256',
                'post_sampling_driver_complete_state_sha256', 'post_sampling_pause_raw_sha256',
                'decisive_normalizer_evidence_sha256'):
        require(approval[key] == identity[key], 'Different probability evidence: ' + key)
    from calibration_state_v1 import validate_pause_workers, state_digest
    workers = validate_pause_workers(identity['post_sampling_worker_identities'])
    require(state_digest(workers) == identity['post_sampling_worker_pause_sha256'], 'Incomplete eight-rank pause')
    require(approval['rank_count'] == 8 and approval['sampled_trajectory_count'] == 2560
            and approval['prefix_mismatch_count'] == approval['action_token_mismatch_count'] == 0
            and approval['same_position_ids_and_mask'] is True, 'Missing structural identity')
    admitted = dict(binding)
    for key in ('actor_recompute_abs_tolerance', 'actor_vllm_abs_tolerance',
                'actor_recompute_joint_abs_tolerance', 'actor_vllm_joint_abs_tolerance'):
        value = approval[key]
        require(type(value) in (float, int) and math.isfinite(value) and value > 0,
                'Missing independently reviewed actual numerical tolerance')
        admitted[key] = value
    require(approval['production_thresholds_selected'] is False
            and approval['target_old_policy_kl'] is None and approval['negative_tail_limit'] is None,
            'Probability acceptance cannot select production thresholds')
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
            require(len(sampled) >= len(ids), 'Raw action longer than the actual completion token sequence')
    return dict(trajectory_rows=len(records), recorded_actions=actions,
                prefix_mismatch_count=prefix_bad, action_token_mismatch_count=token_bad)


def admit_alignment(alignment):
    require(alignment['prefix_mismatch_count'] == 0,
            'Generation prefix differs from the actual independently forwarded HF action prefix; preserve and investigate')
    require(alignment['action_token_mismatch_count'] == 0,
            'Raw actor action differs from actual sampled token support; preserve and investigate')
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
        require(row.get('actual_complete_hf_inputs_observed') is True, 'Complete raw HF action input not observed')
        values = (row['full_trajectory_vs_sample_prefix_old_max_abs'], row['full_vs_prefix_old_joint_sum_delta'],
                  row['full_vs_vllm_joint_sum_delta'], row['prefix_old_vs_vllm_joint_sum_delta'])
        require(all(math.isfinite(v) for v in values), 'Nonfinite complete action probability difference')
        require(row['full_trajectory_vs_sample_prefix_old_max_abs'] <= binding['actor_recompute_abs_tolerance'],
                'Independent full-action HF token probability differs from per-turn HF prefix')
        require(abs(row['full_vs_prefix_old_joint_sum_delta']) <= binding['actor_recompute_joint_abs_tolerance'],
                'Independent full-action HF joint probability differs from per-turn HF prefix')
        require(row['complete_current_vs_old_max_abs'] <= binding['actor_recompute_abs_tolerance']
                and abs(row['complete_current_vs_old_joint_sum_delta']) <= binding['actor_recompute_joint_abs_tolerance'],
                'Same-state complete current/old probabilities differ')
        require(abs(row['full_vs_vllm_joint_sum_delta']) <= binding['actor_vllm_joint_abs_tolerance']
                and abs(row['prefix_old_vs_vllm_joint_sum_delta']) <= binding['actor_vllm_joint_abs_tolerance'],
                'Actual sampler versus scored complete joint action probability differs')


def summarize_guards(rank_records):
    """Both actual outers; fixed monitor is identical WITHIN each rollout reuse."""
    require(len(rank_records) == 8, 'Missing eight-rank monitor logs')
    for rank, rows in enumerate(rank_records):
        require(len(rows) == 80, 'Two outers require 40 Adam writes, each before/after')
        for outer in (1, 2):
            part = [r for r in rows if r['outer_step'] == outer]
            require(len(part) == 40 and [r['actual_optimizer_writes'] for r in part] ==
                    [v for n in range(20) for v in (n, n+1)], 'Incomplete real per-outer Adam accounting')
            require([r['phase'] for r in part] == [v for _ in range(20) for v in ('before', 'after')],
                    'Wrong monitor placement')
            require(all(r['fixed_monitor_hashes'] == part[0]['fixed_monitor_hashes'] for r in part),
                    'Monitor old policy changed within an outer')
        require(all(r['rank'] == rank and r['execution_mode'] == 'calibration'
                    and r['admission_scope'] == 'calibration_candidate_token_kl_and_numeric_only'
                    and r['allowed'] and not r['monitor_identity_changed'] for r in rows), 'Invalid guard admission')
        require(all(math.isfinite(r['gradient_norm_after_clip']) and 0 <= r['gradient_norm_after_clip'] <= 1.000001
                    for r in rows if r['phase'] == 'after'), 'Original gradient clipping differs')
    for index in range(80):
        require(all(rank_records[r][index]['statistics'] == rank_records[0][index]['statistics']
                    for r in range(8)), 'Global statistics differ across ranks')
    observed = [r['statistics'] for r in rank_records[0]]
    for s in observed:
        require(s['weight_mass'] > 0 and s['valid_token_count'] > 0 and s['rejected_numeric_count'] == 0,
                'Invalid numerical monitor')
        require(s['sampled_old_policy_kl'] is not None and math.isfinite(s['sampled_old_policy_kl'])
                and 0 <= s['sampled_old_policy_kl'] <= .01, 'Candidate known KL exceeded')
    return dict(status='OBSERVATIONS_ONLY_NOT_PRODUCTION_ACCEPTANCE', guard_count=80,
        actual_optimizer_steps_per_rank=40, actual_gradient_micro_per_rank=320,
        max_observed_token_kl=max(s['sampled_old_policy_kl'] for s in observed),
        max_observed_negative_tail_fraction=max(s['negative_tail_fraction_count'] for s in observed),
        candidate_old_policy_kl=.01, target_old_policy_kl=None, negative_tail_limit=None, observations=observed)
