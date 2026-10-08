"""Admission counterexamples, not GPU/native/calibration evidence."""
import copy
import math
import pytest
from calibration_contract_v1 import *
from calibration_state_v1 import state_digest, assert_complete_pause_unchanged


def fixture_workers():
    return [dict(rank=rank, pid=1000 + rank, start_ticks=str(2000 + rank), scheduler_step=25,
        learning_rates=[1e-6 * 25 / 57], model_shards_sha256='a' * 64,
        optimizer_tensors_sha256='b' * 64, rng_tensors_sha256='c' * 64,
        local_shards_sha256='d' * 64, buffers_sha256='e' * 64,
        complete_state_sha256='f' * 64) for rank in range(8)]


def fixture_binding():
    source = 'f' * 40
    guard = dict(resource_guard=dict(path='/native-v2/resource_guard_v2.py', sha256='a' * 64),
                 resource_ownership=dict(path='/native-v2/resource_ownership_v2.py', sha256='b' * 64))
    return dict(**guard, files=[copy.deepcopy(item) for item in guard.values()],
        status='REVIEWED_FOR_NATIVE_ONLY', validation_scope=NATIVE_SCOPE,
        calibration_scope=SCOPE, allowed_safety_modes=['calibration'], code_commit=source,
        restore_step=25, restore_checkpoint=CHECKPOINT, restore_manifest_sha256=MANIFEST,
        common_manifest_sha256=DATA, calibration_actual_outer_budget=1,
        calibration_phase='one_outer',
        retained_training_updates_contributed=0, optimizer_steps_per_outer=10,
        source_native_acceptance=dict(status='PASS', code_commit=source, guard_identity=copy.deepcopy(guard)),
        target_old_policy_kl=None, negative_tail_limit=None, prefix_mismatch_limit=0,
        healthy_baseline_f1_floor=.3231979640745351 / 2,
        action_token_mismatch_limit=0, actor_recompute_abs_tolerance=.0001,
        actor_vllm_abs_tolerance=.005, actor_recompute_joint_abs_tolerance=.001,
        actor_vllm_joint_abs_tolerance=.01, probability_tolerance_evidence=[dict(scope='UNIT_FIXTURE_ONLY')])


def fixture_probability():
    return [dict(rank=i, serial=0, source_commit='f' * 40,
        comparison_scope='SAME_PREFIX_SAME_TOKENS', temperature=1, top_p=1, top_k=-1, n=1,
        effective_probability_scope='RETAINED_ACTOR_ACTION_PREFIX_EXCLUDES_DISCARDED_AND_PADDED_ROWS',
        valid_tokens=10, nonfinite_effective_count=0, actor_old_max_abs_delta=0.,
        actor_vllm_max_abs_delta=.001, repeated_probability_probe_rng_unchanged=True,
        repeated_probability_probe_complete_state_unchanged=True,
        repeated_probe_identity_before=fixture_workers()[i], repeated_probe_identity_after=fixture_workers()[i])
        for i in range(8)]


def fixture_guards():
    stats = dict(weight_mass=256., valid_token_count=1024, rejected_numeric_count=0,
                 sampled_old_policy_kl=.001, negative_tail_fraction_count=.01)
    return [[dict(rank=rank, execution_mode='calibration', admission_scope='numeric_only',
                  actual_optimizer_writes=write, phase=phase, allowed=True,
                  monitor_identity_changed=False, fixed_monitor_hashes={'old': str(rank)},
                  gradient_norm_after_clip=.1,
                  statistics=copy.deepcopy(stats))
             for n in range(10) for write, phase in ((n, 'before'), (n + 1, 'after'))] for rank in range(8)]


def test_approved_isolated_fixture_and_invalid_binding_modes():
    binding = fixture_binding()
    validate_binding(binding, 'f' * 40)
    for key, value in [('restore_step', 80), ('restore_manifest_sha256', 'x' * 64),
                       ('allowed_safety_modes', ['production']), ('target_old_policy_kl', .01),
                       ('negative_tail_limit', .1), ('retained_training_updates_contributed', 1),
                       ('probability_tolerance_evidence', []), ('optimizer_steps_per_outer', 20),
                       ('prefix_mismatch_limit', 1), ('calibration_actual_outer_budget', 300)]:
        bad = dict(binding, **{key: value})
        with pytest.raises(ValueError):
            validate_binding(bad, 'f' * 40)


def test_guard_spawn_identity_requires_actual_accepted_native_same_path_and_bytes():
    binding = fixture_binding()
    native = binding['source_native_acceptance']['guard_identity']
    assert validate_guard_binding(binding, native) == native
    for key, value in [('resource_guard', dict(path='/old/resource_guard.py', sha256='a' * 64)),
                       ('resource_guard', dict(path='/other/resource_guard_v2.py', sha256='a' * 64)),
                       ('resource_ownership', dict(path='/native-v2/resource_ownership_v2.py', sha256='c' * 64)),
                       ('resource_guard', dict(path='resource_guard_v2.py', sha256='a' * 64))]:
        bad = copy.deepcopy(binding)
        bad[key] = value
        with pytest.raises(ValueError):
            validate_guard_binding(bad, native)
    with pytest.raises(ValueError, match='actual full native'):
        validate_guard_binding(binding, dict(native, resource_guard=dict(path='/native-v2/resource_guard_v2.py', sha256='d' * 64)))
    for files in (binding['files'][:1], binding['files'] + binding['files'][:1],
                  [dict(binding['files'][0], sha256='d' * 64), binding['files'][1]]):
        with pytest.raises(ValueError, match='unique full binding'):
            validate_guard_binding(dict(binding, files=files), native)


def test_probability_probe_cannot_update_or_invent_tolerances():
    binding = dict(fixture_binding(), calibration_phase='probability_probe_only',
                   calibration_actual_outer_budget=0, actor_recompute_abs_tolerance=None,
                   actor_vllm_abs_tolerance=None, actor_recompute_joint_abs_tolerance=None,
                   actor_vllm_joint_abs_tolerance=None, probability_tolerance_evidence=[])
    validate_binding(binding, 'f' * 40)
    for key, value in [('calibration_actual_outer_budget', 1), ('actor_vllm_abs_tolerance', .01),
                       ('probability_tolerance_evidence', ['fake'])]:
        with pytest.raises(ValueError):
            validate_binding(dict(binding, **{key: value}), 'f' * 40)


def test_same_process_wait_needs_bound_probability_review_and_does_not_set_kl():
    binding = dict(fixture_binding(), calibration_phase='probability_probe_then_wait',
                   actor_recompute_abs_tolerance=None, actor_vllm_abs_tolerance=None,
                   actor_recompute_joint_abs_tolerance=None, actor_vllm_joint_abs_tolerance=None,
                   probability_tolerance_evidence=[], probability_review_wait_seconds=3600)
    validate_binding(binding, 'f' * 40)
    workers = fixture_workers()
    identity = dict(code_commit='f' * 40, job_id='999', run_id='test-calibration',
                    material_manifest_sha256='a' * 64, native_binding_sha256='b' * 64,
                    post_sampling_worker_identities=workers, post_sampling_worker_pause_sha256=state_digest(workers),
                    post_sampling_driver_complete_state_sha256='d' * 64, post_sampling_pause_raw_sha256='e' * 64)
    approval = dict(identity, status='PASS', scope='GIGPO_HEALTHY25_PROBABILITY_IDENTITY',
        restore_manifest_sha256=MANIFEST, common_manifest_sha256=DATA, rank_count=8,
        sampled_trajectory_count=2560, prefix_mismatch_count=0, action_token_mismatch_count=0,
        same_position_ids_and_mask=True, actor_recompute_abs_tolerance=.0001,
        actor_vllm_abs_tolerance=.005, production_thresholds_selected=False,
        actor_recompute_joint_abs_tolerance=.001, actor_vllm_joint_abs_tolerance=.01,
        target_old_policy_kl=None, negative_tail_limit=None)
    result = validate_probability_approval(approval, identity, binding)
    assert result['target_old_policy_kl'] is None and result['negative_tail_limit'] is None
    for key, value in [('job_id', '998'), ('material_manifest_sha256', 'c' * 64),
                       ('prefix_mismatch_count', 1), ('target_old_policy_kl', .01),
                       ('production_thresholds_selected', True), ('same_position_ids_and_mask', False),
                       ('post_sampling_driver_complete_state_sha256', 'a' * 64),
                       ('post_sampling_pause_raw_sha256', 'a' * 64),
                       ('post_sampling_worker_pause_sha256', 'a' * 64),
                       ('post_sampling_worker_identities', fixture_workers()[:7])]:
        with pytest.raises(ValueError):
            validate_probability_approval(dict(approval, **{key: value}), identity, binding)


def test_small_token_errors_cannot_hide_large_complete_joint_action_error():
    row = dict(effective_tokens=500, same_position_ids=True, same_effective_mask=True,
        engine_prefix_matches_generation=True, full_trajectory_vs_sample_prefix_old_max_abs=0.,
        full_vs_prefix_old_joint_sum_delta=0., full_vs_vllm_joint_sum_delta=1.,
        prefix_old_vs_vllm_joint_sum_delta=1.)
    with pytest.raises(ValueError, match='complete joint'):
        validate_action_probability_checks([row], fixture_binding())
    row['full_vs_vllm_joint_sum_delta'] = row['prefix_old_vs_vllm_joint_sum_delta'] = .001
    validate_action_probability_checks([row], fixture_binding())


def test_full_prefix_cap_difference_is_refused_not_absorbed_into_kl():
    records = [dict(source=0, actions=[dict(generation_prefix=[4, 5, 6], scoring_prefix=[1, 2, 3, 4, 5, 6],
                                          sampled_tokens=[7, 8, 9], retained_action_tokens=[7, 8])])]
    value = validate_alignment(records, 1)
    assert value['prefix_mismatch_count'] == 1
    with pytest.raises(ValueError, match='Generation prefix differs'):
        admit_alignment(value)


def test_retokenized_action_mismatch_refused_even_when_text_equal():
    records = [dict(source=0, actions=[dict(generation_prefix=[1, 2], scoring_prefix=[1, 2],
                                          sampled_tokens=[7, 8, 9], retained_action_tokens=[78, 9])])]
    with pytest.raises(ValueError, match='Retokenized'):
        admit_alignment(validate_alignment(records, 1))


def test_same_prefix_retained_action_prefix_and_empty_effective_action():
    records = [dict(source=0, actions=[dict(generation_prefix=[1], scoring_prefix=[1],
                                          sampled_tokens=[2, 3, 4], retained_action_tokens=[2, 3]),
                                     dict(generation_prefix=[1], scoring_prefix=[1],
                                          sampled_tokens=[], retained_action_tokens=[])])]
    value = validate_alignment(records, 1)
    value.update(position_ids_mismatch_count=0, effective_mask_mismatch_count=0, engine_prefix_mismatch_count=0)
    admit_alignment(value)
    assert value['recorded_actions'] == 2
    with pytest.raises(ValueError, match='Missing actual rollout'):
        validate_alignment(records, 2560)


def test_missing_position_and_mask_proof_cannot_default_to_identity():
    value = dict(prefix_mismatch_count=0, action_token_mismatch_count=0)
    with pytest.raises(ValueError, match='Missing/different'):
        admit_alignment(value)
    value.update(position_ids_mismatch_count=0, effective_mask_mismatch_count=1, engine_prefix_mismatch_count=0)
    with pytest.raises(ValueError, match='Missing/different'):
        admit_alignment(value)


def test_probability_source_sampling_distribution_and_probe_rng_counterexamples():
    binding = fixture_binding()
    records = fixture_probability()
    validate_probability_receipts(records, binding)
    for key, value in [('source_commit', 'e' * 40), ('temperature', .8), ('n', 5),
                       ('actor_old_max_abs_delta', .001), ('actor_vllm_max_abs_delta', .006),
                       ('actor_vllm_max_abs_delta', math.nan), ('nonfinite_effective_count', 1),
                       ('repeated_probability_probe_rng_unchanged', False),
                       ('repeated_probability_probe_complete_state_unchanged', False),
                       ('repeated_probe_identity_after', dict(fixture_workers()[4], model_shards_sha256='0' * 64))]:
        bad = copy.deepcopy(records)
        bad[4][key] = value
        with pytest.raises(ValueError):
            validate_probability_receipts(bad, binding)


def test_observations_do_not_select_production_thresholds():
    guards = fixture_guards()
    value = summarize_guards(guards)
    assert value['status'] == 'OBSERVATIONS_ONLY_NOT_PRODUCTION_ACCEPTANCE'
    assert value['target_old_policy_kl'] is value['negative_tail_limit'] is None
    assert not value['candidate_kl_0_01_would_reject']
    for rows in guards:
        rows[-1]['statistics']['sampled_old_policy_kl'] = .02
    value = summarize_guards(guards)
    assert value['candidate_kl_0_01_would_reject']
    assert value['target_old_policy_kl'] is None


def test_actual_writes_distributed_equal_and_finite_guard_counterexamples():
    for mutation in ('missing_rank', 'missing_post', 'changed_old', 'changed_stats', 'wrong_scope', 'nan_kl'):
        records = fixture_guards()
        if mutation == 'missing_rank':
            records.pop()
        elif mutation == 'missing_post':
            records[4].pop()
        elif mutation == 'changed_old':
            records[1][1]['fixed_monitor_hashes'] = {'old': 'changed'}
        elif mutation == 'changed_stats':
            records[3][2]['statistics']['sampled_old_policy_kl'] = .2
        elif mutation == 'wrong_scope':
            records[1][1]['admission_scope'] = 'production_calibrated'
        else:
            for rows in records:
                rows[1]['statistics']['sampled_old_policy_kl'] = math.nan
        with pytest.raises(ValueError):
            summarize_guards(records)
