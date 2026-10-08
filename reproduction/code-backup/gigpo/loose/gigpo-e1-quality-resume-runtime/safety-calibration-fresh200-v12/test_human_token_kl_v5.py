"""UNEXECUTED .2 human gate fixtures; no claim of optimality or output quality."""
import copy
import math
import pytest
from calibration_contract_v1 import PROBABILITY_SCOPE, SCOPE, initial_fields, validate_probability_approval
from calibration_state_v1 import state_digest
from safety_audit_v1 import validate_guard
from test_calibration_contract_v1 import fixture, fixture_workers


def guard_record(kl):
    stats = dict(coverage='fixed_first_global256_trajectories_complete_actions', quantiles='exact_global_samples',
        weight_mass=1., valid_token_count=1, independent_action_count=1, negative_action_count=0,
        negative_action_weight_mass=0., rejected_numeric_count=0, invalid_count=0, kl_log_bound_exceeded_count=0,
        negative_tail_fraction_count=0., negative_tail_fraction_weight=0., dual_clip_saturated_action_fraction=0.,
        dual_clip_saturated_objective_mass_fraction=0., max_abs_token_delta=.1, action_token_length_p95=1.,
        action_token_length_max=1., negative_action_logratio_p95=None, negative_action_logratio_p99=None,
        negative_action_logratio_max=None, sampled_old_policy_kl=kl)
    return dict(execution_mode='calibration', admission_scope='calibration_candidate_token_kl_and_numeric_only',
        dual_clip=3., kl_log_bound=30., rank=0, pid=1000, commit='b'*40, run_id='CPU-static-fixture',
        temperature=1., fixed_monitor_hashes={'actual_fixture_only':'a'*64}, statistics=stats,
        native_binding_sha256='c'*64, allowed=True, rejection_reason=None,
        candidate_old_policy_kl=.2, target_old_policy_kl=.2, negative_tail_limit=None,
        production_thresholds_accepted=False)


@pytest.mark.parametrize('kl', [0., math.nextafter(.2, -math.inf), .2])
def test_human_kl_below_and_equal_boundary_are_admissible(kl):
    validate_guard(guard_record(kl), 'calibration', 'c'*64)
    from search_r1.baselines.policy_safety import calibrated_or_candidate_rejection
    config = dict(mode='calibration', calibration_scope=SCOPE,
        candidate_old_policy_kl=.2, target_old_policy_kl=.2, negative_tail_limit=None)
    assert calibrated_or_candidate_rejection(guard_record(kl)['statistics'], config) is None


@pytest.mark.parametrize('kl', [math.nextafter(.2, math.inf), .2001, -.001, math.nan, math.inf])
def test_human_kl_above_boundary_or_invalid_allowed_record_refused(kl):
    with pytest.raises(ValueError): validate_guard(guard_record(kl), 'calibration', 'c'*64)
    if math.isfinite(kl) and kl > .2:
        from search_r1.baselines.policy_safety import calibrated_or_candidate_rejection
        config = dict(mode='calibration', calibration_scope=SCOPE,
            candidate_old_policy_kl=.2, target_old_policy_kl=.2, negative_tail_limit=None)
        assert calibrated_or_candidate_rejection(guard_record(kl)['statistics'], config) is not None


def probability_fixture():
    identity = {key:'d'*64 for key in ('code_commit', 'job_id', 'run_id', 'material_manifest_sha256',
        'native_binding_sha256', 'post_sampling_driver_complete_state_sha256',
        'post_sampling_pause_raw_sha256', 'decisive_normalizer_evidence_sha256')}
    identity['post_sampling_worker_identities'] = fixture_workers()
    identity['post_sampling_worker_pause_sha256'] = state_digest(fixture_workers())
    identity['actual_hf_role_micro_batch_sizes'] = {'old': 4, 'current': 2, 'reference': 4}
    identity['actual_dropout_neutral_train_eval_verified'] = True
    approval = dict(initial_fields(), **identity, status='PASS', scope=PROBABILITY_SCOPE,
        raw_alignment_schema='RAW_ACTION_PREFIX_ALIGNMENT_V1', actual_complete_hf_inputs_observed=True,
        actual_hf_roles=['old','current','reference'], actual_vllm_runner_positions_observed=True,
        actual_normalizer_inputs_observed=True, actual_adam_writes=0, rank_count=8,
        sampled_trajectory_count=2560, prefix_mismatch_count=0, action_token_mismatch_count=0,
        same_position_ids_and_mask=True, production_thresholds_selected=False, probability_only=True,
        candidate_old_policy_kl=.2, target_old_policy_kl=.2, token_kl_threshold_source='human',
        token_kl_human_goal_updated_at=1791282908, negative_tail_limit=None,
        actor_recompute_abs_tolerance=1e-6, actor_vllm_abs_tolerance=1e-6,
        actor_recompute_joint_abs_tolerance=1e-6, actor_vllm_joint_abs_tolerance=1e-6)
    return approval, identity, fixture()


def test_probability_approval_retains_human_kl_and_unselected_tail():
    approval, identity, binding = probability_fixture()
    assert all(binding[key] is None for key in ('actor_recompute_abs_tolerance','actor_vllm_abs_tolerance',
        'actor_recompute_joint_abs_tolerance','actor_vllm_joint_abs_tolerance'))
    admitted = validate_probability_approval(approval, identity, binding)
    assert admitted['target_old_policy_kl'] == admitted['candidate_old_policy_kl'] == .2
    assert admitted['token_kl_threshold_source'] == 'human' and admitted['negative_tail_limit'] is None


@pytest.mark.parametrize('key,value', [('target_old_policy_kl',.01),('target_old_policy_kl',None),
    ('negative_tail_limit',.1),('candidate_old_policy_kl',.3),('token_kl_threshold_source','calibration'),
    ('token_kl_human_goal_updated_at',0),('probability_only',False),('production_thresholds_selected',True)])
def test_probability_approval_cannot_select_tail_or_change_human_kl(key,value):
    approval, identity, binding = probability_fixture()
    approval[key] = value
    with pytest.raises(ValueError): validate_probability_approval(approval, identity, binding)
