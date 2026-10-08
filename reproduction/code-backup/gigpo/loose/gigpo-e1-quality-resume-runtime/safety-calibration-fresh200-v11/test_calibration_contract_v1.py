"""New fresh scope/budget counterexamples only; fixtures prove no GPU acceptance."""
import copy
import pytest
from calibration_contract_v1 import SCOPE, LIVE_SCOPE, initial_fields, validate_binding
from guard_uuid_input_v5 import (ACTUAL_GUARD_UUID_PATH, ACTUAL_GUARD_UUID_SHA256,
    ACTUAL_GUARD_UUID_BYTES, ACTUAL_GUARD_UUID_SCHEMA, GUARD_SHA256, OWNERSHIP_SHA256)
from test_guard_uuid_input_v5 import canonical_uuids


def fixture_workers():
    return [dict(rank=i, pid=1000+i, start_ticks=str(42+i), scheduler_step=0, learning_rates=[0.0],
        **{key:'a'*64 for key in ('model_shards_sha256','optimizer_tensors_sha256','rng_tensors_sha256',
              'local_shards_sha256','buffers_sha256','complete_state_sha256')}) for i in range(8)]


def fixture():
    root = '/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct200-safety-20261006/tools/guard-v2'
    guards = {key:dict(path=root+'/'+filename, sha256=GUARD_SHA256 if key == 'resource_guard' else OWNERSHIP_SHA256) for key, filename in
        (('resource_guard','resource_guard_v2.py'), ('resource_ownership','resource_ownership_v2.py'))}
    uuid_input = dict(path=ACTUAL_GUARD_UUID_PATH, sha256=ACTUAL_GUARD_UUID_SHA256,
        bytes=ACTUAL_GUARD_UUID_BYTES)
    return dict(initial_fields(), **guards, canonical_uuid_inventory_record=uuid_input, expected_gpu_uuid_inventory=canonical_uuids(), status='REVIEWED_FOR_FRESH200_LIVE_TRAINING_ONLY',
        validation_scope=LIVE_SCOPE, calibration_scope=SCOPE, allowed_safety_modes=['calibration', 'production'],
        live_goal_updated_at=1791295951, live_authorization_goal_updated_at=1791293201, official_training_outer_target=200,
        initial_calibration_counted_outer_limit=5, first_full_acceptance_step=5,
        first5_count_toward_goal=True, restart_after_calibration=False, production_tail_required_before_outer6=True,
        first5_known_token_kl_enforced=True, first5_tail_observation_only=True,
        outer_counting_authorization_source='human', outer_counting_human_goal_updated_at=1791283983,
        actual_gpu_acceptance=None, actual_probability_acceptance=None,
        code_commit='b'*40, source_commit='b'*40, actor_token_identity='sampled_action_prefix_v1',
        capture_actual_runner_inputs=True, capture_actual_normalizer_inputs=True,
        calibration_phase='probability_probe_then_wait',
        optimizer_opportunities_per_outer=20, diagnostic_total_optimizer_opportunities=100,
        no_skip_adam_per_outer_per_rank=20, no_skip_gradient_micro_per_outer_per_rank=160,
        counters_are_no_skip_reference=True, full_outer_requires_all_opportunities_processed=True,
        full_outer_requires_at_least_one_actual_adam_write=True, actual_write_counts_must_be_reported_separately=True,
        zero_write_or_abandoned_outer_counts=False,
        retained_training_updates_contributed=0, source_native_acceptance=None,
        candidate_old_policy_kl=.2, target_old_policy_kl=.2, token_kl_threshold_source='human', token_kl_human_goal_updated_at=1791282908, negative_tail_limit=None,
        debug_evaluation_rows=512, extra_stage64_evaluation=False,
        prefix_mismatch_limit=0, action_token_mismatch_limit=0,
        actor_recompute_abs_tolerance=None, actor_vllm_abs_tolerance=None,
        actor_recompute_joint_abs_tolerance=None, actor_vllm_joint_abs_tolerance=None,
        probability_tolerance_evidence=[], probability_review_wait_seconds=60,
        files=[dict(value, bytes=1) for value in guards.values()] + [dict(uuid_input)])


def test_fresh_live_scope_requires_retained_five_to200_with_no_native_claim():
    value=fixture(); validate_binding(value,'b'*40)
    assert value['restore_checkpoint'] is None and value['source_native_acceptance'] is None


@pytest.mark.parametrize('key,value', [
    ('restore_checkpoint','old25'),('initial_adam_steps',500),('initial_outer_step',False),
    ('ppo_mini_batch_size',256),('optimizer_opportunities_per_outer',10),('initial_calibration_counted_outer_limit',2),
    ('reference_kl_coefficient',.001),('candidate_old_policy_kl',.01),('target_old_policy_kl',None),('token_kl_threshold_source','calibration'),('debug_evaluation_rows',64),('extra_stage64_evaluation',True),
    ('source_native_acceptance',{'status':'PASS'}),('actor_vllm_abs_tolerance',2.84),
    ('calibration_scope','GIGPO_POLICY_SAFETY_RAW_ACTION_HEALTHY25_CALIBRATION_V1')])
def test_superseded_recipe_and_borrowed_native_or_tolerance_refused(key,value):
    bound=fixture(); bound[key]=value
    with pytest.raises(ValueError): validate_binding(bound,'b'*40)
