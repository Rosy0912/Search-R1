"""New fresh scope/budget counterexamples only; fixtures prove no GPU acceptance."""
import copy
import pytest
from calibration_contract_v1 import SCOPE, initial_fields, validate_binding


def fixture_workers():
    return [dict(rank=i, pid=1000+i, start_ticks=str(42+i), scheduler_step=0, learning_rates=[0.0],
        **{key:'a'*64 for key in ('model_shards_sha256','optimizer_tensors_sha256','rng_tensors_sha256',
              'local_shards_sha256','buffers_sha256','complete_state_sha256')}) for i in range(8)]


def fixture():
    root = '/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct200-safety-20261006/tools/guard-v2'
    guards = {key:dict(path=root+'/'+filename, sha256='a'*64) for key, filename in
        (('resource_guard','resource_guard_v2.py'), ('resource_ownership','resource_ownership_v2.py'))}
    return dict(**initial_fields(), **guards, status='REVIEWED_FOR_CALIBRATION_ONLY',
        validation_scope=SCOPE, calibration_scope=SCOPE, allowed_safety_modes=['calibration'],
        code_commit='b'*40, source_commit='b'*40, actor_token_identity='sampled_action_prefix_v1',
        capture_actual_runner_inputs=True, capture_actual_normalizer_inputs=True,
        calibration_phase='probability_probe_then_wait', calibration_actual_outer_budget=2,
        optimizer_steps_per_outer=20, gradient_micro_per_outer_per_rank=160,
        retained_training_updates_contributed=0, source_native_acceptance=None,
        candidate_old_policy_kl=.01, target_old_policy_kl=None, negative_tail_limit=None,
        debug_evaluation_rows=512, extra_stage64_evaluation=False,
        prefix_mismatch_limit=0, action_token_mismatch_limit=0,
        actor_recompute_abs_tolerance=None, actor_vllm_abs_tolerance=None,
        actor_recompute_joint_abs_tolerance=None, actor_vllm_joint_abs_tolerance=None,
        probability_tolerance_evidence=[], probability_review_wait_seconds=60,
        files=[dict(value, bytes=1) for value in guards.values()])


def test_fresh_scope_accepts_only_diagnostic_two_with_no_native_claim():
    value=fixture(); validate_binding(value,'b'*40)
    assert value['restore_checkpoint'] is None and value['source_native_acceptance'] is None


@pytest.mark.parametrize('key,value', [
    ('restore_checkpoint','old25'),('initial_adam_steps',500),('initial_outer_step',False),
    ('ppo_mini_batch_size',256),('optimizer_steps_per_outer',10),('calibration_actual_outer_budget',5),
    ('reference_kl_coefficient',.001),('debug_evaluation_rows',64),('extra_stage64_evaluation',True),
    ('source_native_acceptance',{'status':'PASS'}),('actor_vllm_abs_tolerance',2.84),
    ('calibration_scope','GIGPO_POLICY_SAFETY_RAW_ACTION_HEALTHY25_CALIBRATION_V1')])
def test_superseded_recipe_and_borrowed_native_or_tolerance_refused(key,value):
    bound=fixture(); bound[key]=value
    with pytest.raises(ValueError): validate_binding(bound,'b'*40)
