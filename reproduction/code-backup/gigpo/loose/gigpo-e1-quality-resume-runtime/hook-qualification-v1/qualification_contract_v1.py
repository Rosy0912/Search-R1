"""Read-only gate for a future isolated external-hook A/B; no launch/threshold choice."""
import hashlib
import json
import math
from pathlib import Path
import re

SCOPE = 'GIGPO_EXTERNAL_PROBABILITY_HOOK_REAL_EIGHT_GPU'
RESTORE_SHA = '3a073aba77147db7d581cd514ee5c4ae34965e5ff9c163716bd25b47aeb478e4'
DATA_SHA = '0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'


def require(value, reason):
    if not value: raise ValueError(reason)


def record(path):
    path = Path(path)
    require(path.is_absolute() and path.is_file() and not path.is_symlink(), 'Missing actual qualification original')
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def read_bound(row):
    require(isinstance(row, dict), 'Unaccepted/null qualification prerequisite')
    require(record(row['path']) == row, 'Qualification original bytes/SHA differ')
    return json.loads(Path(row['path']).read_text())


def admit(binding):
    # The local null template must fail before importing Ray/Torch or touching CUDA.
    require(binding.get('status') == 'REVIEWED_FOR_EXTERNAL_HOOK_ONLY'
            and binding.get('scope') == SCOPE, 'DRAFT cannot initialize external-hook GPU qualification')
    source = binding['actual_source']
    require(re.fullmatch('[0-9a-f]{40}', str(source)) and binding['node'] == 'h100-8'
            and binding['restore_manifest_sha256'] == RESTORE_SHA
            and binding['common_manifest_sha256'] == DATA_SHA, 'Wrong source/restore/data/node')
    require(binding['diagnostic_budget'] == dict(A_outer=1, B_outer=1, Adam_per_outer=10,
            total_diagnostic_Adam=20, counter_Adam=0, retained_goal_contribution=0),
            'External qualification cannot contribute to formal300 or conceal additional writes')
    for name in ('files', 'evidence'):
        rows = binding[name]
        require(rows and len({row['path'] for row in rows}) == len(rows), 'Missing/duplicate physical qualification closure')
        for row in rows: require(record(row['path']) == row, 'Physical qualification closure differs')
    native = read_bound(binding['native_acceptance'])
    probability = read_bound(binding['probability_review'])
    position = read_bound(binding['position_semantics_review'])
    # Native uses the real accepted producer schema (code_commit, not a
    # synthesized source_commit view). This imported immutable adapter closes
    # the raw result, terminal, physical payload, guard and independent review.
    from formal_protection_contract_v1 import validate_actual_native
    validate_actual_native(native, source, binding['guard_identity'])
    for proof in (probability, position):
        require(proof['status'] == 'PASS' and proof['source_commit'] == source,
                'Existing source52 acceptance cannot authorize a changed compatible source')
    require(probability['actual_trajectories'] == 2560
            and probability['prefix_mismatch_count'] == probability['token_mismatch_count'] == 0
            and probability['hf_position_and_mask_identity'] is True
            and probability['post_sampling_live_pause_verified'] is True,
            'Need actual same-source sampler identity before qualification Adam')
    expected_keys = ('actor_recompute_abs_tolerance','actor_vllm_abs_tolerance',
                     'actor_recompute_joint_abs_tolerance','actor_vllm_joint_abs_tolerance')
    tolerances = binding['probability_tolerances']
    require(isinstance(tolerances,dict) and set(tolerances) == set(expected_keys)
            and tolerances == probability['probability_tolerances']
            and all(type(v) in (int,float) and math.isfinite(v) and v >= 0 for v in tolerances.values()),
            'No actual accepted token/joint probability tolerances; never borrow PPO KL limits')
    require(position['sampler_runner_positions_observed'] is False
            and position['actual_runner_model_config_and_active_branch_verified'] is True
            and position['actual_engine_literal_prefix_observed'] is True,
            'Exact installed backend semantics with actual active branch still required')
    isolation = read_bound(binding['isolated_authorization'])
    require(isolation['status'] == 'REVIEWED_FOR_NATIVE_ONLY'
            and isolation['validation_scope'] == 'GIGPO_POLICY_SAFETY_NATIVE_ONLY'
            and isolation['code_commit'] == source and isolation['restore_step'] == 25
            and isolation['restore_manifest_sha256'] == RESTORE_SHA
            and 'calibration' in isolation['allowed_safety_modes'], 'Source actor must be independently authorized in numeric-only scope')
    require(binding['policy_mode'] == 'calibration' and binding['production_thresholds'] is None
            and binding['production_artifact'] is None, 'Qualification cannot invent production thresholds/artifacts')
    return probability


def assert_same_state(before, after, context):
    require(before == after, context + ' changed real full state; exception alone does not prove zero Adam')


def assert_counter_result(before, after, acknowledgements):
    assert_same_state(before, after, 'Rejected counterexample')
    require(len(acknowledgements) == 8 and {row['rank'] for row in acknowledgements} == set(range(8))
            and all(row['matched_collective_rejected'] is True and row['actual_optimizer_steps_delta'] == 0
                    and row['actual_gradient_forward_delta'] == 0 for row in acknowledgements),
            'No physical matched all-rank zero-Adam/zero-forward counterexample')


def assert_AB(A, B):
    keys = ('before_worker_state','before_driver_state','actual_questions_and_groups', 'generated_tensors',
            'balanced_scored_batch', 'old_and_ref_probabilities', 'gradient_micro_inputs_and_current',
            'fixed_global256_monitor_inputs_and_current','loss_and_gradient_records',
            'after_worker_state','after_driver_state','saved_reloaded_worker_state','saved_reloaded_driver_state')
    for key in keys: assert_same_state(A[key], B[key], 'A/B ' + key)
    require(A['actual_Adam_per_rank'] == B['actual_Adam_per_rank'] == 10
            and A['gradient_micro_per_rank'] == B['gradient_micro_per_rank'] == 160
            and A['retained_goal_contribution'] == B['retained_goal_contribution'] == 0,
            'Actual A/B diagnostic optimizer budget differs')
    return dict(status='AB_PRODUCER_CHECKS_ONLY_NOT_INDEPENDENT_ACCEPTANCE',scope=SCOPE,
                retained_goal_contribution=0,actual_diagnostic_Adam_per_rank=20,
                exact_keys_checked=list(keys))
