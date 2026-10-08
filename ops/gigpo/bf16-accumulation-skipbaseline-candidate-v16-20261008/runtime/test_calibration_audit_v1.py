"""Actual-rollout local-empty counterexamples; no GPU result is claimed."""
import copy
import pytest
from calibration_audit_v1 import validate_calibration_update, validate_calibration_numeric


def empty_micro():
    return dict(current_log_probs=dict(count=0, finite=0), joint_action_logratios=dict(count=0, finite=0),
        joint_action_ratios=dict(count=0), dual_clip=3.,
        ratio_scope='Dual-Clip: raw_joint_exp_not_computed; complete_joint_logratios_only',
        policy_loss=dict(count=1, finite=1, minimum=0., maximum=0.))


def fixture():
    return dict(kind='actor_outer', outer_step=1, rank=0, commit='f' * 40, config_sha256='a' * 64,
        run_id='unit-fixture', attempt_id=1, counted_complete=True, opportunities_processed=20,
        actual_gradient_micro_batches=160, rows=320, epochs=1, optimizer_steps=20, skipped_empty_minibatches=0,
        micro_stats=[empty_micro() for _ in range(160)], grad_finite=20, grad_nonzero=0,
        grad_norms=[0.] * 20, fixed_unchanged=True, shard_layout_unchanged=True,
        row_denominators=[dict(local_rows=16, actual_denominator=16, world_size=8) for _ in range(20)])


def receipt():
    return dict(attempt_id=1, counted_complete=True, opportunities_processed=20,
        actual_successful_adam_writes=20, skipped_opportunities=0, gradient_micro_batches=160, refusals=[])


def test_empty_local_micros_keep_zero_loss_and_fixed_global_budget():
    value = validate_calibration_update(fixture(), 'f' * 40, 'a' * 64, 'unit-fixture', 0, 1, receipt())
    assert value == dict(gradient_micro_batches=160, observed_numeric_micro_batches=160,
        legal_empty_local_micro_batches=160, actual_adam_writes=20, opportunities_processed=20, counted_complete=True)


def test_empty_micro_cannot_fabricate_gradient_objective_or_probability_mass():
    for key in ('nonzero_loss', 'fabricated_action', 'skipped_mini', 'extra_adam', 'micro_128_denominator'):
        row = fixture()
        if key == 'nonzero_loss':
            row['micro_stats'][3]['policy_loss']['minimum'] = row['micro_stats'][3]['policy_loss']['maximum'] = .1
        elif key == 'fabricated_action':
            row['micro_stats'][3]['joint_action_logratios'] = dict(count=1, finite=1)
        elif key == 'skipped_mini':
            row['skipped_empty_minibatches'] = 1
        elif key == 'extra_adam':
            row['optimizer_steps'] = 11
        else:
            row['row_denominators'][5]['actual_denominator'] = 128
        with pytest.raises(ValueError):
            validate_calibration_update(row, 'f' * 40, 'a' * 64, 'unit-fixture', 0, 1, receipt())


def test_empty_numeric_components_must_be_actual_finite_zero_scalars():
    row = dict(reference_minus_current=dict(count=0, finite=0),
        loss_parts={name: dict(count=1, finite=1, minimum=0., maximum=0.) for name in ('policy', 'entropy', 'kl', 'total')})
    validate_calibration_numeric(row)
    row['loss_parts']['kl']['finite'] = 0
    with pytest.raises(ValueError):
        validate_calibration_numeric(row)
