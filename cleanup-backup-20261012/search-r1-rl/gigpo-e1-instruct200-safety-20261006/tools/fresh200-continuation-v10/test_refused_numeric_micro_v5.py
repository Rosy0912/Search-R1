"""UNEXECUTED CPU diagnostics for a matched loss observation that never backwarded."""
import copy
import pytest
from calibration_audit_v1 import validate_calibration_update, validate_attempt_numeric
from test_calibration_audit_v1 import empty_micro


def refused_loss_fixture():
    rejection = dict(outer_step=1, mini_index=0, phase='gradient_micro0_loss',
        actual_optimizer_writes=0, matched_guard='ORIGINAL_ALL_REDUCE_MAX_LOSS_REFUSAL')
    receipt = dict(rank=0, pid=9000, attempt_id=1, counted_complete=True, opportunities_processed=20,
        actual_successful_adam_writes=19, skipped_opportunities=1, gradient_micro_batches=152,
        refusals=[dict(partial_update=dict(rejection=rejection))])
    failed = empty_micro()
    failed['policy_loss'] = dict(count=1, finite=0, minimum=None, maximum=None)
    micros = [failed] + [empty_micro() for _ in range(152)]
    actor = dict(kind='actor_outer', outer_step=1, rank=0, pid=9000,
        commit='f'*40, config_sha256='a'*64, run_id='CPU-loss-fixture', attempt_id=1,
        counted_complete=True, opportunities_processed=20, actual_gradient_micro_batches=152,
        rows=320, epochs=1, optimizer_steps=19, skipped_empty_minibatches=1,
        micro_stats=micros, grad_finite=19, grad_nonzero=0, grad_norms=[0.] * 19,
        fixed_unchanged=True, shard_layout_unchanged=True,
        row_denominators=[dict(mini_index=mini, local_rows=16, actual_denominator=16, world_size=8)
                          for mini in range(20)])
    coordinates = [(0, 0)] + [(mini, micro) for mini in range(1, 20) for micro in range(8)]
    numeric = []
    for index, (mini, micro) in enumerate(coordinates, 1):
        parts = {name: dict(count=1, finite=1, minimum=0., maximum=0.)
                 for name in ('policy', 'entropy', 'kl', 'total')}
        if index == 1:
            parts['total'] = dict(count=1, finite=0, minimum=None, maximum=None)
        numeric.append(dict(attempt_id=1, rank=0, pid=9000, micro_index=index,
            mini_index=mini, micro_index_in_mini=micro, commit='f'*40,
            config_sha256='a'*64, run_id='CPU-loss-fixture', statistics=copy.deepcopy(micros[index - 1]),
            reference_minus_current=dict(count=0, finite=0), loss_parts=parts))
    return receipt, actor, numeric


def test_refused_loss_is_observed_but_not_counted_as_successful_gradient_micro():
    receipt, actor, numeric = refused_loss_fixture()
    checked = validate_calibration_update(actor, 'f'*40, 'a'*64, 'CPU-loss-fixture', 0, 1, receipt)
    assert checked['gradient_micro_batches'] == 152
    assert checked['observed_numeric_micro_batches'] == 153
    result = validate_attempt_numeric(numeric, actor, receipt)
    assert result == dict(actual_gradient_backward_micro_batches=152,
        observed_numeric_micro_batches=153, matched_refused_loss_observations=1)


@pytest.mark.parametrize('changed', ['coordinate', 'typed_guard', 'receipt_count', 'missing_row', 'pid'])
def test_unknown_or_misattributed_loss_is_not_a_legal_skip(changed):
    receipt, actor, numeric = refused_loss_fixture()
    if changed == 'coordinate':
        numeric[0]['mini_index'] = 1
    elif changed == 'typed_guard':
        receipt['refusals'][0]['partial_update']['rejection']['matched_guard'] = 'NCCL_TEXT_MATCH'
    elif changed == 'receipt_count':
        receipt['gradient_micro_batches'] = 153
    elif changed == 'missing_row':
        numeric.pop()
    else:
        numeric[0]['pid'] += 1
    with pytest.raises(ValueError):
        validate_calibration_update(actor, 'f'*40, 'a'*64, 'CPU-loss-fixture', 0, 1, receipt)
        validate_attempt_numeric(numeric, actor, receipt)
