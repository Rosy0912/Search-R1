"""UNEXECUTED CPU diagnostics for a matched loss observation that never backwarded."""
import os
import tempfile
from unittest.mock import patch
import pytest
from calibration_audit_v1 import validate_calibration_update, validate_attempt_numeric
from test_typed_refusal_producer_v11 import (
    emit_actual_loss, numeric_attempt_from_actual_record, producer_fixture, source_root)


def refused_loss_fixture():
    # Keep the old row checks while removing its fabricated marker provenance.
    with tempfile.TemporaryDirectory(prefix='actual-loss-producer-v11-') as folder, patch.dict(os.environ):
        actor, admit, rejected, _, _, _ = producer_fixture(source_root(), folder)
        record = emit_actual_loss(actor, admit, rejected)
    return numeric_attempt_from_actual_record(record)


def test_refused_loss_is_observed_but_not_counted_as_successful_gradient_micro():
    receipt, actor, numeric = refused_loss_fixture()
    checked = validate_calibration_update(actor, actor['commit'], actor['config_sha256'], actor['run_id'], 0, 1, receipt)
    assert checked['gradient_micro_batches'] == 152
    assert checked['observed_numeric_micro_batches'] == 153
    result = validate_attempt_numeric(numeric, actor, receipt)
    assert result == dict(actual_gradient_backward_micro_batches=152,
        observed_numeric_micro_batches=153, matched_refused_loss_observations=1)


@pytest.mark.parametrize('changed', ['coordinate', 'typed_guard', 'receipt_count', 'missing_row', 'pid'])
def test_unknown_or_misattributed_loss_is_not_a_legal_skip(changed):
    receipt, actor, numeric = refused_loss_fixture()
    if changed == 'coordinate':
        numeric[16]['mini_index'] = 3
    elif changed == 'typed_guard':
        receipt['refusals'][0]['partial_update']['rejection']['matched_guard'] = 'NCCL_TEXT_MATCH'
    elif changed == 'receipt_count':
        receipt['gradient_micro_batches'] = 153
    elif changed == 'missing_row':
        numeric.pop()
    else:
        numeric[0]['pid'] += 1
    with pytest.raises(ValueError):
        validate_calibration_update(actor, actor['commit'], actor['config_sha256'], actor['run_id'], 0, 1, receipt)
        validate_attempt_numeric(numeric, actor, receipt)
