"""Actual rollout audit: legal empty local micros retain zero objective contribution."""
import math
from calibration_contract_v1 import require
from safety_audit_v1 import validate_guard, validate_micro, validate_numeric


def validate_calibration_update(row, source, config_sha, run_id, rank, outer, receipt):
    require(row['kind'] == ('actor_outer' if receipt['counted_complete'] else 'actor_attempt_incomplete')
            and row['outer_step'] == outer and row['attempt_id'] == receipt['attempt_id']
            and row['rank'] == rank and row['commit'] == source
            and row['config_sha256'] == config_sha and row['run_id'] == run_id,
            'Wrong actual actor update identity')
    writes, processed = receipt['actual_successful_adam_writes'], receipt['opportunities_processed']
    micros = receipt['gradient_micro_batches']
    require(row['rows'] == 320 and row['epochs'] == 1 and row['optimizer_steps'] == writes
            and row['skipped_empty_minibatches'] == receipt['skipped_opportunities']
            and row['opportunities_processed'] == processed
            and row['counted_complete'] is receipt['counted_complete']
            and row['actual_gradient_micro_batches'] == micros,
            'Actor audit differs from real opportunity/write receipts')
    require(row['grad_finite'] == writes and 0 <= row['grad_nonzero'] <= writes and len(row['grad_norms']) == writes
            and all(math.isfinite(v) and v >= 0 for v in row['grad_norms']), 'Missing finite real gradients')
    require(row['fixed_unchanged'] and row['shard_layout_unchanged'], 'Old inputs or local shard layout changed')
    require(len(row['row_denominators']) == processed and all(
            r['local_rows'] == 16 and r['actual_denominator'] == 16 and r['world_size'] == 8
            for r in row['row_denominators']), 'Globalmini128 reduction differs')
    refused_losses = [value['partial_update']['rejection'] for value in receipt['refusals']
                      if value['partial_update']['rejection']['phase'].endswith('_loss')]
    require(len(row['micro_stats']) == micros + len(refused_losses),
            'Observed numeric micros differ from actual backwards and matched refused losses')
    # Numeric rows carry real mini/micro coordinates. They are validated alongside
    # this complete micro_stats list by validate_attempt_numeric below.
    empty = 0
    for micro in row['micro_stats']:
        if micro['policy_loss']['finite'] != micro['policy_loss']['count']:
            require(refused_losses, 'An invalid loss has no actual matched refusal')
            continue
        if micro['current_log_probs']['count']:
            validate_micro(micro)
            continue
        empty += 1
        require(micro['current_log_probs']['finite'] == 0
                and micro['joint_action_logratios']['count'] == micro['joint_action_logratios']['finite'] == 0
                and micro['joint_action_ratios']['count'] == 0
                and micro['dual_clip'] == 3.
                and micro['ratio_scope'] == 'Dual-Clip: raw_joint_exp_not_computed; complete_joint_logratios_only',
                'Empty micro fabricated probability contributions')
        loss = micro['policy_loss']
        require(loss['count'] == loss['finite'] == 1 and loss['minimum'] == loss['maximum'] == 0.,
                'Empty effective micro must retain actual zero PG contribution')
    return dict(gradient_micro_batches=micros, observed_numeric_micro_batches=len(row['micro_stats']),
                legal_empty_local_micro_batches=empty, actual_adam_writes=writes,
                opportunities_processed=processed, counted_complete=receipt['counted_complete'])


def validate_calibration_numeric(row):
    if row['reference_minus_current']['count']:
        validate_numeric(row)
        return
    parts = row['loss_parts']
    require(set(parts) == {'policy', 'entropy', 'kl', 'total'}, 'Empty micro lost scalar loss components')
    for value in parts.values():
        require(value['count'] == value['finite'] == 1 and value['minimum'] == value['maximum'] == 0.,
                'Empty effective micro changed mathematical zero loss')
    require(row['reference_minus_current']['finite'] == 0, 'Empty mask fabricated reference probability')


def validate_attempt_numeric(rows, actor_row, receipt):
    """A nonfinite recorded loss is evidence only for its exact matched rejection."""
    from calibration_state_v1 import state_digest
    refused = {}
    for value in receipt['refusals']:
        rejection = value['partial_update']['rejection']
        phase = rejection['phase']
        if phase.startswith('gradient_micro') and phase.endswith('_loss'):
            micro_index = int(phase[len('gradient_micro'):-len('_loss')])
            key = (rejection['mini_index'], micro_index)
            require(key not in refused, 'Duplicate matched refused loss coordinate')
            refused[key] = rejection
    require(len(rows) == len(actor_row['micro_stats']), 'Missing actual observed numeric micro rows')
    observed_refusals = set()
    for index, row in enumerate(rows, 1):
        require(row['attempt_id'] == receipt['attempt_id'] and row['rank'] == receipt['rank']
                and row['pid'] == receipt['pid'] and row['micro_index'] == index
                and row['commit'] == actor_row['commit'] and row['run_id'] == actor_row['run_id']
                and row['config_sha256'] == actor_row['config_sha256']
                and state_digest(row['statistics']) == state_digest(actor_row['micro_stats'][index - 1]),
                'Numeric micro differs from the actual rank attempt audit')
        key = (row['mini_index'], row['micro_index_in_mini'])
        if key in refused:
            require(refused[key].get('matched_guard') == 'ORIGINAL_ALL_REDUCE_MAX_LOSS_REFUSAL'
                    and refused[key].get('refusal_kind') == 'known_numeric'
                    and refused[key].get('global_diagnostic_io_failed') is False
                    and refused[key].get('details') == 'NONFINITE_TRAINING_LOSS',
                    'Unknown failure cannot authorize an invalid numeric observation')
            observed_refusals.add(key)
        else:
            validate_calibration_numeric(row)
    require(observed_refusals == set(refused), 'Missing exact matched refused loss observations')
    return dict(actual_gradient_backward_micro_batches=receipt['gradient_micro_batches'],
                observed_numeric_micro_batches=len(rows), matched_refused_loss_observations=len(refused))
