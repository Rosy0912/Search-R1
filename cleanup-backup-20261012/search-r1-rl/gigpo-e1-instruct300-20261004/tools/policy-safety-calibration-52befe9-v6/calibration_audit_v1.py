"""Actual rollout audit: legal empty local micros retain zero objective contribution."""
import math
from calibration_contract_v1 import require
from safety_audit_v1 import validate_guard, validate_micro, validate_numeric


def validate_calibration_update(row, source, config_sha, run_id, rank):
    require(row['kind'] == 'actor_outer' and row['outer_step'] == 26
            and row['rank'] == rank and row['commit'] == source
            and row['config_sha256'] == config_sha and row['run_id'] == run_id,
            'Wrong actual actor update identity')
    require(row['rows'] == 320 and row['epochs'] == 1 and row['optimizer_steps'] == 10
            and row['skipped_empty_minibatches'] == 0 and len(row['micro_stats']) == 160,
            'Incomplete real rollout outer budget')
    require(row['grad_finite'] == row['grad_nonzero'] == 10 and len(row['grad_norms']) == 10
            and all(math.isfinite(v) and v > 0 for v in row['grad_norms']), 'Missing finite real gradients')
    require(row['fixed_unchanged'] and row['shard_layout_unchanged'], 'Old inputs or local shard layout changed')
    require(len(row['row_denominators']) == 10 and all(
            r['local_rows'] == 32 and r['actual_denominator'] == 32 and r['world_size'] == 8
            for r in row['row_denominators']), 'Globalmini256 reduction differs')
    empty = 0
    for micro in row['micro_stats']:
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
    return dict(gradient_micro_batches=160, legal_empty_local_micro_batches=empty, actual_adam_writes=10)


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
