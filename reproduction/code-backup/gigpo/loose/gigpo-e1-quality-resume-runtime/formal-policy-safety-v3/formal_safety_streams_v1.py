"""New bounded-ratio audit adapter: never accept calibration as production."""
import math
from pathlib import Path
from formal_protection_contract_v1 import require, DATA
from formal_audit_utils_v1 import rows
from safety_audit_v1 import validate_micro, validate_numeric, validate_guard


def bounded_micro(record):
    if record['current_log_probs']['count']:
        validate_micro(record)
        return
    require(record['current_log_probs']['finite'] == 0
            and record['joint_action_logratios']['count'] == record['joint_action_logratios']['finite'] == 0
            and record['joint_action_ratios']['count'] == 0
            and record['dual_clip'] == 3.
            and record['ratio_scope'] == 'Dual-Clip: raw_joint_exp_not_computed; complete_joint_logratios_only',
            'Empty localmicro fabricated probability/action contributions')
    loss = record['policy_loss']
    require(loss['count'] == loss['finite'] == 1 and loss['minimum'] == loss['maximum'] == 0.,
            'Legal empty effective localmicro must have actual zeroPG')


def bounded_numeric(record):
    bounded_micro(record['statistics'])
    if record['reference_minus_current']['count']:
        validate_numeric(record)
        return
    require(record['reference_minus_current']['finite'] == 0, 'Empty effective ref tensor fabricated')
    require(set(record['loss_parts']) == {'policy','entropy','kl','total'}, 'Missing actual loss parts')
    for value in record['loss_parts'].values():
        require(value['count'] == value['finite'] == 1 and value['minimum'] == value['maximum'] == 0.,
                'Empty effective micro changed mathematical zero loss')


def audit_production_streams(run, start, end, source, config_sha, workers, policy):
    """Complete outers only. Rejected partial writes need separate failure evidence."""
    require(policy['mode'] == 'production' and policy['calibration_evidence_sha256'], 'No production artifact')
    guards_by_event = {}
    for rank in range(8):
        micros = rows(Path(run) / 'audit/numeric' / f'micro-{rank}.jsonl')
        guards = rows(Path(run) / 'audit/policy-safety' / f'rank-{rank}.jsonl')
        require(len(micros) == (end-start+1)*160 and len(guards) == (end-start+1)*20,
                'Actual160gradientmicro/20prepost evidence incomplete')
        for i, record in enumerate(micros):
            step, micro = start + i//160, i%160
            require(record['outer_step'] == step and record['micro_index'] == micro+1
                    and record['optimizer_steps_completed'] == micro//16, 'Actual micro/Adam order differs')
            require(record['rank'] == rank and record['pid'] == workers[rank]['pid']
                    and record['commit'] == source and record['config_sha256'] == config_sha
                    and record['data_identity'] == DATA and record['run_id'] == Path(run).name,
                    'Actual numerical stream identity differs')
            bounded_numeric(record)
        for i, record in enumerate(guards):
            step, inner = start+i//20, i%20
            mini, post = inner//2, inner%2
            require(record['outer_step'] == step and record['mini_index'] == mini
                    and record['phase'] == ('after' if post else 'before')
                    and record['actual_optimizer_writes'] == mini+post, 'Actual pre/post Adam placement differs')
            require(record['rank'] == rank and record['pid'] == workers[rank]['pid']
                    and record['commit'] == source and record['config_sha256'] == config_sha
                    and record['data_identity'] == DATA and record['run_id'] == Path(run).name,
                    'Actual production guard identity differs')
            validate_guard(record, 'production', calibration_sha=policy['calibration_evidence_sha256'])
            require(record['target_old_policy_kl'] == policy['target_old_policy_kl']
                    and record['negative_tail_limit'] == policy['negative_tail_limit']
                    and record['allowed'] and not record['monitor_identity_changed'],
                    'Rejected/unbound guard cannot complete an outer')
            expected_monitor = guards[(i//20)*20]['fixed_monitor_hashes']
            require(record['fixed_monitor_hashes'] == expected_monitor, 'Rollout monitor/old policy changed during reuse')
            if post:
                norm = record['gradient_norm_after_clip']
                require(math.isfinite(norm) and 0 <= norm <= 1.000001, 'Actual postclip gradient exceeds1')
            guards_by_event.setdefault((step, inner), []).append(record)
    for event, records in guards_by_event.items():
        require(len(records) == 8 and {r['rank'] for r in records} == set(range(8)), 'Missing synchronized guard participant')
        require(all(r['statistics'] == records[0]['statistics'] for r in records), 'Ranks do not agree on global weighted statistics')
    return dict(status='RAW_STREAMS_VERIFIED_NOT_INDEPENDENT_ACCEPTANCE', actual_new_complete_outers=end-start+1,
                rank_count=8, adam_per_outer10=10, gradient_micro_per_outer160=160,
                prepost_guards_per_outer_per_rank20=20, source_commit=source,
                production_artifact_sha256=policy['calibration_evidence_sha256'])
