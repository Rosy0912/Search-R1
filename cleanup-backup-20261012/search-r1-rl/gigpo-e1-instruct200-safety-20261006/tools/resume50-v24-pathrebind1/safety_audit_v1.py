"""Strict new opt-in audit schema. Never accept old raw-ratio scope or a calibration as production."""
import math


def require(ok, message):
    if not ok:raise ValueError(message)


def validate_micro(record):
    require(record.get('dual_clip') == 3.0, 'Missing explicit Dual-Clip coefficient')
    require(record.get('ratio_scope') == 'Dual-Clip: raw_joint_exp_not_computed; complete_joint_logratios_only', 'Wrong ratio scope')
    require(record['joint_action_ratios']['count'] == 0, 'Raw joint exp must not be materialized')
    for key in ('current_log_probs','policy_loss','joint_action_logratios'):
        value=record[key];require(value['count']>0 and value['finite']==value['count'], 'Missing/nonfinite '+key)
        for name in ('minimum','maximum','mean'):
            if name in value:require(value[name] is not None and math.isfinite(value[name]), 'Invalid '+key+' '+name)
    require(record['joint_action_logratios']['count'] > 0, 'Missing actual action deltas')
    require(record['policy_loss']['count']==1, 'Policy loss must be the actual scalar')


def validate_guard(record, mode, binding_sha=None, calibration_sha=None, native_thresholds=(100., 1.)):
    expected={'native_fixture':'test_thresholds','calibration':'calibration_candidate_token_kl_and_numeric_only','production':'production_calibrated'}[mode]
    require(record['execution_mode']==mode and record['admission_scope']==expected, 'Execution scope mismatch')
    require(record['dual_clip']==3. and record['kl_log_bound']==30., 'Wrong safety bounds')
    require(record['rank'] in range(8) and record['pid']>0 and record['commit'] and record['run_id'], 'Missing actual identity')
    require(record['temperature']==1. and record['fixed_monitor_hashes'], 'Unbound fixed monitor')
    require(record['statistics']['coverage']=='fixed_first_global256_trajectories_complete_actions', 'Wrong coverage')
    require(record['statistics']['quantiles']=='exact_global_samples', 'Not exact global quantiles')
    if mode=='production':
        require(record['calibration_evidence_sha256']==calibration_sha and calibration_sha, 'Missing production calibration')
        require(record['target_old_policy_kl']>0 and 0<=record['negative_tail_limit']<=1, 'Missing production thresholds')
    else:
        require(record['native_binding_sha256']==binding_sha and binding_sha, 'Wrong isolated native binding')
    if mode=='native_fixture':
        require((record['target_old_policy_kl'],record['negative_tail_limit'])==tuple(native_thresholds), 'Wrong bound native test thresholds')
    if mode=='calibration':
        require(record['candidate_old_policy_kl']==record['target_old_policy_kl']==.2
                and record['negative_tail_limit'] is None and record['production_thresholds_accepted'] is False,
                'Observed guard differs from human token KL or unselected empirical tail')
    stats=record['statistics']
    for key in ('weight_mass','valid_token_count','independent_action_count','negative_action_count',
                'negative_action_weight_mass','rejected_numeric_count','invalid_count','kl_log_bound_exceeded_count'):
        require(math.isfinite(stats[key]) and stats[key]>=0, 'Invalid statistic '+key)
    for key in ('negative_tail_fraction_count','negative_tail_fraction_weight','dual_clip_saturated_action_fraction',
                'dual_clip_saturated_objective_mass_fraction'):
        require(math.isfinite(stats[key]) and 0<=stats[key]<=1+1e-12, 'Invalid fraction '+key)
    for key in ('max_abs_token_delta','action_token_length_p95','action_token_length_max'):
        require(stats[key] is not None and math.isfinite(stats[key]) and stats[key]>=0, 'Invalid magnitude '+key)
    for key in ('negative_action_logratio_p95','negative_action_logratio_p99','negative_action_logratio_max'):
        require((stats['negative_action_count']==0 and stats[key] is None)
                or (stats['negative_action_count']>0 and stats[key] is not None and math.isfinite(stats[key])), 'Invalid negative-tail quantile')
    if record['allowed']:
        require(stats['weight_mass']>0 and stats['valid_token_count']>0, 'Empty effective monitor')
        require(stats['rejected_numeric_count']==0 and stats['invalid_count']==0 and stats['kl_log_bound_exceeded_count']==0, 'Invalid allowed guard')
        kl=stats['sampled_old_policy_kl'];require(kl is not None and math.isfinite(kl) and kl>=0, 'Missing/nonfinite KL')
        if mode=='calibration':
            require(record['candidate_old_policy_kl']==record['target_old_policy_kl']==.2 and kl<=.2
                and record['negative_tail_limit'] is None and record['production_thresholds_accepted'] is False,
                'Human token KL or unselected empirical tail differs')
        if mode!='calibration':
            require(kl<=record['target_old_policy_kl'] and stats['negative_tail_fraction_count']<=record['negative_tail_limit'], 'Allowed guard exceeds thresholds')
    else:require(record['rejection_reason'], 'Missing rejection reason')


def validate_update(record, expected_steps=10):
    require(record['optimizer_steps']==expected_steps and record['skipped_empty_minibatches']==0, 'Incomplete update budget')
    require(len(record['micro_stats'])==expected_steps*16, 'Wrong gradient micro count')
    require(record['grad_finite']==expected_steps and record['grad_nonzero']==expected_steps, 'Invalid gradients')
    require(len(record['grad_norms'])==expected_steps and all(math.isfinite(v) and v>0 for v in record['grad_norms']), 'Missing actual gradients')
    require(len(record['row_denominators'])==10 and all(r['local_rows']==32 and r['actual_denominator']==32 and r['world_size']==8 for r in record['row_denominators']), 'Invalid real row denominators')
    for micro in record['micro_stats']:validate_micro(micro)


def validate_numeric(record):
    parts=record['loss_parts']
    require(set(parts)=={'policy','entropy','kl','total'}, 'Missing exact loss components')
    for key,value in parts.items():
        require(value['count']==1 and value['finite']==1, 'Missing actual scalar '+key)
        require(all(value.get(n) is not None and math.isfinite(value[n]) for n in ('minimum','maximum')), 'Nonfinite scalar '+key)
    value=record['reference_minus_current']
    require(value['count']>0 and value['finite']==value['count'], 'Missing/nonfinite reference delta')
    require(all(value.get(n) is not None and math.isfinite(value[n]) for n in ('minimum','maximum')), 'Nonfinite reference delta')
