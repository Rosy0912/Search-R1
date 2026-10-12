import copy
import pytest
from safety_audit_v1 import validate_micro, validate_guard

def micro():
    s=dict(count=1,finite=1,minimum=0.,maximum=0.,mean=0.)
    return dict(dual_clip=3.,ratio_scope='Dual-Clip: raw_joint_exp_not_computed; complete_joint_logratios_only',joint_action_ratios=dict(count=0),current_log_probs=s.copy(),policy_loss=s.copy(),joint_action_logratios=s.copy())

def guard():
    s=dict(coverage='fixed_first_global256_trajectories_complete_actions',quantiles='exact_global_samples',weight_mass=256.,valid_token_count=256,independent_action_count=256,negative_action_count=0,negative_action_weight_mass=0.,rejected_numeric_count=0,invalid_count=0,kl_log_bound_exceeded_count=0,negative_tail_fraction_count=0.,negative_tail_fraction_weight=0.,dual_clip_saturated_action_fraction=0.,dual_clip_saturated_objective_mass_fraction=0.,max_abs_token_delta=0.,action_token_length_p95=1.,action_token_length_max=1.,negative_action_logratio_p95=None,negative_action_logratio_p99=None,negative_action_logratio_max=None,sampled_old_policy_kl=0.)
    return dict(execution_mode='native_fixture',admission_scope='test_thresholds',dual_clip=3.,kl_log_bound=30.,rank=0,pid=1,commit='a'*40,run_id='native',temperature=1.,fixed_monitor_hashes={'input':'a'},native_binding_sha256='b'*64,target_old_policy_kl=100.,negative_tail_limit=1.,statistics=s,allowed=True)

def test_valid_and_empty_loss_refused():
    r=micro();validate_micro(r);r['policy_loss']['count']=0;r['policy_loss']['finite']=0
    with pytest.raises(ValueError):validate_micro(r)

@pytest.mark.parametrize('key,value',[('sampled_old_policy_kl',float('nan')),('sampled_old_policy_kl',float('inf')),('sampled_old_policy_kl',101.),('weight_mass',0.),('invalid_count',1),('negative_tail_fraction_count',1.1)])
def test_allowed_bad_statistics_refused(key,value):
    r=guard();validate_guard(r,'native_fixture','b'*64);r['statistics'][key]=value
    with pytest.raises(ValueError):validate_guard(r,'native_fixture','b'*64)

def test_self_reported_threshold_inflation_refused():
    r=guard();r['target_old_policy_kl']=1000.
    with pytest.raises(ValueError):validate_guard(r,'native_fixture','b'*64)


def test_numeric_requires_all_actual_finite_scalar_components():
    from safety_audit_v1 import validate_numeric
    scalar=dict(count=1,finite=1,minimum=0.,maximum=0.)
    r=dict(loss_parts={k:scalar.copy() for k in ('policy','entropy','kl','total')},reference_minus_current=scalar.copy())
    validate_numeric(r)
    for key in ('policy','entropy','kl','total'):
        bad=copy.deepcopy(r);del bad['loss_parts'][key]
        with pytest.raises(ValueError):validate_numeric(bad)
        bad=copy.deepcopy(r);bad['loss_parts'][key]['count']=0;bad['loss_parts'][key]['finite']=0
        with pytest.raises(ValueError):validate_numeric(bad)
    bad=copy.deepcopy(r);bad['loss_parts']={}
    with pytest.raises(ValueError):validate_numeric(bad)
