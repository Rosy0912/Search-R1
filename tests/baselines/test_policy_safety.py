import math
import pytest
import torch
from search_r1.baselines.policy_safety import (dual_clip_terms, joint_log_ratios, local_statistics,
                                              global_statistics, rejection_reason, validate_config)
from search_r1.baselines.objectives import weighted_policy_loss


def test_direct_formula_and_gradient_normal_points():
    x = torch.tensor([-.4, -.1, .1, .5, 2.] * 3, dtype=torch.float64, requires_grad=True)
    a = torch.tensor([1.] * 5 + [-1.] * 5 + [0.] * 5)
    mask = torch.ones_like(a)
    exact = torch.maximum(-a * x.exp(), -a * x.exp().clamp(.8, 1.2))
    exact = torch.where(a < 0, torch.minimum(exact, -3 * a), exact)
    safe = dual_clip_terms(x, a, mask, .2, 3.)
    torch.testing.assert_close(safe, exact)
    torch.testing.assert_close(torch.autograd.grad(safe.sum(), x, retain_graph=True)[0],
                               torch.autograd.grad(exact.sum(), x)[0])


@pytest.mark.parametrize('adv', [-1., 0., 1.])
def test_extreme_loss_and_gradient_finite(adv):
    delta = torch.tensor([800., -800.], requires_grad=True)
    value = dual_clip_terms(delta, torch.full_like(delta, adv), torch.ones_like(delta), .2, 3.)
    value.sum().backward()
    assert torch.isfinite(value).all() and torch.isfinite(delta.grad).all()
    if adv == -1:
        torch.testing.assert_close(value, torch.tensor([3., .8], dtype=torch.float64))
    if adv == 0:
        assert not value.any() and not delta.grad.any()


def test_masked_nan_and_effective_nan_rejection():
    x = torch.tensor([float('nan'), .1], requires_grad=True)
    value = dual_clip_terms(x, torch.ones(2), torch.tensor([0, 1]), .2, 3.)
    assert torch.isfinite(value).all()
    value.sum().backward();assert torch.isfinite(x.grad).all()
    with pytest.raises(ValueError):dual_clip_terms(x, torch.ones(2), torch.ones(2), .2, 3.)


def test_joint_cast_before_subtraction_and_sum_and_zero_saturated_gradient():
    current = torch.tensor([[1e8, -1e8, 3., 800.]], dtype=torch.float32, requires_grad=True)
    old = torch.zeros_like(current)
    mask = torch.ones_like(current);adv = -torch.ones_like(current)
    _, joint = joint_log_ratios(old, current, adv, mask, torch.tensor([[0, 0, 0, 1]]))
    assert joint.dtype == torch.float64 and joint[0, 0] == 3
    value = dual_clip_terms(joint, adv, mask, .2, 3.)
    value.sum().backward();assert not current.grad.any()


def test_joint_action_gradient_and_global_row_denominator_under_micro_splits():
    x = torch.tensor([[.03, .02, .01], [-.02, -.01, .01], [.01, .01, .01]], requires_grad=True)
    old = torch.zeros_like(x);a = torch.tensor([[1., 1., -1.]]).expand_as(x)
    ids = torch.tensor([[0, 0, 1]]).expand_as(x);mask = torch.ones_like(x);w = mask / 3
    whole = weighted_policy_loss(old, x, a, mask, w, .2, 3., ids, dual_clip=3.)[0]
    split = sum(weighted_policy_loss(old[i:i+1], x[i:i+1], a[i:i+1], mask[i:i+1], w[i:i+1], .2, 3., ids[i:i+1], dual_clip=3.)[0] for i in range(3))
    torch.testing.assert_close(whole, split)
    torch.testing.assert_close(torch.autograd.grad(whole, x, retain_graph=True)[0], torch.autograd.grad(split, x)[0])


def stats(delta, adv=None, mask=None):
    delta = torch.tensor(delta, dtype=torch.float64).reshape(1, -1)
    adv = -torch.ones_like(delta) if adv is None else torch.tensor(adv).reshape_as(delta)
    mask = torch.ones_like(delta) if mask is None else torch.tensor(mask).reshape_as(delta)
    ids = torch.arange(delta.numel()).reshape_as(delta).long()
    return local_statistics(torch.zeros_like(delta), delta, adv, mask, mask / max(1, delta.numel()), ids, 3., 30.)


def test_numeric_refusal_is_not_clipped_kl_and_action_counts_not_tokens():
    s = global_statistics([stats([800., .1])])
    assert s['sampled_old_policy_kl'] is None and s['kl_log_bound_exceeded_count'] == 1
    assert rejection_reason(s, {'target_old_policy_kl': .01, 'negative_tail_limit': .9})
    s = global_statistics([stats([0., 2.])])
    assert s['negative_tail_fraction_count'] == .5 and s['independent_action_count'] == 2
    assert rejection_reason(s, {'target_old_policy_kl': 100., 'negative_tail_limit': .4}) == 'NEGATIVE_JOINT_ACTION_TAIL_COUNT'
    s = global_statistics([stats([0.], adv=[1.])])
    assert s['negative_action_count'] == 0 and rejection_reason(s, {'target_old_policy_kl': .01, 'negative_tail_limit': .1}) is None
    s = global_statistics([stats([0.], mask=[0])])
    assert rejection_reason(s, {'target_old_policy_kl': .01, 'negative_tail_limit': .1})


def test_production_requires_calibration_thresholds():
    c = dict(enabled=True, dual_clip=3., kl_log_bound=30., monitor_first_global_rows=256,
             target_old_policy_kl=None, negative_tail_limit=None, calibration_evidence_sha256=None)
    validate_config(c, require_thresholds=False)
    with pytest.raises(ValueError):validate_config(c)


def test_corrupt_mask_rejected_but_padding_probability_nan_not_rejected():
    part = local_statistics(torch.zeros(1,1),torch.zeros(1,1),-torch.ones(1,1),
                            torch.tensor([[float('nan')]]),torch.ones(1,1),torch.zeros(1,1,dtype=torch.long),3.,30.)
    result=global_statistics([part])
    assert rejection_reason(result,dict(target_old_policy_kl=.01,negative_tail_limit=.5))
    with pytest.raises(ValueError):dual_clip_terms(torch.zeros(1),torch.ones(1),torch.tensor([float('nan')]),.2,3.)
    c=dict(enabled=True,dual_clip=3.,kl_log_bound=30.,monitor_first_global_rows=256,
           target_old_policy_kl=.01,negative_tail_limit=.5,calibration_evidence_sha256='x')
    with pytest.raises(ValueError):validate_config(c)


def test_native_scope_requires_real_reviewed_binding_and_cannot_be_production_proof(tmp_path,monkeypatch):
    import hashlib,json
    c=dict(enabled=True,mode='native_fixture',dual_clip=3.,kl_log_bound=30.,monitor_first_global_rows=256,
           target_old_policy_kl=.01,negative_tail_limit=.5)
    with pytest.raises(ValueError):validate_config(c)
    p=tmp_path/'binding.json'
    raw=json.dumps(dict(status='REVIEWED_FOR_NATIVE_ONLY',validation_scope='GIGPO_POLICY_SAFETY_NATIVE_ONLY',
         code_commit='f'*40,restore_step=25,restore_manifest_sha256='3a073aba77147db7d581cd514ee5c4ae34965e5ff9c163716bd25b47aeb478e4',restore_checkpoint='/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004/runs/gigpo-e1-instruct300-job3185-1791112239036063948/checkpoints/actor/global_step_25',allowed_safety_modes=['native_fixture','calibration'],native_test_threshold_sets=[[.01,.5]])).encode();p.write_bytes(raw)
    monkeypatch.setenv('GIGPO_POLICY_NATIVE_BINDING',str(p))
    monkeypatch.setenv('GIGPO_POLICY_NATIVE_BINDING_SHA256',hashlib.sha256(raw).hexdigest())
    monkeypatch.setenv('SEARCH_R1_BASELINE_COMMIT','f'*40)
    validate_config(c)
    c['mode']='production'
    with pytest.raises(ValueError):validate_config(c)
    c['mode']='native_fixture';p.write_bytes(raw+b' ')
    with pytest.raises(ValueError):validate_config(c)
