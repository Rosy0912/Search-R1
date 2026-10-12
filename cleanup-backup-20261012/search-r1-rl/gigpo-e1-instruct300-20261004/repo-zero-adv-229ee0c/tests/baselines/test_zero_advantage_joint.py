"""The actual3227 overflow is an unused PG exponential, not a bad gradient."""
import pytest
import torch
from search_r1.baselines.objectives import weighted_policy_loss
from search_r1.baselines.loss_scaling import active_joint_bound, coordinated_scale
from search_r1.baselines.runtime_audit import ActorAudit
from verl.trainer.ppo.core_algos import kl_penalty


def fixture(value=866.7740478515625, advantage=0., dtype=torch.float64):
    old = torch.full((1, 264), -5., dtype=dtype)
    current = (old + value / 264).clone().requires_grad_(True)
    adv = torch.full_like(current, advantage)
    mask = torch.ones_like(current)
    return old, current, adv, mask, mask / 264, torch.ones_like(old, dtype=torch.long)


def test_actual_overflow_exact_zero_pg_keeps_reference_kl_and_original_logratio():
    old, x, adv, mask, weights, ids = fixture()
    pg, clip, diagnostic = weighted_policy_loss(old, x, adv, mask, weights, .2, 1., ids)
    assert pg == 0 and clip == 0
    torch.testing.assert_close(diagnostic, -(x - old).sum())
    g, = torch.autograd.grad(pg, x, retain_graph=True)
    assert torch.isfinite(g).all() and torch.count_nonzero(g) == 0
    bound, bad = active_joint_bound(old, x, adv, mask, weights, ids, .2)
    assert bad == 0 and bound == -float('inf') and coordinated_scale(bound, bad) == 1
    kl = (kl_penalty(x, old + 3., 'low_var_kl') * weights).sum() * .001
    kg, = torch.autograd.grad(pg + kl, x)
    assert torch.isfinite(kl) and torch.isfinite(kg).all() and torch.count_nonzero(kg) == 264


@pytest.mark.parametrize('advantage', [-1., 1., 1e-12])
def test_nonzero_overflow_remains_rejected(advantage):
    args = fixture(advantage=advantage)
    _, bad = active_joint_bound(*args, .2)
    assert bad == 1
    with pytest.raises(ValueError, match='ratio overflow'):
        weighted_policy_loss(*args[:-1], .2, 1., args[-1])


def test_near_zero_action_is_not_erased_by_allclose_tolerance():
    old, x, adv, mask, weights, ids = fixture()
    adv[0, -1] = 1e-12
    assert torch.allclose(adv, torch.zeros_like(adv))
    assert active_joint_bound(old, x, adv, mask, weights, ids, .2)[1] == 1
    with pytest.raises(ValueError, match='ratio overflow'):
        weighted_policy_loss(old, x, adv, mask, weights, .2, 1., ids)


def test_zero_advantage_cannot_hide_native_sum_overflow_or_bad_inputs():
    old, x, adv, mask, weights, ids = fixture(dtype=torch.float32)
    with torch.no_grad():
        old.fill_(-1e38); x.fill_(1e38)
    assert torch.isfinite(old).all() and torch.isfinite(x).all()
    assert active_joint_bound(old, x, adv, mask, weights, ids, .2)[1] == 1
    with pytest.raises(ValueError, match='joint logratio'):
        weighted_policy_loss(old, x, adv, mask, weights, .2, 1., ids)
    with torch.no_grad():
        x.fill_(float('nan'))
    assert active_joint_bound(old, x, adv, mask, weights, ids, .2)[1] == 1
    with pytest.raises(ValueError, match='Nonfinite policy'):
        weighted_policy_loss(old, x, adv, mask, weights, .2, 1., ids)


def test_ordinary_mixed_actions_match_original_surrogate_and_full_gradients():
    old = torch.full((2, 8), -5., dtype=torch.float64)
    x = (old + torch.tensor([[.1, .1, -.1, -.1, .05, .05, .5, .5]]).double()).requires_grad_()
    adv = torch.tensor([[0., 0., -1., -1., .4, .4, 1., 1.]]).double().expand_as(old)
    ids = (torch.arange(8) // 2).expand_as(old)
    mask = torch.ones_like(old); mask[1, 6:] = 0
    weights = mask / mask.sum(-1, keepdim=True)
    loss, clip, kl = weighted_policy_loss(old, x, adv, mask, weights, .2, 2., ids)
    L = (x - old).reshape(2, 4, 2).sum(-1).repeat_interleave(2, -1)
    ratio = L.exp(); one = -adv * ratio; two = -adv * ratio.clamp(.8, 1.2)
    baseline = (torch.maximum(one, two) * weights).sum() / 2
    torch.testing.assert_close(loss, baseline, rtol=0, atol=0)
    torch.testing.assert_close(clip, ((two > one).double() * weights).sum() / weights.sum())
    torch.testing.assert_close(kl, (-L * weights).sum() / weights.sum())
    a, = torch.autograd.grad(loss, x, retain_graph=True)
    b, = torch.autograd.grad(baseline, x)
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_audit_does_not_invent_finite_raw_ratio_for_zero_action(monkeypatch):
    monkeypatch.delenv('SEARCH_R1_BASELINE_AUDIT_DIR', raising=False)
    old, x, adv, mask, weights, ids = fixture()
    audit = ActorAudit.__new__(ActorAudit); audit.micro_stats = []
    audit.micro(x, old, mask, x.sum() * 0, ids, advantages=adv)
    r = audit.micro_stats[-1]
    assert r['current_log_probs']['count'] == 264
    assert r['ratios']['count'] == 0 and r['joint_action_ratios']['count'] == 0
    assert r['exact_zero_pg_actions'] == 1 and r['exact_zero_pg_tokens'] == 264
    assert r['joint_action_logratios']['count'] == r['joint_action_logratios']['finite'] == 1
    assert r['joint_action_logratios']['maximum'] > 709
