import pytest
import torch
from search_r1.baselines.objectives import weighted_policy_loss
from search_r1.baselines.loss_scaling import active_joint_bound, coordinated_scale
from search_r1.baselines.gradient_norm import clip_grad_norm64
from verl.trainer.ppo.core_algos import kl_penalty


def objective(x, advantage):
    old = torch.zeros_like(x)
    mask = torch.ones_like(x)
    weights = mask / x.shape[1]
    ids = torch.zeros_like(x, dtype=torch.long)
    adv = torch.full_like(x, advantage)
    loss, _, _ = weighted_policy_loss(old, x, adv, mask, weights, .2, 1., ids)
    kl = (kl_penalty(x, old, 'low_var_kl') * weights).sum()
    bound, bad = active_joint_bound(old, x, adv, mask, weights, ids, .2)
    return loss + .001 * kl, coordinated_scale(bound, bad)


@pytest.mark.parametrize('delta', [0., 1., 100., 500.])
def test_combined_pg_kl_clipped_gradient_matches_double_oracle(delta):
    model = torch.nn.Linear(4, 1, bias=False)
    model.weight.data.fill_(delta / 4)
    loss, scale = objective(model.weight, -1.)
    oracle = model.weight.detach().double().requires_grad_()
    expected_loss, _ = objective(oracle, -1.)
    expected_loss.backward()
    peak = oracle.grad.abs().max()
    norm = (oracle.grad / peak).norm() * peak if peak else oracle.grad.norm()
    expected = oracle.grad * (1. / (norm + 1e-6)).clamp(max=1)
    (loss * scale).backward()
    actual_norm = clip_grad_norm64(model, 1., scale)
    torch.testing.assert_close(actual_norm, norm, rtol=2e-6, atol=1e-8)
    torch.testing.assert_close(model.weight.grad.double(), expected, rtol=2e-6, atol=1e-8)
    assert torch.isfinite(loss)


def test_kl_dominates_clipped_pg_and_zero_advantage():
    for adv in (0., 1.):
        x = torch.full((1, 4), 25., requires_grad=True)
        loss, scale = objective(x, adv)
        assert scale == 1
        loss.backward()
        assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16, torch.float64])
def test_no_clip_tiny_scaled_gradient_does_not_cast_huge_coefficient_first(dtype):
    model = torch.nn.Linear(2, 1, bias=False).to(dtype)
    scale = 1e-30 if dtype == torch.bfloat16 else 1e-40
    model.weight.grad = torch.full_like(model.weight, scale * .1)
    actual = model.weight.grad.double() / scale
    clip_grad_norm64(model, 1., scale)
    torch.testing.assert_close(model.weight.grad.double(), actual, rtol=.02, atol=1e-4)


def test_global_scale_has_one_collective_order_even_when_invalid(monkeypatch):
    import search_r1.baselines.loss_scaling as module
    calls = []
    monkeypatch.setattr(module.dist, 'is_initialized', lambda: True)
    monkeypatch.setattr(module.dist, 'all_reduce', lambda t, **kw: calls.append((t.dtype, kw['op'])))
    with pytest.raises(ValueError, match='all ranks'):
        coordinated_scale(torch.tensor(100., dtype=torch.float64), torch.tensor(1, dtype=torch.int32))
    assert [entry[0] for entry in calls] == [torch.int32, torch.float64]


def test_unrepresentable_scale_refuses_instead_of_clamping():
    with pytest.raises(ValueError, match='Unrepresentable'):
        coordinated_scale(torch.tensor(1000., dtype=torch.float64), torch.tensor(0, dtype=torch.int32))


def test_empty_supervision_participates_with_identity_scale():
    x = torch.tensor([[float('nan')]])
    bound, bad = active_joint_bound(x, x, x, torch.zeros_like(x), torch.zeros_like(x),
                                  torch.full_like(x, -1, dtype=torch.long), .2)
    assert bad == 0
    assert coordinated_scale(bound, bad) == 1


def test_probe_restores_rng_buffers_modes_and_manager_rng_even_on_error():
    from types import SimpleNamespace
    from search_r1.baselines.loss_scaling import TransparentProbe
    from search_r1.baselines.recovery import rng_state
    from search_r1.baselines.instruct300 import equal_state
    model = torch.nn.Sequential(torch.nn.Dropout(.5), torch.nn.Linear(4, 4))
    model.register_buffer('counter', torch.zeros(1))
    manager = SimpleNamespace(gen_random_states=torch.get_rng_state().clone(), torch_random_states=None)
    original_rng = rng_state()
    original_manager = manager.gen_random_states.clone()
    with pytest.raises(RuntimeError, match='forward failed'):
        with TransparentProbe(model, manager):
            with torch.no_grad():
                model(torch.ones(2, 4))
                model.counter.add_(3)
                model.eval()
                manager.gen_random_states.zero_()
                raise RuntimeError('forward failed')
    assert equal_state(rng_state(), original_rng)
    assert torch.equal(manager.gen_random_states, original_manager)
    assert model.training and model[0].training
    assert model.counter.item() == 0


def test_strided_gradient_is_updated_in_place():
    model = torch.nn.Linear(4, 4, bias=False)
    model.weight.grad = torch.full((4, 4), .001).t()
    assert not model.weight.grad.is_contiguous()
    clip_grad_norm64(model, 1., .01)
    torch.testing.assert_close(model.weight.grad, torch.full((4, 4), .1))


def test_bad_action_dtype_collects_flag_before_coordinated_refusal():
    x = torch.zeros(1, 1)
    bound, bad = active_joint_bound(x, x, torch.ones_like(x), torch.ones_like(x),
                                   torch.ones_like(x), torch.full_like(x, float('nan')), .2)
    assert bad == 1
    with pytest.raises(ValueError, match='all ranks'):
        coordinated_scale(bound, bad)


def test_probe_boundary_failure_restores_rng_and_collects_flag(monkeypatch):
    from search_r1.baselines.loss_scaling import TransparentProbe
    model = torch.nn.Linear(1, 1)
    probe = TransparentProbe(model)
    rng = torch.get_rng_state().clone()
    with probe:
        torch.rand(10)
        def fail():
            raise RuntimeError('boundary failure')
        monkeypatch.setattr(probe, '_boundary', fail)
    assert probe.invalid
    assert torch.equal(rng, torch.get_rng_state())


def test_probe_flags_changed_buffer_dtype():
    from search_r1.baselines.loss_scaling import TransparentProbe
    model = torch.nn.Linear(1, 1)
    model.register_buffer('counter', torch.zeros(1))
    with TransparentProbe(model) as probe:
        model.counter = model.counter.double()
    assert probe.invalid
    assert model.counter.dtype == torch.float32


def test_probe_flags_and_restores_changed_buffer_device():
    from search_r1.baselines.loss_scaling import TransparentProbe
    model = torch.nn.Linear(1, 1)
    model.register_buffer('counter', torch.zeros(1))
    with TransparentProbe(model) as probe:
        model.counter = model.counter.to('meta')
    assert probe.invalid
    assert model.counter.device.type == 'cpu'
