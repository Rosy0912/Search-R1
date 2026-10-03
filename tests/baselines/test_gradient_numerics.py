"""Real loss/gradient counterexamples; no GPU/model budget modifications."""
import copy
import pytest
import torch
from verl.trainer.ppo.core_algos import kl_penalty
from search_r1.baselines.gradient_norm import clip_grad_norm64, _norm64


def test_saturated_kl_values_and_derivatives_match_fp64_oracle():
    points = torch.cat((torch.linspace(-12, 5, 400), torch.tensor([-10000., -11., 4., 80., 90., 10000.])))
    current = (-points).requires_grad_()
    actual = kl_penalty(current, torch.zeros_like(current), 'low_var_kl')
    oracle_current = (-points.double()).requires_grad_()
    oracle_delta = -oracle_current
    oracle = (oracle_delta.exp() - oracle_delta - 1).clamp(-10, 10)
    # x10000 itself overflows even FP64; the saturated analytic oracle is10.
    oracle = torch.where(oracle_delta > 700, torch.full_like(oracle, 10), oracle)
    torch.testing.assert_close(actual.double(), oracle, rtol=1e-5, atol=1e-6)
    actual.sum().backward()
    finite_oracle_current = (-points.double()).clamp(min=-700).detach().requires_grad_()
    delta = -finite_oracle_current
    (delta.exp()-delta-1).clamp(-10,10).sum().backward()
    torch.testing.assert_close(current.grad.double(), finite_oracle_current.grad, rtol=1e-5, atol=1e-6)
    assert torch.isfinite(current.grad).all()


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), float('-inf')])
def test_kl_does_not_hide_nonfinite_input(bad):
    with pytest.raises(ValueError, match='Nonfinite'):
        kl_penalty(torch.tensor([bad]), torch.zeros(1), 'low_var_kl')


def test_finite_gradients_with_overflowing_fp32_norm_clip_correctly():
    model = torch.nn.Linear(500, 1, bias=False)
    model.weight.grad = torch.full_like(model.weight, 1.6e20)
    assert torch.isfinite(model.weight.grad).all()
    assert not torch.isfinite(torch.linalg.vector_norm(model.weight.grad.float()))
    expected_norm = torch.linalg.vector_norm(model.weight.grad.double())
    expected = model.weight.grad.clone() * (1. / (expected_norm + 1e-6)).float()
    actual_norm = clip_grad_norm64(model, 1.)
    torch.testing.assert_close(actual_norm, expected_norm)
    torch.testing.assert_close(model.weight.grad, expected)


def test_normal_clipping_matches_native_and_does_not_change_rng():
    torch.manual_seed(20261004)
    for dtype in (torch.float32, torch.bfloat16, torch.float64):
        model = torch.nn.Linear(8, 4).to(dtype)
        for p in model.parameters():
            p.grad = torch.randn_like(p)
        native = copy.deepcopy(model)
        for p, q in zip(model.parameters(), native.parameters()):
            q.grad = p.grad.clone()
        before = torch.get_rng_state().clone()
        norm_oracle = torch.linalg.vector_norm(torch.cat([p.grad.reshape(-1).double() for p in model.parameters()]))
        coefficient = 1. / (norm_oracle + 1e-6)
        exact_grads = [p.grad.clone()*coefficient.clamp(max=1).to(dtype) for p in model.parameters()]
        actual = clip_grad_norm64(model, 1.)
        expected = torch.nn.utils.clip_grad_norm_(native.parameters(), 1.)
        torch.testing.assert_close(actual, norm_oracle, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(actual.float(), expected.float(), rtol=.02 if dtype==torch.bfloat16 else 1e-5, atol=1e-6)
        for p, q, exact in zip(model.parameters(), native.parameters(), exact_grads):
            torch.testing.assert_close(p.grad, exact, rtol=1e-12 if dtype==torch.float64 else 0, atol=1e-12 if dtype==torch.float64 else 0)
            torch.testing.assert_close(p.grad, q.grad, rtol=.02 if dtype==torch.bfloat16 else 1e-5, atol=1e-6)
        torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)


def test_nonfinite_gradients_are_refused_before_mutation():
    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.grad = torch.tensor([[float('nan'), 1.]])
    before = model.weight.grad.clone()
    with pytest.raises(ValueError, match='Nonfinite'):
        clip_grad_norm64(model, 1.)
    torch.testing.assert_close(model.weight.grad, before, equal_nan=True)


def test_chunked_norm_matches_full_and_empty_norm():
    p = torch.nn.Parameter(torch.zeros(100))
    p.grad = torch.linspace(-10, 10, 100)
    torch.testing.assert_close(_norm64([p], 'cpu', chunk_size=7), torch.linalg.vector_norm(p.grad.double()))
    assert _norm64([], 'cpu').item()==0


def test_actual_actor_optimizer_step_keeps_finite_large_gradients():
    from types import SimpleNamespace
    from verl.workers.actor.dp_actor import DataParallelPPOActor
    module = torch.nn.Linear(2, 1, bias=False)
    module.weight.grad = torch.full_like(module.weight, 1.e20)
    before = module.weight.detach().clone()
    actor = SimpleNamespace(config=SimpleNamespace(grad_clip=1.), actor_module=module,
                            actor_optimizer=torch.optim.SGD(module.parameters(), lr=1.e-3))
    norm = DataParallelPPOActor._optimizer_step(actor)
    assert torch.isfinite(norm) and norm > 1.e20
    assert torch.isfinite(module.weight.grad).all() and module.weight.grad.abs().sum() > 0
    assert torch.isfinite(module.weight).all() and not torch.equal(module.weight, before)
