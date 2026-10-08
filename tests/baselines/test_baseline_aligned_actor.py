"""Execute the aligned actor loop with real AdamW and synthetic CPU rollouts."""
from types import SimpleNamespace
import math
import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDictBase
from verl import DataProto
from verl.workers.actor.dp_actor import DataParallelPPOActor
from search_r1.baselines.policy_safety import BaselineAlignedUpdateFault
from search_r1.baselines.objectives import weighted_policy_loss


def aligned_actor(monkeypatch, outer=50, delta=0., fail_after_writes=None, sampling_nan=False):
    import search_r1.baselines.policy_safety as safety
    import search_r1.baselines.distillation as reductions
    import search_r1.baselines.objectives as objectives
    import search_r1.baselines.fresh200_contract as contract
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 0)
    monkeypatch.setattr(TensorDictBase, 'cuda', lambda self, *a, **kw: self)
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.distributed, 'get_world_size', lambda *a: 8)
    monkeypatch.setattr(safety, 'distributed', lambda *a: False)
    monkeypatch.setattr(reductions, 'global_token_count', lambda mask: (mask.sum(), 1))
    monkeypatch.setattr(objectives, 'row_denominator', lambda rows, device: torch.tensor(float(rows), device=device))
    def forbidden(*args, **kwargs):
        raise AssertionError('Aligned actor reached legacy sampling or threshold admission')
    monkeypatch.setattr(contract, 'verify_fixed_policy_safety_artifact', forbidden)
    monkeypatch.setattr(contract, 'verify_sampling_policy_binding', forbidden)
    config = OmegaConf.create(dict(baseline_aligned=True, baseline_method='bapo',
        loss_reduction='token_mean', ppo_mini_batch_size=16, ppo_micro_batch_size=2,
        ppo_epochs=1, state_masking=True, use_dynamic_bsz=False, use_kl_loss=True,
        kl_loss_type='low_var_kl', kl_loss_coef=.001, entropy_coeff=0., clip_ratio=.2,
        grad_clip=1., policy_safety=dict(enabled=False, target_old_policy_kl=.00001, negative_tail_limit=.00001)))
    one = torch.ones(320, 1, dtype=torch.float64)
    data = DataProto.from_dict(dict(responses=one.long(), input_ids=one.long(),
        attention_mask=one.long(), position_ids=one.long(), loss_mask=one.clone(),
        old_log_probs=-one, advantages=-one, ref_log_prob=-one),
        meta_info=dict(temperature=1., actual_outer=outer, attempt_id=1, rollout_identity='aligned-cpu'))
    if sampling_nan:
        data.batch['sampling_log_probs'] = torch.full_like(one, float('nan'))
        data.batch['sampling_mask'] = one.clone()
    model = torch.nn.Linear(1, 1, bias=False).double()
    model.weight.data.fill_(delta)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    def forward(micro_batch, temperature):
        current = micro_batch['old_log_probs'] + model.weight.reshape(())
        if fail_after_writes is not None and int(optimizer.state.get(model.weight, {}).get('step', 0)) >= fail_after_writes:
            current = current * float('nan')
        return torch.zeros_like(current), current
    actor = SimpleNamespace(config=config, actor_module=model, actor_optimizer=optimizer,
        _forward_micro_batch=forward, ulysses_sequence_parallel_size=1)
    actor._optimizer_step = lambda: DataParallelPPOActor._optimizer_step(actor)
    return actor, data, model, optimizer


@pytest.mark.parametrize('outer', [1, 49, 50, 200])
def test_every_outer_performs_twenty_actual_adam_writes(monkeypatch, outer):
    actor, data, model, optimizer = aligned_actor(monkeypatch, outer, delta=math.log(4), sampling_nan=True)
    metrics = DataParallelPPOActor.update_policy(actor, data)
    event = metrics['fresh200_update_state']
    assert event['baseline_aligned'] and event['status'] == 'COMPLETE'
    assert event['expected_optimizer_steps'] == event['optimizer_steps'] == 20
    assert event['processed_opportunities'] == 20 and event['skipped_opportunities'] == 0
    assert optimizer.state[model.weight]['step'].item() == 20
    assert metrics['actor/kl_coef'] == .001
    reports = metrics['fresh200_policy_monitor']
    assert all(r['diagnostic_only'] and r['denominator_role'] == 'frozen_hf_old_policy' for r in reports)
    assert any('old_policy_kl' in r['reasons'] and 'negative_advantage_tail' in r['reasons'] for r in reports)
    assert all(r['sampling_policy_kl'] is None for r in reports)
    assert any(not r['raw_sampler_token_transport_diagnostic']['selected_logprobs_finite'] for r in reports)


@pytest.mark.parametrize('writes', [0, 3, 19])
def test_nonfinite_is_fatal_and_retains_previous_writes(monkeypatch, writes):
    actor, data, model, optimizer = aligned_actor(monkeypatch, fail_after_writes=writes)
    with pytest.raises(BaselineAlignedUpdateFault) as caught:
        DataParallelPPOActor.update_policy(actor, data)
    event = caught.value.event
    assert event['baseline_aligned'] and event['fatal'] and event['status'] == 'INFRASTRUCTURE_FAULT'
    assert not event['update_complete'] and not event['attempt_complete']
    assert event['optimizer_steps'] == event['confirmed_previous_optimizer_steps'] == writes
    assert event['current_state_requires_native_save']
    assert model.weight.grad is None and torch.isfinite(model.weight).all()
    assert int(optimizer.state.get(model.weight, {}).get('step', 0)) == writes


def test_aligned_ordinary_ppo_has_no_dualclip_or_logratio_bound():
    current = torch.tensor([[math.log(4.)]], dtype=torch.float64, requires_grad=True)
    one = torch.ones_like(current)
    loss, _, _ = weighted_policy_loss(torch.zeros_like(current), current, -one, one, one,
                                      .2, one.sum(), baseline_aligned=True)
    assert loss.item() == pytest.approx(4.)
    assert torch.autograd.grad(loss, current)[0].item() == pytest.approx(4.)
    current = torch.tensor([[21.]], dtype=torch.float64, requires_grad=True)
    loss, _, _ = weighted_policy_loss(torch.zeros_like(current), current, -one, one, one,
                                      .2, one.sum(), baseline_aligned=True)
    assert loss.item() == pytest.approx(math.exp(21.))
    assert torch.autograd.grad(loss, current)[0].item() == pytest.approx(math.exp(21.))


@pytest.mark.parametrize('fault', ['ratio_overflow', 'reference_nan', 'gradient_nan', 'candidate_nonfinite'])
def test_actual_numeric_admission_refuses_live_write_and_raises(monkeypatch, fault):
    actor, data, model, optimizer = aligned_actor(monkeypatch, delta=1000. if fault == 'ratio_overflow' else 0.)
    if fault == 'reference_nan':
        data.batch['ref_log_prob'][0, 0] = float('nan')
    elif fault == 'gradient_nan':
        model.weight.register_hook(lambda gradient: torch.full_like(gradient, float('nan')))
    elif fault == 'candidate_nonfinite':
        import search_r1.baselines.optimizer_transaction as transaction
        monkeypatch.setattr(transaction, 'transactional_adam_step',
                            lambda *a: dict(committed=False, reason='nonfinite_candidate_parameters_or_optimizer_state'))
    previous = model.weight.detach().clone()
    with pytest.raises(BaselineAlignedUpdateFault) as caught:
        DataParallelPPOActor.update_policy(actor, data)
    event = caught.value.event
    assert event['optimizer_steps'] == 0 and event['fatal']
    assert event['status'] == 'INFRASTRUCTURE_FAULT' and not event['update_complete']
    assert not optimizer.state and model.weight.grad is None
    torch.testing.assert_close(model.weight, previous)


def test_actual_aligned_matches_archive_unequal_length_micro_token_mean(monkeypatch):
    actor, data, model, optimizer = aligned_actor(monkeypatch, delta=.31)
    rows, width = 320, 4
    mask = torch.zeros(rows, width, dtype=torch.float64)
    for row in range(rows):
        mask[row, :row % 5] = 1
    advantages = torch.linspace(-1.7, 2.1, rows * width, dtype=torch.float64).reshape(rows, width)
    data.batch = __import__('tensordict').TensorDict(dict(
        responses=torch.zeros(rows, width, dtype=torch.long), input_ids=torch.zeros(rows, width + 1, dtype=torch.long),
        attention_mask=torch.ones(rows, width + 1, dtype=torch.long), position_ids=torch.zeros(rows, width + 1, dtype=torch.long),
        loss_mask=mask, old_log_probs=torch.full_like(mask, -3.),
        ref_log_prob=torch.full_like(mask, -2.7), advantages=advantages), batch_size=[rows])
    gradients, expected = [], []
    def forward(micro_batch, temperature):
        current = micro_batch['old_log_probs'] + model.weight.reshape(()) * (micro_batch['advantages'] * .1 + .2)
        return torch.zeros_like(current), current
    actor._forward_micro_batch = forward
    def step():
        mini = data.batch[len(gradients) * 16:(len(gradients) + 1) * 16]
        p = model.weight.detach().reshape(()).clone().requires_grad_()
        objective = p * 0.
        for micro in mini.split(2):
            valid = micro['loss_mask']
            delta = p * (micro['advantages'] * .1 + .2)
            ratio = delta.exp()
            pg = torch.maximum(-micro['advantages'] * ratio,
                               -micro['advantages'] * ratio.clamp(.8, 1.2))
            ref_difference = micro['ref_log_prob'] - (micro['old_log_probs'] + delta)
            reference_kl = (ref_difference.exp() - ref_difference - 1).clamp(-10, 10)
            objective = objective + ((pg + .001 * reference_kl) * valid).sum() / valid.sum().clamp_min(1) / 8
        reference_gradient = torch.autograd.grad(objective, p)[0]
        gradients.append(model.weight.grad.detach().reshape(()).clone())
        expected.append(reference_gradient)
        torch.testing.assert_close(gradients[-1], reference_gradient, atol=1e-12, rtol=1e-12)
        return DataParallelPPOActor._optimizer_step(actor)
    actor._optimizer_step = step
    metrics = DataParallelPPOActor.update_policy(actor, data)
    assert len(gradients) == 20 and optimizer.state[model.weight]['step'].item() == 20
    assert metrics['fresh200_update_state']['status'] == 'COMPLETE'


@pytest.mark.parametrize('fail_after_writes', [None, 3, 20])
def test_aligned_scans_only_outer_boundaries_and_fault_accounting(monkeypatch, fail_after_writes):
    actor, data, model, optimizer = aligned_actor(monkeypatch, fail_after_writes=fail_after_writes)
    original_forward = actor._forward_micro_batch
    calls = []
    def counted(micro_batch, temperature):
        calls.append(model.training)
        return original_forward(micro_batch, temperature)
    actor._forward_micro_batch = counted
    if fail_after_writes is None:
        result = DataParallelPPOActor.update_policy(actor, data)
        assert optimizer.state[model.weight]['step'].item() == 20
        assert result['fresh200_update_state']['status'] == 'COMPLETE'
        reports = result['fresh200_policy_monitor']
        assert [r['stage'] for r in reports] == ['before_adam', 'after_adam']
        assert all(r['monitoring_scope'] == 'full_frozen_rollout' for r in reports)
        assert calls.count(False) == 320 and calls.count(True) == 160
    else:
        with pytest.raises(BaselineAlignedUpdateFault) as caught:
            DataParallelPPOActor.update_policy(actor, data)
        event = caught.value.event
        assert event['optimizer_steps'] == fail_after_writes
        assert not event['update_complete'] and not event['attempt_complete']
        assert optimizer.state[model.weight]['step'].item() == fail_after_writes
        if fail_after_writes == 3:
            # Three full training minis and the first actual forward of mini4
            # reject numerically; no diagnostic scan occurs between Adam writes.
            assert calls.count(False) == 160 and calls.count(True) == 25
            assert event['processed_opportunities'] == 4
        else:
            # A nonfinite post-outer diagnostic retains all20 writes but still
            # refuses to mark the outer complete or advance its scheduler.
            assert calls.count(False) == 320 and calls.count(True) == 160
            assert event['processed_opportunities'] == 20


def test_ordinary_ratio_overflow_is_collectively_refused_before_exponent(monkeypatch):
    from search_r1.baselines.policy_safety import SkipOptimizerOpportunity
    current = torch.tensor([[1000.]], dtype=torch.float64, requires_grad=True)
    one = torch.ones_like(current)
    def forbidden_exp(self):
        raise AssertionError('Overflow admission reached exponentiation')
    monkeypatch.setattr(torch.Tensor, 'exp', forbidden_exp)
    with pytest.raises(SkipOptimizerOpportunity, match='exponent_overflow'):
        weighted_policy_loss(torch.zeros_like(current), current, -one, one, one,
                             .2, one.sum(), baseline_aligned=True)
