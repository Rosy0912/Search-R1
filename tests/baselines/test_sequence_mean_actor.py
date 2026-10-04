"""Actual actor/DataProto reductions, with real Gloo gradient averaging."""
from types import SimpleNamespace
import json
import torch
import pytest
from omegaconf import OmegaConf
from tensordict import TensorDictBase
from verl import DataProto
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.trainer.ppo.core_algos import kl_penalty


def fixture(rows=16, empty=False):
    mask = torch.zeros(rows, 7, dtype=torch.float64)
    for i in range(rows):
        if not empty:
            mask[i, :i % 8] = 1
    old = torch.full_like(mask, -3.)
    return DataProto.from_dict({
        'responses': torch.zeros_like(mask, dtype=torch.long),
        'input_ids': torch.zeros(rows, 8, dtype=torch.long),
        'attention_mask': torch.ones(rows, 8, dtype=torch.long),
        'position_ids': torch.zeros(rows, 8, dtype=torch.long),
        'loss_mask': mask, 'old_log_probs': old,
        'ref_log_prob': old + torch.linspace(-.8, .8, rows * 7).reshape_as(mask),
        'advantages': torch.linspace(-2, 2, rows * 7).reshape_as(mask),
        }, meta_info={'temperature': 1.})


def actor(data, mini, micro, klcoef=.001, entropy=.03, reduction='sequence_mean', distributed=False,
          dynamic=False, adamw=False):
    model = torch.nn.Linear(1, 1, bias=False).double()
    model.weight.data.fill_(.31)
    config = OmegaConf.create(dict(ppo_mini_batch_size=mini, ppo_micro_batch_size=micro,
        ppo_epochs=1, state_masking=True, use_kl_loss=True, use_dynamic_bsz=dynamic,
        ppo_max_token_len_per_gpu=1000,
        baseline_method='bapo', loss_reduction=reduction, grad_clip=1.e9,
        kl_loss_type='low_var_kl', kl_loss_coef=klcoef, clip_ratio=.2, entropy_coeff=entropy))
    optimizer = (torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.1)
                 if adamw else torch.optim.SGD(model.parameters(), lr=0.))
    gradients = []
    def forward(micro_batch, temperature):
        old = micro_batch['old_log_probs']
        # Row identity survives arbitrary DataProto permutation and micro split.
        features = micro_batch['advantages'] * .4 + .1
        current = torch.full_like(old, -3.) + model.weight.reshape(()) * features
        return current.square() * .05, current
    def step():
        if distributed:
            torch.distributed.all_reduce(model.weight.grad)
            model.weight.grad /= torch.distributed.get_world_size()
        gradients.append(model.weight.grad.detach().clone())
        return DataParallelPPOActor._optimizer_step(instance)
    instance = SimpleNamespace(config=config, actor_module=model, actor_optimizer=optimizer,
                               _forward_micro_batch=forward, _optimizer_step=step,
                               ulysses_sequence_parallel_size=1)
    metrics = DataParallelPPOActor.update_policy(instance, data)
    metrics['test/final_weight'] = model.weight.item()
    return gradients, metrics


def oracle(data, klcoef=.001, entropy=.03):
    p = torch.tensor(.31, dtype=torch.float64, requires_grad=True)
    b = data.batch
    mask = b['loss_mask']
    features = b['advantages'] * .4 + .1
    current = b['old_log_probs'] + p * features
    delta = current - b['old_log_probs']
    ratio = delta.exp()
    pg = torch.maximum(-b['advantages'] * ratio, -b['advantages'] * ratio.clamp(.8, 1.2))
    kld = kl_penalty(torch.where(mask.bool(), current, 0.), torch.where(mask.bool(), b['ref_log_prob'], 0.), 'low_var_kl')
    value = pg + klcoef * kld - entropy * current.square() * .05
    row = torch.where(mask.bool(), value, 0.).sum(-1) / mask.sum(-1).clamp_min(1)
    row.mean().backward()
    return p.grad


@pytest.mark.parametrize('klcoef,entropy', [(0., 0.), (.001, 0.), (0., .03), (.001, .03)])
def test_actual_actor_micro_permutation_and_tail(monkeypatch, klcoef, entropy):
    monkeypatch.setattr(TensorDictBase, 'cuda', lambda self, *a, **kw: self)
    data = fixture(18)
    for micro in (1, 2, 4, 8):
        gradients, metrics = actor(data, 16, micro, klcoef, entropy)
        expected = [oracle(DataProto(batch=b), klcoef, entropy) for b in data.batch.split(16)]
        for actual, ref in zip(gradients, expected):
            torch.testing.assert_close(actual.reshape(()), ref, atol=2e-7, rtol=2e-7)
        assert metrics['actor/optimizer_steps'] == 2
        assert metrics['actor/skipped_empty_minibatches'] == 0
    from search_r1.baselines.rollout import select_rows
    permutation = torch.randperm(16).tolist()
    permuted = select_rows(fixture(), permutation)
    grad, _ = actor(permuted, 16, 2, klcoef, entropy)
    torch.testing.assert_close(grad[0].reshape(()), oracle(fixture(), klcoef, entropy), atol=2e-7, rtol=2e-7)


def test_empty_sequence_steps_but_legacy_skips(monkeypatch):
    monkeypatch.setattr(TensorDictBase, 'cuda', lambda self, *a, **kw: self)
    gradients, metrics = actor(fixture(empty=True), 16, 2)
    assert len(gradients) == 1 and gradients[0].item() == 0
    assert metrics['actor/optimizer_steps'] == 1
    gradients, metrics = actor(fixture(empty=True), 16, 2, reduction='token_mean')
    assert gradients == [] and metrics['actor/skipped_empty_minibatches'] == 1
    _, metrics = actor(fixture(empty=True), 16, 2, adamw=True)
    assert metrics['test/final_weight'] == pytest.approx(.31 * .999)


def test_ragged_dynamic_local_and_metrics(monkeypatch):
    monkeypatch.setattr(TensorDictBase, 'cuda', lambda self, *a, **kw: self)
    import verl.workers.actor.dp_actor as module
    monkeypatch.setattr(module, 'rearrange_micro_batches',
                        lambda batch, max_token_len: ([batch[:3], batch[3:10], batch[10:]], None))
    data = fixture()
    grad, ragged = actor(data, 16, 2, dynamic=True)
    refgrad, full = actor(data, 16, 16)
    torch.testing.assert_close(grad[0], refgrad[0], atol=2e-7, rtol=2e-7)
    for key in ('actor/pg_loss', 'actor/kl_loss', 'actor/entropy_loss', 'actor/ppo_kl', 'actor/pg_clipfrac'):
        assert ragged[key][0] == pytest.approx(full[key][0], abs=2e-7)


def test_masked_nonfinite_and_valid_rejection(monkeypatch):
    monkeypatch.setattr(TensorDictBase, 'cuda', lambda self, *a, **kw: self)
    data = fixture()
    for key in ('old_log_probs', 'ref_log_prob'):
        data.batch[key][data.batch['loss_mask'] == 0] = float('nan')
    grads, _ = actor(data, 16, 2)
    assert torch.isfinite(grads[0]).all()
    data.batch['old_log_probs'][1, 0] = float('nan')
    with pytest.raises(ValueError, match='Nonfinite policy'):
        actor(data, 16, 2)


def gloo_worker(rank, rendezvous, output):
    torch.distributed.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=2)
    TensorDictBase.cuda = lambda self, *a, **kw: self
    data = fixture()
    from search_r1.baselines.rollout import select_rows
    results = []
    for make_rank_empty in (False, True):
        global_data = fixture()
        if make_rank_empty:
            global_data.batch['loss_mask'][:8] = 0
        local = select_rows(global_data, list(range(rank * 8, (rank + 1) * 8)))
        for micro in (1, 2, 4, 8):
            grads, metrics = actor(local, 8, micro, distributed=True)
            torch.testing.assert_close(grads[0].reshape(()), oracle(global_data), atol=2e-7, rtol=2e-7)
            assert metrics['actor/optimizer_steps'] == 1
            results.append({'empty_rank': make_rank_empty, 'micro': micro, 'gradient': grads[0].item()})
    empty_grads, _ = actor(select_rows(fixture(empty=True), list(range(rank * 8, (rank + 1) * 8))), 8, 2, distributed=True)
    assert empty_grads[0].item() == 0
    with pytest.raises(ValueError, match='fixed micro batches'):
        actor(select_rows(data, list(range(rank * 8, (rank + 1) * 8))), 8, 2, dynamic=True)
    torch.distributed.destroy_process_group()
    with open(output + str(rank) + '.json', 'w') as f:
        json.dump(results, f)


def test_real_two_rank_gloo(tmp_path):
    torch.multiprocessing.spawn(gloo_worker, args=(str(tmp_path / 'rendezvous'), str(tmp_path / 'rank')), nprocs=2, join=True)
    assert len(json.loads((tmp_path / 'rank0.json').read_text())) == 8
