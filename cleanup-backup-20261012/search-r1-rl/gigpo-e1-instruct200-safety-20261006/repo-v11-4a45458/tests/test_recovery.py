import random

import numpy as np
import pytest
import torch

from search_r1.baselines.recovery import (
    RecoverableSampler, atomic_torch_save, cpu_snapshot,
    rebuild_scheduler_at_logical_step, restore_rng, rng_state,
)


def test_data_continuation_and_next_epoch_match():
    sampler = RecoverableSampler(23)
    iterator = iter(sampler)
    head = [next(iterator) for _ in range(9)]
    restored = RecoverableSampler(23)
    restored.load_state_dict(sampler.state_dict())
    tail = list(iterator)
    assert list(restored) == tail
    assert sorted(head + tail) == list(range(23))
    assert list(restored) == list(sampler)


def test_data_rejects_changed_dataset_and_bad_permutation():
    sampler = RecoverableSampler(10)
    next(iter(sampler))
    state = sampler.state_dict()
    with pytest.raises(ValueError):
        RecoverableSampler(11).load_state_dict(state)
    state['order'][0] = state['order'][1]
    with pytest.raises(ValueError):
        RecoverableSampler(10).load_state_dict(state)


def test_scheduler_reconstruction_preserves_original_horizon():
    def build():
        parameter = torch.nn.Parameter(torch.tensor(1.))
        optimizer = torch.optim.AdamW([parameter], lr=1e-6)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: min(1., step / int(501 * .285)))
        return parameter, optimizer, scheduler
    p, original, schedule = build()
    for _ in range(200):
        p.grad = torch.ones_like(p)
        original.step()
        schedule.step()
    _, restored, reconstructed = build()
    rebuild_scheduler_at_logical_step(reconstructed, 200)
    assert reconstructed.state_dict() == schedule.state_dict()
    assert restored.param_groups[0]['lr'] == original.param_groups[0]['lr'] == 1e-6
    assert not restored.state  # optimizer history is explicitly rebuilt


def test_rng_recovery_matches_next_draws():
    state = rng_state()
    expected = (random.random(), np.random.random(), torch.rand(3))
    restore_rng(state)
    actual = (random.random(), np.random.random(), torch.rand(3))
    assert actual[:2] == expected[:2]
    assert torch.equal(actual[2], expected[2])


def test_snapshot_does_not_alias_live_optimizer_and_atomic_save(tmp_path):
    p = torch.nn.Parameter(torch.tensor([1., 2.]))
    optimizer = torch.optim.AdamW([p])
    p.grad = torch.ones_like(p)
    optimizer.step()
    state = cpu_snapshot(optimizer.state_dict())
    saved_moment = next(iter(state['state'].values()))['exp_avg'].clone()
    optimizer.step()
    assert torch.equal(next(iter(state['state'].values()))['exp_avg'], saved_moment)
    path = tmp_path / 'state.pt'
    receipt = atomic_torch_save(state, path)
    assert receipt['bytes'] == path.stat().st_size
    assert len(receipt['sha256']) == 64
    assert not (tmp_path / 'state.pt.partial').exists()
    restored = torch.optim.AdamW([torch.nn.Parameter(torch.tensor([1., 2.]))])
    restored.load_state_dict(torch.load(path, weights_only=False))
    assert torch.equal(next(iter(restored.state.values()))['exp_avg'], saved_moment)


def test_restored_adam_next_update_matches_uninterrupted():
    from search_r1.baselines.recovery import worker_training_state, load_worker_training_state
    def build():
        module = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(module.parameters(), lr=.002)
        schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.)
        return module, optimizer, schedule
    model, optimizer, schedule = build()
    x = torch.tensor([[.2, .5, .9]])
    def update(m, o, s):
        o.zero_grad()
        m(x).square().sum().backward()
        o.step()
        s.step()
    for _ in range(5):
        update(model, optimizer, schedule)
    rebuild_scheduler_at_logical_step(schedule, 205)
    state = worker_training_state(model, optimizer, schedule, 0, 1, 205)
    restored, new_optimizer, new_schedule = build()
    assert load_worker_training_state(state, restored, new_optimizer, new_schedule, 0, 1) == 205
    update(model, optimizer, schedule)
    update(restored, new_optimizer, new_schedule)
    for a, b in zip(model.parameters(), restored.parameters()):
        assert torch.equal(a, b)
    with pytest.raises(ValueError):
        load_worker_training_state(state, restored, new_optimizer, new_schedule, 0, 8)


def test_approval_rejects_standalone_pass_and_wrong_binding(tmp_path):
    import json
    from search_r1.baselines.recovery import wait_for_independent_approval
    manifest = dict(identity=dict(job_id='test', source='/original200'), sha256='abc')
    approval = tmp_path / 'approval.json'
    approval.write_text(json.dumps(dict(status='PASS')))
    with pytest.raises(RuntimeError, match='identity'):
        wait_for_independent_approval(manifest, str(approval), 1)
    approval.write_text(json.dumps(dict(job_id='test', source='/original200', logical_step=205,
                                        actual_new_updates=5, manifest_sha256='abc',
                                        review_path=str(tmp_path/'review.json'), review_sha256='wrong')))
    (tmp_path/'review.json').write_text(json.dumps(dict(status='PASS')))
    with pytest.raises(RuntimeError, match='SHA mismatch'):
        wait_for_independent_approval(manifest, str(approval), 1)


@pytest.mark.parametrize('corruption', ['scheduler_nan', 'rng_layout', 'optimizer_group'])
def test_corrupt_state_rejected_before_live_model_changes(corruption):
    from search_r1.baselines.recovery import worker_training_state, load_worker_training_state
    source = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(source.parameters(), lr=.002)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.)
    source(torch.ones(1,3)).sum().backward()
    optimizer.step()
    rebuild_scheduler_at_logical_step(schedule, 205)
    state = worker_training_state(source, optimizer, schedule, 0, 1, 205)
    target = torch.nn.Linear(3, 2)
    new_optimizer = torch.optim.AdamW(target.parameters(), lr=.002)
    new_schedule = torch.optim.lr_scheduler.LambdaLR(new_optimizer, lambda step: 1.)
    before = [p.detach().clone() for p in target.parameters()]
    if corruption == 'scheduler_nan':
        state['scheduler']['_last_lr'] = [float('nan')]
    elif corruption == 'rng_layout':
        state['rng']['cuda'] = torch.zeros(3, dtype=torch.uint8)
        state['rng']['cuda_device'] = -1
    else:
        state['optimizer']['param_groups'][0]['params'].pop()
    with pytest.raises(ValueError):
        load_worker_training_state(state, target, new_optimizer, new_schedule, 0, 1)
    assert all(torch.equal(old, new) for old, new in zip(before, target.parameters()))
    assert not new_optimizer.state
    assert new_schedule.last_epoch == 0
