"""Actual CPU Adam admission counterexamples; no GPU/calibration acceptance claim."""
import copy

import numpy as np
import pytest
import torch

from calibration_state_v1 import state_digest, assert_complete_pause_unchanged
from test_calibration_contract_v1 import fixture_workers


def fixture_driver():
    return dict(pid=4000, start_ticks='123', global_steps=26, actual_outer_updates=0,
        rng=dict(python=(3, (1, 2, 3), None), numpy=np.arange(8, dtype=np.uint32), torch=torch.arange(8, dtype=torch.uint8)),
        sampler=dict(cursor=13312, order=list(range(32)), generator=torch.arange(8, dtype=torch.uint8)),
        data_iterator=dict(base_seed=42, num_yielded=1),
        algorithm=dict(gamma=.95, omega=1.), fixed_batch=dict(old_log_probs=torch.ones(3, 4)))


def attempt_adam(before_workers, after_workers, before_driver, after_driver, parameter, optimizer):
    # The same admission function used immediately before the helper update RPC.
    assert_complete_pause_unchanged(before_workers, after_workers, before_driver, after_driver)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()


@pytest.mark.parametrize('field', ['pid', 'start_ticks', 'model_shards_sha256', 'local_shards_sha256',
    'buffers_sha256', 'optimizer_tensors_sha256', 'scheduler_step', 'learning_rates',
    'rng_tensors_sha256', 'complete_state_sha256'])
def test_real_adam_remains_unwritten_when_worker_restarts_or_any_complete_state_changes(field):
    before, driver = fixture_workers(), fixture_driver()
    after = copy.deepcopy(before)
    old = after[4][field]
    after[4][field] = ('0' * 64 if isinstance(old, str) and len(old) == 64 else
                       str(int(old) + 1) if field == 'start_ticks' else
                       [1e-6] if field == 'learning_rates' else old + 1)
    parameter = torch.nn.Parameter(torch.tensor([2.]))
    optimizer = torch.optim.Adam([parameter], lr=1e-6)
    with pytest.raises(ValueError, match='zero Adam'):
        attempt_adam(before, after, driver, copy.deepcopy(driver), parameter, optimizer)
    assert not optimizer.state and parameter.item() == 2.


@pytest.mark.parametrize('mutation', ['pid', 'start_ticks', 'rng_python', 'rng_numpy', 'rng_torch',
    'cursor', 'permutation', 'sampler_rng', 'iterator_seed', 'iterator_progress',
    'algorithm', 'fixed_old', 'actual_update_count'])
def test_real_adam_remains_unwritten_when_sampling_after_driver_state_changes(mutation):
    workers, before = fixture_workers(), fixture_driver()
    after = copy.deepcopy(before)
    if mutation in ('pid', 'actual_update_count'):
        after['pid' if mutation == 'pid' else 'actual_outer_updates'] += 1
    elif mutation == 'start_ticks':
        after['start_ticks'] = '124'
    elif mutation == 'rng_python':
        after['rng']['python'] = (3, (1, 2, 4), None)
    elif mutation in ('rng_numpy', 'rng_torch'):
        after['rng']['numpy' if mutation == 'rng_numpy' else 'torch'][0] += 1
    elif mutation == 'cursor':
        after['sampler']['cursor'] += 512
    elif mutation == 'permutation':
        after['sampler']['order'][:2] = [1, 0]
    elif mutation == 'sampler_rng':
        after['sampler']['generator'][0] += 1
    elif mutation in ('iterator_seed', 'iterator_progress'):
        after['data_iterator']['base_seed' if mutation == 'iterator_seed' else 'num_yielded'] += 1
    elif mutation == 'algorithm':
        after['algorithm']['omega'] = .9
    else:
        after['fixed_batch']['old_log_probs'][0, 0] += .01
    parameter = torch.nn.Parameter(torch.tensor([2.]))
    optimizer = torch.optim.Adam([parameter], lr=1e-6)
    with pytest.raises(ValueError, match='zero Adam'):
        attempt_adam(workers, copy.deepcopy(workers), before, after, parameter, optimizer)
    assert not optimizer.state and parameter.item() == 2.


def test_exact_post_sampling_state_allows_one_real_adam_without_restoring_rng():
    workers, driver = fixture_workers(), fixture_driver()
    parameter = torch.nn.Parameter(torch.tensor([2.]))
    optimizer = torch.optim.Adam([parameter], lr=1e-6)
    expected = state_digest(driver)
    attempt_adam(workers, copy.deepcopy(workers), driver, copy.deepcopy(driver), parameter, optimizer)
    assert optimizer.state[parameter]['step'] == 1 and parameter.item() != 2.
    assert state_digest(driver) == expected


def test_digest_preserves_type_shape_keys_and_object_array_values():
    assert state_digest(dict(x=1, y=torch.arange(4))) == state_digest(dict(y=torch.arange(4), x=1))
    assert state_digest(torch.arange(4)) != state_digest(torch.arange(4).reshape(2, 2))
    assert state_digest([1, 2]) != state_digest((1, 2))
    value = np.array([{'raw_prompt': ['question']}], dtype=object)
    assert state_digest(value) == state_digest(copy.deepcopy(value))
    changed = copy.deepcopy(value)
    changed[0]['raw_prompt'][0] = 'other'
    assert state_digest(value) != state_digest(changed)
