"""Independent regression review: deployment refusal and empty OPSD momentum."""
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import torch
from tensordict import TensorDict
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from verl import DataProto
from search_r1.baselines.preflight import check_selected_gpus
from search_r1.baselines.actor import DistillationActor


def test_idle_boundary_with_existing_process():
    inventory = {'0': dict(uuid='GPU-a', memory_used=15., memory_total=100., utilization=15.)}
    assert len(check_selected_gpus(['0'], inventory, set())) == 1
    for field in ('memory_used', 'utilization'):
        busy = {'0': dict(inventory['0'], **{field: 15.01})}
        try:
            check_selected_gpus(['0'], busy, set())
            raise AssertionError('busy GPU accepted')
        except RuntimeError:
            pass
    assert len(check_selected_gpus(['0'], inventory, {'GPU-a'})) == 1
    for field, value in [('memory_total', 0.), ('memory_used', float('nan')), ('utilization', float('nan'))]:
        invalid = {'0': dict(inventory['0'], **{field: value})}
        try:
            check_selected_gpus(['0'], invalid, {'GPU-a'})
            raise AssertionError('unknown occupancy accepted')
        except RuntimeError:
            pass
    try:
        check_selected_gpus(['0', 'GPU-a'], inventory, set())
        raise AssertionError('duplicate physical GPU accepted')
    except ValueError:
        pass


def test_opsd_global_empty_minibatch_preserves_adam_momentum_and_weights():
    actor = DistillationActor.__new__(DistillationActor)
    actor.actor_module = torch.nn.Linear(2, 2)
    actor.actor_optimizer = torch.optim.AdamW(actor.actor_module.parameters(), lr=.01, weight_decay=.1)
    # Prime nonzero optimizer history: skipping only backward would still change weights.
    actor.actor_module(torch.ones(1, 2)).sum().backward()
    actor.actor_optimizer.step()
    actor.actor_optimizer.zero_grad()
    before = [p.detach().clone() for p in actor.actor_module.parameters()]
    steps = [state['step'].clone() for state in actor.actor_optimizer.state.values()]
    actor.config = SimpleNamespace(distillation=SimpleNamespace(), ppo_epochs=2,
                                   ppo_mini_batch_size=2, ppo_micro_batch_size=1)
    actor.teacher = SimpleNamespace(parameter_count=5, scored_tokens=0)
    def forbidden():
        raise AssertionError('globally empty batch must not call optimizer step')
    actor._optimizer_step = forbidden
    data = DataProto(batch=TensorDict({'responses': torch.zeros(2, 3, dtype=torch.long),
                                      'loss_mask': torch.zeros(2, 3)}, batch_size=[2]),
                     non_tensor_batch={'reward_model': np.array([{}, {}], dtype=object)})
    result = actor.update_policy(data)
    assert result['actor/optimizer_steps'] == 0
    assert result['actor/skipped_empty_minibatches'] == 2
    for old, p in zip(before, actor.actor_module.parameters()):
        torch.testing.assert_close(old, p)
    for old, state in zip(steps, actor.actor_optimizer.state.values()):
        torch.testing.assert_close(old, state['step'])


def test_final_allocation_and_real_advisory_lease_are_required(monkeypatch):
    import fcntl
    import json
    import os
    import uuid
    from search_r1.baselines import preflight
    uid = 'GPU-review-test-' + uuid.uuid4().hex
    inventory = {'0': dict(uuid=uid, memory_used=0., memory_total=100., utilization=0.)}
    monkeypatch.setattr(preflight, 'gpu_snapshot', lambda: (inventory, set()))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    monkeypatch.delenv('SEARCH_R1_GPU_PREFLIGHT', raising=False)
    monkeypatch.delenv('SEARCH_R1_GPU_LEASES', raising=False)
    config = SimpleNamespace(trainer=SimpleNamespace(nnodes=1, n_gpus_per_node=1))
    import pytest
    with pytest.raises(ValueError, match='through'):
        preflight.validate_gpu_lease(config)
    config.trainer.n_gpus_per_node = 2
    with pytest.raises(ValueError, match='does not match'):
        preflight.validate_gpu_lease(config)
    config.trainer.n_gpus_per_node = 1
    config.trainer.nnodes = 2
    with pytest.raises(ValueError, match='one node'):
        preflight.validate_gpu_lease(config)
    config.trainer.nnodes = 1
    parent = Path('/tmp') / ('search-r1-gpu-leases-' + str(os.getuid()))
    parent.mkdir(mode=0o700, exist_ok=True)
    path = parent / (uid + '.lock')
    fd = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        monkeypatch.setenv('SEARCH_R1_GPU_PREFLIGHT', '1')
        monkeypatch.setenv('SEARCH_R1_GPU_LEASES', json.dumps([dict(uuid=uid, fd=fd)]))
        preflight.validate_gpu_lease(config)
        fcntl.flock(fd, fcntl.LOCK_UN)
        with pytest.raises(ValueError, match='not held'):
            preflight.validate_gpu_lease(config)
    finally:
        os.close(fd)
        path.unlink()
