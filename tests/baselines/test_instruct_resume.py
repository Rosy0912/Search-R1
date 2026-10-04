import copy
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from search_r1.baselines.instruct_resume import migration_diff, start_step


def configs(monkeypatch):
    monkeypatch.setenv('GIGPO_E1_ROOT', '/immutable-input-root')
    monkeypatch.setenv('GIGPO_E1_RUN', '/new-run')
    directory = Path(__file__).resolve().parents[2] / 'verl/trainer/config'
    with initialize_config_dir(config_dir=str(directory), version_base=None):
        old = compose(config_name='ppo_gigpo_instruct300')
        new = compose(config_name='ppo_gigpo_instruct300_resume75')
    # Reproduce the immutable prebuilt runtime override in the saved driver.
    old.baselines.holdout.output_dir = new.baselines.holdout.output_dir
    return old, new


def test_resume_identity_and_only_explicit_migration(monkeypatch):
    old, new = configs(monkeypatch)
    assert start_step(old) == 0
    assert start_step(new) == 75
    changes = migration_diff(OmegaConf.to_container(old, resolve=True),
                             OmegaConf.to_container(new, resolve=True))
    assert changes['trainer.acceptance_step'] == dict(old=5, new=80)
    assert changes['actor_rollout_ref.actor.joint_loss_scaling'] == dict(old=None, new=True)
    assert new.actor_rollout_ref.actor.optim.lr_warmup_steps == 57
    assert new.actor_rollout_ref.actor.ppo_mini_batch_size == 128
    assert new.actor_rollout_ref.ref.model_path == new.actor_rollout_ref.model.path


@pytest.mark.parametrize('path,value', [
    ('actor_rollout_ref.actor.optim.lr', 2e-6),
    ('actor_rollout_ref.actor.ppo_micro_batch_size', 8),
    ('actor_rollout_ref.actor.kl_loss_coef', .002),
    ('baselines.gigpo.gamma', .9),
    ('data.train_files', '/different-data.parquet'),
    ('actor_rollout_ref.ref.model_path', '/trained-actor'),
    ('actor_rollout_ref.rollout.temperature', .7),
])
def test_migration_rejects_unapproved_math_input_or_reference_changes(monkeypatch, path, value):
    old, new = configs(monkeypatch)
    source = OmegaConf.to_container(old, resolve=True)
    target = OmegaConf.to_container(new, resolve=True)
    parent = target
    parts = path.split('.')
    for part in parts[:-1]:
        parent = parent[part]
    parent[parts[-1]] = value
    with pytest.raises(ValueError, match='Unapproved'):
        migration_diff(source, target)


def test_resume_refuses_other_checkpoint_or_fresh_initialization(monkeypatch):
    _, cfg = configs(monkeypatch)
    for key, value in [('restore_checkpoint', '/old-base-step200'), ('restore_manifest_sha256', 'f' * 64)]:
        bad = copy.deepcopy(cfg)
        setattr(bad.trainer.instruct300, key, value)
        with pytest.raises(ValueError, match='identity'):
            start_step(bad)
    cfg.trainer.val_before_train = True
    with pytest.raises(ValueError, match='boundaries'):
        start_step(cfg)


def test_new_iterator_is_constructed_before_restoring_saved_driver_rng():
    import torch
    from torch.utils.data import DataLoader
    from search_r1.baselines.recovery import RecoverableSampler
    sampler = RecoverableSampler(169028, seed=42)
    iterator = iter(sampler)
    for _ in range(38400):
        next(iterator)
    state = sampler.state_dict()
    saved_rng = torch.get_rng_state().clone()
    loader = DataLoader(list(range(169028)), sampler=sampler, batch_size=512, num_workers=0)
    resumed_iterator = iter(loader)
    assert not torch.equal(saved_rng, torch.get_rng_state())
    sampler.load_state_dict(state)
    torch.set_rng_state(saved_rng)
    assert sampler.cursor == 38400
    expected = torch.tensor(state['order'][38400:38912])
    torch.testing.assert_close(next(resumed_iterator), expected)
    assert sampler.cursor == 38912
    assert torch.equal(saved_rng, torch.get_rng_state())
