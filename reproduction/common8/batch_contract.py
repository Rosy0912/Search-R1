"""Check the actual worker config before and after global-to-rank normalization."""
def independent_worker_config(config):
    # Colocated actor and reference constructors receive the same object.
    # Never let one role's in-place normalization change another role's input.
    import copy
    return copy.deepcopy(config)


def check_global(config, world_size, sequence_parallel):
    assert world_size == 8 and sequence_parallel == 1, 'common8 requires eight data-parallel ranks'
    assert config.rollout.n == 1 and config.rollout.n_agent == 5
    assert config.actor.ppo_mini_batch_size == 128
    assert config.actor.ppo_micro_batch_size == 16
    assert config.actor.ppo_epochs == 1
    assert config.rollout.log_prob_micro_batch_size == 32
    assert config.ref.log_prob_micro_batch_size == 32
    assert config.actor.optim.lr == 1e-6
    assert config.actor.optim.lr_warmup_steps_ratio == 0


def check_local(config, *, actor, rollout, reference):
    if actor:
        assert config.actor.ppo_mini_batch_size == 16
        assert config.actor.ppo_micro_batch_size == 2
    if rollout:
        assert config.rollout.log_prob_micro_batch_size == 4
    if reference:
        assert config.ref.log_prob_micro_batch_size == 4
