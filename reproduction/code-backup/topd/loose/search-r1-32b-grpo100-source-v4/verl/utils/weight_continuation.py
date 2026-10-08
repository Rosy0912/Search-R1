"""Explicit HF-weight continuation; never claims to restore Adam/RNG state."""
from pathlib import Path


def model_initialization_path(model, actor):
    original = model.path
    checkpoint = model.get('actor_checkpoint_path', None)
    if checkpoint:
        if Path(checkpoint).resolve() == Path(original).resolve():
            raise ValueError('Continuation actor must differ from original KL reference')
    return checkpoint if actor and checkpoint else original


def completed_updates(config, total_steps):
    k = config.trainer.get('weight_continuation_step', 0)
    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise ValueError('Completed update count must be a nonnegative integer')
    checkpoint = config.actor_rollout_ref.model.get('actor_checkpoint_path', None)
    if bool(k) != bool(checkpoint):
        raise ValueError('Actor checkpoint and continuation count must be supplied together')
    if k:
        model_initialization_path(config.actor_rollout_ref.model, actor=False)
        if k + 5 >= total_steps:
            raise ValueError('Five resumed updates must fit before terminal counter')
        if config.trainer.get('acceptance_step', 0) != k + 5:
            raise ValueError('Continuation acceptance must save at K+5')
    return k


def scheduler_from_completed(optimizer, warmup_steps, k):
    """LambdaLR's constructor initial step lands at K, without fake Adam steps."""
    from verl.utils.torch_functional import get_constant_schedule_with_warmup
    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise ValueError('Invalid completed update count')
    if k:
        for group in optimizer.param_groups:
            group['initial_lr'] = group['lr']
    return get_constant_schedule_with_warmup(
        optimizer, warmup_steps, last_epoch=k-1)
