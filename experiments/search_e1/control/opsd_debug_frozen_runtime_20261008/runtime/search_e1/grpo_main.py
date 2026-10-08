"""Limit Ray to Slurm allocation while leaving Search-R1's RL algorithm unchanged."""
import os
from pathlib import Path
import hydra
import ray
from verl.trainer import main_ppo

_original_init = ray.init

def allocated_init(*args, **kwargs):
    kwargs['address'] = 'local'
    kwargs['_system_config'] = dict(kwargs.get('_system_config', {}), enable_metrics_collection=False)
    kwargs.setdefault('num_cpus', int(os.environ.get('SLURM_CPUS_PER_TASK', '32')))
    kwargs.setdefault('object_store_memory', 2 * 1024**3)
    kwargs.setdefault('runtime_env', {}).setdefault('env_vars', {})['PYTHONUNBUFFERED'] = '1'
    if 'SEARCH_E1_TRAJECTORY_BATCHING' in os.environ:
        kwargs['runtime_env']['env_vars']['SEARCH_E1_TRAJECTORY_BATCHING'] = os.environ['SEARCH_E1_TRAJECTORY_BATCHING']
    for key in ['E1_PHYSICAL_MICRO', 'E1_PHYSICAL_SCORE', 'E1_JIT_OPTIMIZER', 'E1_LOGIT_CHUNK', 'E1_OPTIMIZER_CHUNK','E1_STAGE_CHUNK_STATES', 'E1_VIRTUAL_DP_WORLD']:
        if key in os.environ:
            kwargs['runtime_env']['env_vars'][key] = os.environ[key]
    job = os.environ.get('E1_RUN_ID', str(os.getpid()))
    real_temp = Path(os.environ['E1_BASE']) / 'ray' / job
    real_temp.mkdir(parents=True, exist_ok=True)
    short_temp = Path('/tmp') / ('e1-' + str(os.getuid()) + '-' + job[:8])
    if short_temp.exists() or short_temp.is_symlink():
        if short_temp.resolve() != real_temp.resolve():
            raise RuntimeError('Ray temporary link collision')
    else:
        short_temp.symlink_to(real_temp, target_is_directory=True)
    kwargs.setdefault('_temp_dir', str(short_temp))
    return _original_init(*args, **kwargs)

ray.init = allocated_init
# Use an absolute config path: upstream's config directory is not a Python package.
@hydra.main(config_path=str(Path(main_ppo.__file__).parent / 'config'), config_name='ppo_trainer', version_base=None)
def main(config):
    world = int(config.trainer.n_gpus_per_node) * int(config.trainer.nnodes)
    for name, value in (
        ('actor.ppo_micro_batch_size', config.actor_rollout_ref.actor.ppo_micro_batch_size),
        ('rollout.log_prob_micro_batch_size', config.actor_rollout_ref.rollout.log_prob_micro_batch_size),
        ('ref.log_prob_micro_batch_size', config.actor_rollout_ref.ref.log_prob_micro_batch_size),
    ):
        if int(value) < world or int(value) % world:
            raise ValueError(f'{name}={value} must be a positive multiple of GPU count {world}; upstream divides it across ranks')
    mini = int(config.actor_rollout_ref.actor.ppo_mini_batch_size)
    micro = int(config.actor_rollout_ref.actor.ppo_micro_batch_size)
    if mini % micro:
        raise ValueError('PPO minibatch must be divisible by global microbatch')
    try:
        return main_ppo.main.__wrapped__(config)
    finally:
        ray.shutdown()

if __name__ == '__main__':
    main()
