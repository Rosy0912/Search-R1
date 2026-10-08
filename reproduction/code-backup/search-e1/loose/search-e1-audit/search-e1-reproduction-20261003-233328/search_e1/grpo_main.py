"""Limit Ray to Slurm allocation while leaving Search-R1's RL algorithm unchanged."""
import os
from pathlib import Path
import hydra
import ray
from verl.trainer import main_ppo

_original_init = ray.init

def allocated_init(*args, **kwargs):
    kwargs.setdefault('num_cpus', int(os.environ.get('SLURM_CPUS_PER_TASK', '32')))
    kwargs.setdefault('object_store_memory', 2 * 1024**3)
    kwargs.setdefault('runtime_env', {}).setdefault('env_vars', {})['PYTHONUNBUFFERED'] = '1'
    if 'SEARCH_E1_TRAJECTORY_BATCHING' in os.environ:
        kwargs['runtime_env']['env_vars']['SEARCH_E1_TRAJECTORY_BATCHING'] = os.environ['SEARCH_E1_TRAJECTORY_BATCHING']
    job = os.environ.get('SLURM_JOB_ID', str(os.getpid()))
    kwargs.setdefault('_temp_dir', '/tmp/search-e1-ray-' + job)
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
    return main_ppo.main.__wrapped__(config)

if __name__ == '__main__':
    main()
