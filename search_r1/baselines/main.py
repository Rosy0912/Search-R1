"""python -m search_r1.baselines.main baselines.method=gigpo [Hydra overrides]."""
import hydra
import ray
from .controller import validate_config


@hydra.main(config_path='../../verl/trainer/config', config_name='ppo_baselines', version_base=None)
def main(config):
    validate_config(config)
    from verl.trainer.main_ppo import main_task
    if not ray.is_initialized():
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})
    ray.get(main_task.remote(config))


if __name__ == '__main__':
    main()
