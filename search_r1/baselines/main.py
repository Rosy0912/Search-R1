"""python -m search_r1.baselines.main baselines.method=gigpo [Hydra overrides]."""
import hydra
import ray
from .controller import validate_config


@hydra.main(config_path='../../verl/trainer/config', config_name='ppo_baselines', version_base=None)
def main(config):
    from .preflight import validate_gpu_lease
    validate_gpu_lease(config)
    from .holdout import prepare_common_holdout
    prepare_common_holdout(config)
    validate_config(config)
    from .preflight import validate_assets_and_retriever
    validate_assets_and_retriever(config)
    from verl.trainer.main_ppo import main_task
    if not ray.is_initialized():
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})
    ray.get(main_task.remote(config))


if __name__ == '__main__':
    main()
