"""External task adapter; original baseline entry and trainer.fit remain intact.

The runtime-only main_task dispatch replacement changes the worker class and
adds observers around existing RPCs. Source-file bytes are not edited. Complete
restore/validation/first5 wait/health stop/save/final51713 stay in frozen fit.
This local candidate requires exact production+hook acceptance before startup.
"""
import json
import os
from pathlib import Path

import ray
from formal_protection_contract_v1 import admit_binding, require, sha


def actual_binding():
    path = Path(os.environ['GIGPO_PROTECTION_FORMAL_BINDING'])
    require(sha(path) == os.environ['GIGPO_PROTECTION_FORMAL_BINDING_SHA256'],
            'Actual production binding changed')
    binding = json.loads(path.read_text())
    source = os.environ['SEARCH_R1_BASELINE_COMMIT']
    admit_binding(binding, source)
    return binding


def main_task(config):
    """Same original driver initialization, with the external observed worker."""
    from pprint import pprint
    from omegaconf import OmegaConf
    from verl.utils.fs import copy_local_path_from_hdfs
    from verl.utils import hf_tokenizer
    from verl.trainer.main_ppo import RewardManager
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
    from verl.single_controller.ray import RayWorkerGroup
    from verl.workers.fsdp_workers import CriticWorker
    from formal_probability_worker_v1 import ProductionProbabilityWorker
    from formal_probability_driver_v1 import OuterProbabilityHooks
    binding = actual_binding()
    require(config.actor_rollout_ref.actor.strategy == config.critic.strategy == 'fsdp'
            and config.baselines.method == 'gigpo' and not config.reward_model.enable,
            'External production scope cannot change the original algorithm workers')
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)
    tokenizer = hf_tokenizer(local_path)
    mapping = {Role.ActorRollout: 'global_pool', Role.Critic: 'global_pool', Role.RefPolicy: 'global_pool'}
    worker_mapping = {Role.ActorRollout: ray.remote(ProductionProbabilityWorker),
                      Role.Critic: ray.remote(CriticWorker), Role.RefPolicy: ray.remote(ProductionProbabilityWorker)}
    pool = ResourcePoolManager(resource_pool_spec={'global_pool': [config.trainer.n_gpus_per_node] * config.trainer.nnodes},
                               mapping=mapping)
    trainer = RayPPOTrainer(config=config, tokenizer=tokenizer, role_worker_mapping=worker_mapping,
        resource_pool_manager=pool, ray_worker_group_cls=RayWorkerGroup,
        reward_fn=RewardManager(tokenizer, 0), val_reward_fn=RewardManager(tokenizer, 1))
    from metadata_colocation_v1 import scoped_metadata_colocation
    with scoped_metadata_colocation():
        trainer.init_workers()
    hooks = OuterProbabilityHooks(trainer, binding).install()
    # No manual one_outer imitation: original fit owns all state continuity,
    # data iteration, optimization, same-process gate30 and final evaluation.
    return trainer.fit()


class BoundTask:
    """Keep new explicit controls when frozen baseline entry adds audit env."""
    def __init__(self, task, environment):
        self.task, self.environment = task, dict(environment)

    def options(self, **kwargs):
        runtime = dict(kwargs.pop('runtime_env', {}))
        existing = dict(runtime.get('env_vars', {}))
        for key, value in self.environment.items():
            require(key not in existing or existing[key] == value, 'Conflicting frozen entry environment: ' + key)
            existing[key] = value
        runtime['env_vars'] = existing
        return self.task.options(runtime_env=runtime, **kwargs)

    def remote(self, *args, **kwargs):
        return self.options().remote(*args, **kwargs)


def main():
    # This occurs before baseline preflight or creation of any GPU worker.
    actual_binding()
    import verl.trainer.main_ppo as original_main
    from search_r1.baselines import main as original_entry
    environment = {key: os.environ[key] for key in (
        'GIGPO_PROTECTION_FORMAL_BINDING', 'GIGPO_PROTECTION_FORMAL_BINDING_SHA256',
        'SEARCH_R1_BASELINE_OUTPUT', 'SEARCH_R1_BASELINE_RUN_ID', 'SEARCH_R1_BASELINE_COMMIT',
        'SEARCH_R1_BASELINE_CONFIG_SHA256', 'SLURM_JOB_ID', 'PYTHONPATH')}
    original_main.main_task = BoundTask(ray.remote(main_task), environment)
    # Source baseline main still performs original config/holdout/runtime/assets
    # validation and starts only our explicitly bound main_task through Ray.
    original_entry.main()


if __name__ == '__main__':
    main()
