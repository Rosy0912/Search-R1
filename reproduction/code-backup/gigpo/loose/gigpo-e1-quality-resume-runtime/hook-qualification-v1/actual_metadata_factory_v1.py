"""Unlaunched genuine Ray ActorRollout/RefPolicy qualification construction.

Call only from the reviewed owned-runtime adapter. CPU mocks are never used by
this function, and an already-running unrelated Ray cluster is not accepted.
The caller provides an actual owner lease/guard verifier; no permissive default.
"""
from qualification_contract_v1 import admit,require


def create_actual_trainer(config,binding,verify_actual_owned_runtime):
    probability=admit(binding)
    require(callable(verify_actual_owned_runtime),'Missing actual owned Slurm/UUID-OFD/pidfd/cgroup/guard verifier')
    owned=verify_actual_owned_runtime(binding)
    require(owned['same_exclusive_job_node_and_UUIDs'] and owned['guard_alive']
            and owned['local_retriever_alive'] and owned['owned_ray_identity_verified'],
            'Actual Slurm/leases/foreign15/owned Ray lifecycle not established')
    import ray
    from verl.trainer.main_ppo import RewardManager
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer,ResourcePoolManager,Role
    from verl.single_controller.ray import RayWorkerGroup
    from verl.utils import hf_tokenizer
    from qualification_worker_v1 import HookQualificationWorker
    from metadata_colocation_v1 import scoped_metadata_colocation
    require(ray.is_initialized(),'Qualification must use the already-owned loopback Ray bootstrap')
    tokenizer=hf_tokenizer(config.actor_rollout_ref.model.path)
    mapping={Role.ActorRollout:'global_pool',Role.RefPolicy:'global_pool'}
    manager=ResourcePoolManager(resource_pool_spec={'global_pool':[8]},mapping=mapping)
    trainer=RayPPOTrainer(config=config,tokenizer=tokenizer,
        role_worker_mapping={role:ray.remote(HookQualificationWorker) for role in mapping},
        resource_pool_manager=manager,ray_worker_group_cls=RayWorkerGroup,
        reward_fn=RewardManager(tokenizer,0),val_reward_fn=RewardManager(tokenizer,1))
    with scoped_metadata_colocation():trainer.init_workers()
    require(not trainer.use_critic and trainer.use_reference_policy
            and len(trainer.train_dataset)==169028 and len(trainer.val_dataset)==512,
            'Actual algorithm/data execution path differs')
    return trainer,probability
