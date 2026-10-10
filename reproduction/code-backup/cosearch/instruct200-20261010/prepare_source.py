"""Patch only a separate deployment copy; frozen source is never modified."""
import ast
import hashlib
import json
from pathlib import Path


def patch_source(entry, worker):
    start = entry.index('        nnodes = config.trainer.nnodes', entry.index('    def init_resource_pool_mgr'))
    end = entry.index('\n    def add_reward_model_worker', start)
    replacement = '''        if config.trainer.nnodes != 1 or config.trainer.n_gpus_per_node != 4:
            raise ValueError("CoSearch instruct200 requires one node and exactly four GPUs")
        if not is_reranker_trainable or config.reward_model.enable_resource_pool:
            raise ValueError("Both roles must train in the same four-GPU pool")
        pool = "cosearch_shared_four"
        self.mapping[Role.ActorRollout] = pool
        self.mapping[Role.RefPolicy] = pool
        self.mapping[Role.Critic] = pool
        self.mapping[Role.RerankerActorRollout] = pool
        self.mapping[Role.RerankerRefPolicy] = pool
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager
        return ResourcePoolManager(resource_pool_spec={pool: [4]}, mapping=self.mapping)
'''
    entry = entry[:start] + replacement + entry[end:]
    original = '            self.config.actor.ppo_mini_batch_size *= self.config.rollout.n'
    if worker.count(original) != 1:
        raise ValueError('Unexpected actor mini-batch normalization source')
    worker = worker.replace(original, '''            # This deployment specifies trajectory count, independent of rollout.n.
            if not self.config.actor.get("mini_batch_counts_trajectories", False):
                self.config.actor.ppo_mini_batch_size *= self.config.rollout.n''')
    ast.parse(entry)
    ast.parse(worker)
    return entry, worker



def patch_trainer(trainer):
    replacements = {
        'from verl.single_controller.ray.base import create_colocated_worker_cls':
            'from verl.single_controller.ray.base import create_colocated_worker_cls_fused as create_colocated_worker_cls',
        '(is_last_step or self.global_steps % self.config.trainer.test_freq == 0)':
            '(is_last_step or self.global_steps in (5, 50, 100, 150, 200))',
        'is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration':
            'is_last_step or self.global_steps in (5, 50, 100, 150, 200) or esi_close_to_expiration',
        'print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")':
            'raise FileNotFoundError(f"Resume requires dataloader state: {dataloader_local_path}")',
    }
    for before, after in replacements.items():
        if trainer.count(before) != 1:
            raise ValueError('Unexpected trainer source: ' + before)
        trainer = trainer.replace(before, after)
    before = '    batch.meta_info["global_token_num"] ='
    if trainer.count(before) != 1: raise ValueError("Missing PPO metadata point")
    trainer = trainer.replace(before, '    batch.meta_info["cosearch_outer_step"] = global_steps\n' + before)
    before = '                    if reranker_train_enabled:\n                        # Safe check: only pass reranker_ref_policy_wg'
    if trainer.count(before) != 1: raise ValueError("Missing shared-pool ordering point")
    trainer = trainer.replace(before, '                    main_results = ray.get(main_futures)\n' + before)
    trainer = trainer.replace('main_results, reranker_results = ray.get([main_futures, reranker_futures])', 'reranker_results = ray.get(reranker_futures)')
    ast.parse(trainer)
    return trainer


def patch_actor_config(source):
    before = "    ppo_mini_batch_size: int = 256"
    if source.count(before) != 1: raise ValueError("Unexpected ActorConfig")
    return source.replace(before, before + "\n    mini_batch_counts_trajectories: bool = False")


def prepare(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source in destination.parents:
        raise ValueError('Destination must be separate from frozen source')
    targets = ['main_co_search_ppo.py', 'verl/verl/workers/fsdp_workers.py', 'verl/verl/trainer/ppo/search_r1_reranker_ray_trainer.py', 'verl/verl/workers/config/actor.py']
    originals = [(source / p).read_text() for p in targets]
    changed = (*patch_source(*originals[:2]), patch_trainer(originals[2]), patch_actor_config(originals[3]))
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {}
    for path, before, after in zip(targets, originals, changed):
        out = destination / path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(after)
        manifest[path] = {'source_sha256': hashlib.sha256(before.encode()).hexdigest(),
                          'patched_sha256': hashlib.sha256(after.encode()).hexdigest()}
    (destination / 'patch-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('source')
    parser.add_argument('destination')
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.destination)))
