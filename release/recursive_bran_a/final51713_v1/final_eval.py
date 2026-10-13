"""Independent final evaluator. Never import/deploy into the live training process."""
import hashlib
import json
import os
import time
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_checkpoint(manifest, expected_config):
    assert manifest['status'] == 'COMPLETE_SAME_WORLD_STATE_SAVED'
    identity = manifest['identity']
    assert identity['config_sha256'] == expected_config
    assert identity['completed_step'] == 200
    assert identity['guard_continue_accounting']['completed_outer_updates'] == 200
    assert manifest['driver']['next_step'] == 201
    ranks = [w['identity']['rank'] for w in manifest['worker_states']]
    assert sorted(ranks) == list(range(8))
    return manifest['hf_checkpoint']


def run_final(config, output_dir, expected_count=51713):
    import ray
    from verl.utils import hf_tokenizer
    from verl.single_controller.ray import RayWorkerGroup
    from verl.trainer.main_ppo import RewardManager
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
    from final_worker import FrozenFinalWorker
    import search_r1.baselines.evaluation as scoring

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    tokenizer = hf_tokenizer(config.actor_rollout_ref.model.path)
    trainer = RayPPOTrainer(config=config, tokenizer=tokenizer,
        role_worker_mapping={Role.ActorRollout: ray.remote(FrozenFinalWorker)},
        resource_pool_manager=ResourcePoolManager(
            resource_pool_spec={'final': [8]}, mapping={Role.ActorRollout: 'final'}),
        ray_worker_group_cls=RayWorkerGroup,
        reward_fn=RewardManager(tokenizer, 0), val_reward_fn=RewardManager(tokenizer, 0))
    trainer.init_workers()
    trainer._prepare_final_validation()
    assert trainer.actor_rollout_wg.world_size == 8
    assert len(trainer.val_dataset) == expected_count, 'No dropped final questions allowed'
    trainer.global_steps = 200
    original_score = scoring.batch_answer_metrics
    completed = 0
    started = time.monotonic()
    batch_started = started
    rows_path = output / 'question-metrics.jsonl'

    def record_scores(batch, tok):
        nonlocal completed, batch_started
        rows = original_score(batch, tok)
        assert len(rows) == len(batch)
        # Native sequential dataloader, drop_last=False; padded rows were removed
        # by the unchanged SearchRollout before reaching this scorer.
        with rows_path.open('a') as stream:
            for row in rows:
                stream.write(json.dumps(dict(final_index=completed, **row)) + '\n')
                completed += 1
            stream.flush()
        now = time.monotonic()
        progress = dict(completed=completed, total=expected_count,
            batch_rows=len(rows), batch_seconds=now-batch_started,
            elapsed_seconds=now-started, questions_per_second=completed/(now-started),
            workers=8, global_batch=int(config.data.val_batch_size), time=time.time())
        (output / 'progress.json').write_text(json.dumps(progress))
        print('FINAL51713_PROGRESS=' + json.dumps(progress), flush=True)
        batch_started = now
        return rows

    scoring.batch_answer_metrics = record_scores
    try:
        metrics = trainer._validate()
        assert completed == expected_count
        assert metrics['val/qa_count/all'] == expected_count
        trainer.actor_rollout_wg.release_rollout_policy_cache()
        summary = dict(status='COMPLETE', count=completed, metrics=metrics,
            checkpoint_sha=os.environ['RBA_FINAL_CHECKPOINT_SHA'], workers=8,
            batch_size=int(config.data.val_batch_size), elapsed_seconds=time.monotonic()-started,
            question_metrics_sha=sha(rows_path))
        (output / 'summary.json').write_text(json.dumps(summary, indent=2))
        print('SEARCH_R1_FINAL_METRICS_JSON=' + json.dumps(metrics), flush=True)
        return summary
    finally:
        scoring.batch_answer_metrics = original_score


def main():
    import argparse
    from omegaconf import OmegaConf
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    assert plan['ready'] is True
    assert plan['training_complete_outer'] == 200
    assert plan['owned_training_released'] is True
    assert plan['native_final_complete'] is False, 'Never duplicate a completed final evaluation'
    assert sha(plan['config']) == plan['config_sha256']
    assert sha(plan['checkpoint_manifest']) == plan['checkpoint_manifest_sha256']
    manifest = json.loads(Path(plan['checkpoint_manifest']).read_text())
    model = validate_checkpoint(manifest, plan['config_sha256'])
    assert sha(plan['training_start']) == plan['training_start_sha256']
    assert json.loads(Path(plan['training_start']).read_text())['run_id'] == plan['training_run']
    # Before launch, the boundary supervisor must produce the explicit terminal,
    # eight UUID/free-space lease, and complete200 receipts referenced in plan.
    for receipt in plan['admission_receipts']:
        assert sha(receipt['path']) == receipt['sha256']
    assert len(plan['admission_receipts']) >= 3
    config = OmegaConf.load(plan['config'])
    config.actor_rollout_ref.model.path = model
    config.data.val_batch_size = 512
    assert config.trainer.n_gpus_per_node == 8 and config.trainer.nnodes == 1
    assert config.actor_rollout_ref.rollout.tensor_model_parallel_size == 1
    os.environ['RBA_FINAL_CHECKPOINT_SHA'] = plan['checkpoint_manifest_sha256']
    os.environ['SEARCH_R1_BASELINE_RUN_ID'] = plan['run_id']
    os.environ['E1_RUN_ID'] = plan['run_id']
    os.environ['SEARCH_R1_BASELINE_OUTPUT'] = plan['output']
    os.environ['SEARCH_R1_BASELINE_AUDIT_DIR'] = plan['output'] + '/audit'
    os.environ['SEARCH_R1_TREE_RANK_AUDIT_DIR'] = plan['output'] + '/rank-audit'
    import ray
    ray.init(address='local', num_gpus=8, include_dashboard=False,
        runtime_env={'env_vars': {k: v for k, v in os.environ.items()
            if k.startswith(('RBA_', 'SEARCH_R1_', 'RECURSIVE_A_', 'TREE_', 'E1_'))}})
    remote = ray.remote(run_final)
    try:
        ray.get(remote.remote(config, plan['output']))
    finally:
        ray.shutdown()


if __name__ == '__main__':
    main()
