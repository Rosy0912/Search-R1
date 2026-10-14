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


def run_final(config, output_dir, expected_count=51713, resume_rows=None):
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
    from search_r1.baselines.holdout import _validate_prebuilt, frame_identity, ordered_question_ids_hash
    record = _validate_prebuilt(config)['manifest']['splits']['final']
    assert frame_identity(trainer.val_dataset.dataframe, config.data.prompt_key) == record['content_identity']
    assert ordered_question_ids_hash(trainer.val_dataset.dataframe) == record['question_ids_sha256']
    trainer.global_steps = 200
    original_score = scoring.batch_answer_metrics
    completed = 0
    previous_rows = []
    if resume_rows:
        with Path(resume_rows).open() as stream:
            previous_rows = [json.loads(line) for line in stream]
        assert [r['final_index'] for r in previous_rows] == list(range(len(previous_rows)))
        assert len(previous_rows) % 512 == 0 and len(previous_rows) < expected_count
        completed = len(previous_rows)
        from torch.utils.data import DataLoader, Subset
        from verl.utils.dataset.rl_dataset import collate_fn
        trainer.val_dataloader = DataLoader(Subset(trainer.val_dataset, range(completed, expected_count)),
            batch_size=512, shuffle=False, drop_last=False, collate_fn=collate_fn)
    from search_r1.baselines.tree_raw_tokens import register_validation_tokenizer, tokenizer_validation_scope
    tokenizer_binding = register_validation_tokenizer(tokenizer)
    original_reward = trainer.val_reward_fn
    batch_context = None

    def scoped_reward(batch):
        nonlocal batch_context
        stamp = time.monotonic()
        with tokenizer_validation_scope(tokenizer, tokenizer_binding) as context:
            result = original_reward(batch)
        batch_context = context
        print('FINAL51713_SCORING_TIMING=' + json.dumps(dict(event='reward', rows=len(batch),
            seconds=time.monotonic()-stamp, time=time.time())), flush=True)
        return result

    trainer.val_reward_fn = scoped_reward
    started = time.monotonic()
    batch_started = started
    rows_path = output / 'question-metrics.jsonl'
    with rows_path.open('x') as stream:
        for row in previous_rows: stream.write(json.dumps(row)+'\n')

    def record_scores(batch, tok):
        nonlocal completed, batch_started, batch_context
        stamp = time.monotonic()
        assert tok is tokenizer and batch_context is not None
        with tokenizer_validation_scope(tokenizer, tokenizer_binding, _context=batch_context):
            rows = original_score(batch, tok)
        batch_context = None
        print('FINAL51713_SCORING_TIMING=' + json.dumps(dict(event='metrics', rows=len(batch),
            seconds=time.monotonic()-stamp, time=time.time())), flush=True)
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
        assert metrics['val/qa_count/all'] == expected_count-len(previous_rows)
        from search_r1.baselines.evaluation import aggregate_answer_metrics
        with rows_path.open() as stream:
            all_rows = [json.loads(line) for line in stream]
        assert len(all_rows)==expected_count
        metrics = aggregate_answer_metrics(all_rows)
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
        trainer.val_reward_fn = original_reward


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
    if plan.get('resume_rows'):
        assert sha(plan['resume_rows'])==plan['resume_rows_sha256']
        assert sha(plan['previous_evaluation_plan'])==plan['previous_evaluation_plan_sha256']
        previous=json.loads(Path(plan['previous_evaluation_plan']).read_text())
        assert previous['checkpoint_manifest_sha256']==plan['checkpoint_manifest_sha256']
        assert previous['config_sha256']==plan['config_sha256']
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
    # Native Tree config requires the actor/reference path identities to agree.
    # No RefPolicy worker is instantiated during this scoring-only process.
    config.actor_rollout_ref.ref.model_path = model
    config.data.val_batch_size = 512
    assert config.trainer.n_gpus_per_node == 8 and config.trainer.nnodes == 1
    assert config.actor_rollout_ref.rollout.tensor_model_parallel_size == 1
    os.environ['RBA_FINAL_CHECKPOINT_SHA'] = plan['checkpoint_manifest_sha256']
    os.environ['SEARCH_R1_BASELINE_RUN_ID'] = plan['run_id']
    os.environ['E1_RUN_ID'] = plan['run_id']
    os.environ['SEARCH_R1_BASELINE_OUTPUT'] = plan['output']
    os.environ['SEARCH_R1_BASELINE_AUDIT_DIR'] = plan['output'] + '/audit'
    os.environ['SEARCH_R1_TREE_RANK_AUDIT_DIR'] = plan['output'] + '/rank-audit'
    configuration = json.loads(Path(plan['configuration_receipt']).read_text())
    os.environ.update(SEARCH_R1_BASELINE_DATA_IDENTITY=configuration['data_identity'],
        SEARCH_R1_BASELINE_COMMIT=configuration['source_commit'],
        SEARCH_R1_BASELINE_CONFIG_SHA256=plan['config_sha256'],
        SEARCH_R1_EXECUTION_BACKEND='slurm_h100',
        SEARCH_R1_MODEL_IDENTITY='final200@' + plan['checkpoint_manifest_sha256'],
        TREE_COMMON_DATA=configuration['data'], TREE_INSTRUCT_MODEL=model)
    from ray_cli_compat import prepare_click
    prepare_click()
    import ray
    ray.init(address='local', num_gpus=8, include_dashboard=False,
        runtime_env={'env_vars': {k: v for k, v in os.environ.items()
            if k.startswith(('RBA_', 'SEARCH_R1_', 'RECURSIVE_A_', 'TREE_', 'E1_'))}})
    remote = ray.remote(run_final)
    try:
        ray.get(remote.remote(config, plan['output'], resume_rows=plan.get('resume_rows')))
    finally:
        ray.shutdown()


if __name__ == '__main__':
    main()
