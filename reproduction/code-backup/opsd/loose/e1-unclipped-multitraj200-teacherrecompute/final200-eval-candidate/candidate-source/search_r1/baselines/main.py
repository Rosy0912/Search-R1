"""python -m search_r1.baselines.main baselines.method=gigpo [Hydra overrides]."""
import hydra
import ray
from pathlib import Path
from .controller import validate_config


@hydra.main(config_path=str(Path(__file__).resolve().parents[2] / 'verl/trainer/config'), config_name='ppo_baselines', version_base=None)
def main(config):
    from .preflight import validate_gpu_lease
    validate_gpu_lease(config)
    from .local_runtime import check_local_runtime
    check_local_runtime(config)
    from .holdout import prepare_common_holdout
    split_manifest = prepare_common_holdout(config)
    from .corrected_evaluation import configure
    from .final200_evaluation import enabled as final200_enabled, configure as configure_final200
    if final200_enabled(config):
        configure_final200(config, split_manifest)
    else:
        configure(config, split_manifest)
    if config.trainer.smoke_only and config.trainer.total_training_steps > 6:
        raise ValueError('Smoke-only runs allow at most five outer updates')
    if config.baselines.holdout_eval_only:
        if not config.trainer.get('val_only', False) or not config.trainer.val_before_train:
            raise ValueError('Holdout evaluation requires val_only and val_before_train')
        if config.baselines.method != 'grpo' or config.baselines.seed != 42:
            raise ValueError('Teacher selection requires fresh GRPO and seed42')
        if config.data.get('val_data_num') is not None:
            raise ValueError('Holdout-only evaluation cannot subsample questions')
        import re
        from pathlib import Path
        from .select_teacher import model_manifest, CANDIDATES
        target = config.baselines.holdout_record_path
        model = Path(config.actor_rollout_ref.model.path).resolve()
        match = re.search(r'/actor/global_step_(\d+)$', str(model))
        if not target or Path(target).exists():
            raise ValueError('Specify a new holdout_record_path before evaluation')
        if not match or int(match.group(1)) not in CANDIDATES:
            raise ValueError('Holdout candidate must be a specified saved actor update')
        checkpoint_hash = model_manifest(config.actor_rollout_ref.model.path)
        config.data.val_files = config.baselines.bapo.stage_validation_files
        config.data.final_val_files = None
    validate_config(config)
    from .runtime_audit import config_boundary
    config_boundary(config, split_manifest)
    from .preflight import validate_assets_and_retriever
    validate_assets_and_retriever(config)
    from verl.trainer.main_ppo import main_task
    if not ray.is_initialized():
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})
    if config.baselines.local_runtime_only:
        check_local_runtime(config, ray.nodes())
    from .runtime_audit import runtime_env_vars
    audit_env = runtime_env_vars()
    from .corrected_evaluation import enabled
    if enabled(config):
        import os
        audit_env['OPSD_CORRECTED_EVAL_STAGE_SHA256'] = os.environ['OPSD_CORRECTED_EVAL_STAGE_SHA256']
        audit_env['SLURM_JOB_ID'] = os.environ['SLURM_JOB_ID']
    if final200_enabled(config):
        import os
        audit_env['OPSD_FINAL200_EVAL_STAGE_SHA256'] = os.environ['OPSD_FINAL200_EVAL_STAGE_SHA256']
        audit_env['OPSD_CORRECTED_EVAL_STAGE_SHA256'] = os.environ['OPSD_FINAL200_EVAL_STAGE_SHA256']
        audit_env['SLURM_JOB_ID'] = os.environ['SLURM_JOB_ID']
    task = main_task.options(runtime_env={'env_vars': audit_env}) if audit_env else main_task
    result = ray.get(task.remote(config))
    if config.baselines.holdout_eval_only:
        import json
        import re
        from pathlib import Path
        from .select_teacher import model_manifest, CANDIDATES
        model = Path(config.actor_rollout_ref.model.path).resolve()
        match = re.search(r'/actor/global_step_(\d+)$', str(model))
        if not match or int(match.group(1)) not in CANDIDATES:
            raise ValueError('Holdout candidate must be a specified saved actor update')
        if model_manifest(model) != checkpoint_hash:
            raise ValueError('Checkpoint changed during evaluation')
        count = int(result['val/qa_count/all'])
        if count != split_manifest['holdout_rows']:
            raise ValueError('Evaluation dropped holdout questions')
        record = dict(outer_update=int(match.group(1)), evaluation_split='common-holdout', seed=42,
            evaluated_count=count, holdout_rows=split_manifest['holdout_rows'],
            mean_answer_f1=float(result['val/f1/all']),
            split_content_identity=split_manifest['content_identity']['sha256'],
            checkpoint_sha256=checkpoint_hash['sha256'], hf_directory=str(model), metrics=result)
        target = Path(config.baselines.holdout_record_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('x') as handle:
            json.dump(record, handle, indent=2)
        print('BASELINE_HOLDOUT_RECORD_JSON=' + json.dumps(record, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
