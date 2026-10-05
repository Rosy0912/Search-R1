"""Reviewed best-validation full-state branch, preserving the original objective."""
import json
import math
from pathlib import Path
from .recovery import file_sha256


def enabled(config):
    return bool(getattr(config.trainer, 'quality_resume', {}).get('enabled', False))


def selected_origin(config):
    control = config.trainer.quality_resume
    summary_path = Path(control.selection_path)
    if file_sha256(summary_path) != control.selection_sha256:
        raise ValueError('Quality selection bytes changed')
    summary = json.loads(summary_path.read_text())
    records = summary['completed_candidates']
    pinned = json.loads(Path(__file__).with_name('quality_candidates.json').read_text())
    by_step = {item['step']: item for item in pinned}
    if len(records) != 8 or {r['candidate_step'] for r in records} != set(by_step):
        raise ValueError('Quality selection must include all eight full512 candidates')
    for record in records:
        identity = by_step[record['candidate_step']]
        if (record['status'] != 'GPU512_EVALUATED' or record['candidate'] != identity['path']
                or record['metrics']['val/qa_count/all'] != 512
                or record['cleanup_residual_states'] or record['final_gpu_snapshot']['processes']):
            raise ValueError('Incomplete candidate evaluation or resource cleanup')
        for key in ('val/f1/all', 'val/em/all'):
            value = float(record['metrics'][key])
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('Invalid candidate quality metric')
    best = max(records, key=lambda r: (r['metrics']['val/f1/all'], r['candidate_step']))
    if (summary['status'] != 'ALL_CANDIDATES_GPU512_COMPLETE_UNREVIEWED'
            or summary['selected'] != best):
        raise ValueError('Selection rule or final summary differs')
    review_path = Path(control.selection_review_path)
    if file_sha256(review_path) != control.selection_review_sha256:
        raise ValueError('Quality independent review bytes changed')
    review = json.loads(review_path.read_text())
    identity = by_step[best['candidate_step']]
    if (review.get('status') != 'PASS' or review.get('selection_sha256') != control.selection_sha256
            or review.get('selected_checkpoint') != identity['path']
            or review.get('selected_manifest_sha256') != identity['manifest_sha256']
            or review.get('evaluation_job') != '3232'):
        raise ValueError('Selected candidate lacks bound independent acceptance')
    original_actual = identity['step'] if identity['job_id'] == '3185' else identity['step'] - (75 if identity['job_id'] == '3225' else 80)
    return dict(step=identity['step'], checkpoint=identity['path'], manifest=identity['manifest_sha256'],
                source=identity['code_commit'], job=identity['job_id'], actual=original_actual,
                recovery=('fresh_official_instruct' if identity['job_id'] == '3185' else
                          'full_state_instruct75' if identity['job_id'] == '3225' else 'full_state_instruct80'),
                discarded=None, original_optimizer_steps=identity['step'] * 20,
                validation_f1=float(best['metrics']['val/f1/all']))


def optimizer_steps(config, logical_step):
    if not enabled(config):
        return logical_step * 20
    origin = selected_origin(config)
    if not origin['step'] <= logical_step <= 300:
        raise ValueError('Optimizer accounting is outside the selected continuation')
    return origin['original_optimizer_steps'] + (logical_step - origin['step']) * 10


def migration_diff(saved, current):
    from .instruct_resume import _leaves
    old, new = _leaves(saved), _leaves(current)
    missing = object()
    changes = {key: (old.get(key, missing), new.get(key, missing))
               for key in old.keys() | new.keys() if old.get(key, missing) != new.get(key, missing)}
    paths = {'trainer.default_local_dir', 'trainer.instruct300.approval_path', 'trainer.experiment_name'}
    controls = {'trainer.instruct300.restore_checkpoint', 'trainer.instruct300.restore_manifest_sha256',
                'trainer.acceptance_step'}
    for key, (left, right) in changes.items():
        if key in paths | controls:
            if key != 'trainer.acceptance_step' and (not isinstance(right, str) or not right):
                raise ValueError('Invalid continuation identity control: ' + key)
        elif key == 'actor_rollout_ref.actor.ppo_mini_batch_size':
            if (left, right) != (128, 256):
                raise ValueError('Only the authorized128 to256 mini migration is allowed')
        elif key == 'critic.ppo_mini_batch_size':
            if ((left, right) != (128, 256)
                    or old.get('baselines.method') != 'gigpo' or new.get('baselines.method') != 'gigpo'
                    or old.get('algorithm.adv_estimator') != new.get('algorithm.adv_estimator')
                    or new.get('algorithm.adv_estimator') in (None, 'gae')):
                raise ValueError('Only the unused GiGPO critic mini interpolation may follow actor128 to256')
        elif key == 'baselines.holdout.output_dir':
            if (not isinstance(left, str) or not isinstance(right, str)
                    or Path(left) != Path(old['trainer.default_local_dir']).parent / 'common-data'
                    or Path(right) != Path(new['trainer.default_local_dir']).parent / 'common-data'):
                raise ValueError('Holdout output metadata must stay in each independent run/common-data')
        elif key == 'actor_rollout_ref.actor.joint_loss_scaling':
            if left is not missing or right is not True:
                raise ValueError('Unexpected numeric representation migration')
        elif key.startswith('actor_rollout_ref.actor.policy_safety.'):
            if left is not missing or new.get('baselines.method') != 'gigpo':
                raise ValueError('Only newly authorized GiGPO policy protections may be added')
            allowed = {'enabled', 'dual_clip', 'kl_log_bound', 'monitor_first_global_rows',
                       'target_old_policy_kl', 'negative_tail_limit', 'calibration_evidence_sha256', 'calibration_evidence_path'}
            if key.rsplit('.', 1)[-1] not in allowed:
                raise ValueError('Unknown policy safety control')
        elif key == 'trainer.val_before_train':
            if left is not True or right is not False:
                raise ValueError('Unexpected initial-validation migration')
        elif key == 'trainer.trajectory_logging':
            if left is not missing or right is not True:
                raise ValueError('Complete trajectory logging must be explicitly enabled')
        elif key.startswith('trainer.quality_resume.'):
            if left is not missing:
                raise ValueError('Historical checkpoint already has incompatible quality controls')
        else:
            raise ValueError('Unapproved best-checkpoint resume config change: ' + key)
    if (new.get('actor_rollout_ref.actor.ppo_mini_batch_size') != 256
            or new.get('trainer.quality_resume.enabled') is not True
            or new.get('trainer.trajectory_logging') is not True):
        raise ValueError('Missing ten-optimizer/trajectory controls')
    if any(key.startswith('actor_rollout_ref.actor.policy_safety.') for key in changes):
        from .policy_safety import validate_config
        validate_config(current['actor_rollout_ref']['actor']['policy_safety'], require_thresholds=False)
    return {key: dict(old=None if left is missing else left, new=None if right is missing else right)
            for key, (left, right) in sorted(changes.items())}


def initialize_health(trainer):
    """Evaluate actual restored state, then restore every training RNG/state again."""
    from .instruct300 import record_evaluation, validate_stage64
    from .instruct_resume import restore
    origin = selected_origin(trainer.config)
    full = trainer._validate()
    record_evaluation(trainer, full, 'restored512', trainer.config.data.val_files)
    stage = validate_stage64(trainer)
    if float(full['val/f1/all']) < .5 * origin['validation_f1']:
        raise RuntimeError('Restored full-state validation already collapsed relative to selected HF')
    trainer.quality_health = dict(baseline512=float(full['val/f1/all']),
        baseline64=float(stage['val/f1/all']), recent_rewards=[],
        last_checked_step=origin['step'], baseline_step=origin['step'])
    # Deterministic validation still changes rollout/driver RNG and manager state.
    # Reset to the exact saved training boundary before the first resumed fetch.
    restore(trainer)
    return full


def monitor_health(trainer, metrics, stage_metrics=None, periodic_metrics=None):
    """Require full512 confirmation before rejecting a suspected quality collapse."""
    from .instruct300 import record_evaluation, validate_stage64
    from .recovery import atomic_json_save
    state = trainer.quality_health
    reward = float(metrics['critic/rewards/mean'])
    if not math.isfinite(reward):
        raise RuntimeError('Non-finite training reward')
    state['recent_rewards'] = (state['recent_rewards'] + [reward])[-5:]
    origin = selected_origin(trainer.config)['step']
    due = (trainer.global_steps - origin) % 5 == 0
    if due and stage_metrics is None:
        stage_metrics = validate_stage64(trainer)
    stage_low = (stage_metrics is not None and state['baseline64'] > 0
                 and float(stage_metrics['val/f1/all']) < .5 * state['baseline64'])
    reward_low = (due and len(state['recent_rewards']) == 5
                  and sum(state['recent_rewards']) / 5 < .01)
    confirmed = periodic_metrics
    if confirmed is None and (stage_low or reward_low):
        confirmed = trainer._validate()
        record_evaluation(trainer, confirmed, 'quality512', trainer.config.data.val_files)
    collapsed = (confirmed is not None and state['baseline512'] > 0
                 and float(confirmed['val/f1/all']) < .5 * state['baseline512'])
    if due or confirmed is not None:
        state['last_checked_step'] = trainer.global_steps
        evidence = dict(logical_step=trainer.global_steps, health=state,
                        stage_metrics=stage_metrics, full512=confirmed,
                        suspected_stage=stage_low, suspected_reward=reward_low,
                        confirmed_collapse=collapsed)
        quality_dir = Path(trainer.config.trainer.default_local_dir).parent / 'quality'
        quality_dir.mkdir(parents=True, exist_ok=True)
        atomic_json_save(evidence, quality_dir / f'iter_{trainer.global_steps:06d}.json')
    if collapsed:
        # The bad branch is preserved for diagnosis, never silently reused as healthy.
        trainer._save_checkpoint()
        raise RuntimeError('QUALITY_COLLAPSE_CONFIRMED_FULL512: saved failed branch; diagnose before healthy-checkpoint resume')
    return stage_metrics
