"""Strict full-state continuation of the immutable Instruct job3185 step75."""
import json
from pathlib import Path

from omegaconf import OmegaConf
from .recovery import atomic_json_save, file_sha256, restore_rng, rng_state

CHECKPOINT75 = ('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004/runs/'
                'gigpo-e1-instruct300-job3185-1791112239036063948/checkpoints/actor/global_step_75')
MANIFEST75 = '65f5eef374d0eb5378aa50d56b9c883755e961b154526a26c05eefcd127968a9'
SOURCE75 = 'bb1adde13c7fd489aef4feb68a36aeb0d29d5da3'
CHECKPOINT80 = ('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004/runs/'
                'gigpo-e1-instruct300-resume75-job3225-1791150281881004640/checkpoints/actor/global_step_80')
MANIFEST80 = '7e108f4d1de488e8484a25deb3d50d9f9a200f2b30e0c90ffc3f9bc68751ad03'
SOURCE80 = 'e5b30d7cf35c30a83d07375733501392d6fa42f1'


def provenance(config):
    path = config.trainer.instruct300.restore_checkpoint
    if path == CHECKPOINT75:
        return dict(step=75, checkpoint=CHECKPOINT75, manifest=MANIFEST75,
                    source=SOURCE75, job='3185', actual=75,
                    recovery='fresh_official_instruct', discarded=8)
    if path == CHECKPOINT80:
        return dict(step=80, checkpoint=CHECKPOINT80, manifest=MANIFEST80,
                    source=SOURCE80, job='3225', actual=5,
                    recovery='full_state_instruct75', discarded=6)
    raise ValueError('Instruct resume checkpoint identity changed')


def enabled(config):
    return bool(config.trainer.get('instruct300', {}).get('restore_checkpoint'))


def start_step(config):
    if not enabled(config):
        return 0
    control = config.trainer.instruct300
    origin = provenance(config)
    if (control.restore_manifest_sha256 != origin['manifest']
            or config.trainer.acceptance_step != origin['step'] + 5 or config.trainer.val_before_train
            or config.trainer.resume.enabled or config.trainer.total_training_steps != 300
            or not config.actor_rollout_ref.actor.get('joint_loss_scaling', False)):
        raise ValueError('Instruct resume75 identity or continuation boundaries changed')
    return origin['step']


def _leaves(value, prefix=''):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            result.update(_leaves(item, prefix + '.' + key if prefix else key))
        return result
    return {prefix: value}


def migration_diff(saved, current):
    """Reject every configuration change except named continuation controls."""
    old, new = _leaves(saved), _leaves(current)
    absent = object()
    changes = {key: (old.get(key, absent), new.get(key, absent))
               for key in old.keys() | new.keys() if old.get(key, absent) != new.get(key, absent)}
    exact = {
        'trainer.acceptance_step': (5, 80),
        'trainer.val_before_train': (True, False),
        'actor_rollout_ref.actor.joint_loss_scaling': (absent, True),
        'trainer.instruct300.restore_checkpoint': (absent, CHECKPOINT75),
        'trainer.instruct300.restore_manifest_sha256': (absent, MANIFEST75),
    }
    if new.get('trainer.instruct300.restore_checkpoint') == CHECKPOINT80:
        exact = {
            'trainer.acceptance_step': (80, 85),
            'trainer.instruct300.restore_checkpoint': (CHECKPOINT75, CHECKPOINT80),
            'trainer.instruct300.restore_manifest_sha256': (MANIFEST75, MANIFEST80),
        }
    paths = {'trainer.default_local_dir', 'trainer.instruct300.approval_path', 'trainer.experiment_name'}
    for key, values in changes.items():
        if key in paths:
            if not all(isinstance(v, str) and v for v in values):
                raise ValueError('Invalid continuation path/name: ' + key)
        elif key not in exact or values != exact[key]:
            raise ValueError('Unapproved resume configuration change: ' + key)
    if not all(key in changes for key in exact):
        raise ValueError('Missing explicit resume75 controls')
    return {key: dict(old=None if left is absent else left, new=None if right is absent else right)
            for key, (left, right) in sorted(changes.items())}


def restore(trainer):
    """Load driver and all eight rank states before any resumed fetch."""
    import torch
    from .instruct300 import check_manifest, equal_state, REVISION
    step = start_step(trainer.config)
    origin = provenance(trainer.config)
    if step not in (75, 80):
        raise RuntimeError('Restore called outside the explicit continuation')
    path = Path(origin['checkpoint']) / 'training_state/manifest.json'
    if file_sha256(path) != origin['manifest']:
        raise RuntimeError('Original step75 manifest changed')
    identity = json.loads(path.read_text())
    if (identity['job_id'] != origin['job'] or identity['code_commit'] != origin['source']
            or identity['logical_step'] != step or identity['actual_new_updates'] != origin['actual']
            or identity['recovery_mode'] != origin['recovery']):
        raise RuntimeError('Original full checkpoint75 provenance changed')
    manifest = dict(path=str(path), sha256=origin['manifest'], identity=identity)
    check_manifest(manifest)
    files = identity['files']
    driver_record = next(v for v in files if v['path'].endswith('/driver.pt'))
    driver = torch.load(driver_record['path'], map_location='cpu', weights_only=False)
    if (driver['schema'] != 1 or driver['logical_step'] != step or driver['actual_new_updates'] != origin['actual']
            or driver['source_revision'] != REVISION
            or driver['source'] != trainer.config.actor_rollout_ref.model.path
            or driver['sampler']['cursor'] != step * 512 or driver['sampler']['epoch'] != 0
            or driver['sampler']['size'] != 169028):
        raise RuntimeError('Saved driver source, progress, or data cursor changed')
    differences = migration_diff(driver['config'], OmegaConf.to_container(trainer.config, resolve=True))
    algorithm = driver['algorithm_state']
    algorithm_record = next(v for v in files if '/algorithm/' in v['path'])
    if algorithm != json.loads(Path(algorithm_record['path']).read_text()):
        raise RuntimeError('Saved GiGPO receipt differs from driver')
    controller = trainer.baseline_controller
    expected_parameters = OmegaConf.to_container(trainer.config.baselines.gigpo, resolve=True)
    if (algorithm['method'] != 'gigpo' or algorithm['global_step'] != step
            or not algorithm['resume_supported'] or algorithm['gigpo_parameters'] != expected_parameters
            or algorithm['loss_reduction'] != 'sequence_mean'
            or algorithm['groups'] != 'stateless_per_rollout_with_step_ids_and_advantage_broadcast'
            or algorithm['base_commit'] != controller.config.base_commit
            or algorithm['algorithm_source_commit'] != controller.config.algorithm_source_commit
            or algorithm['teacher_path'] != controller.config.distillation.teacher_path):
        raise RuntimeError('GiGPO algorithm identity or grouping semantics changed')
    rank_records = [v for v in files if 'pause_identity' in v]
    shas = {str(v['pause_identity']['rank']): v['sha256'] for v in rank_records}
    loaded = trainer.actor_rollout_wg.load_training_state(origin['checkpoint'], shas, step, step * 20)
    if (len(loaded) != 8 or {v['rank'] for v in loaded} != set(range(8))
            or any(v['logical_step'] != step for v in loaded)):
        raise RuntimeError('Incomplete actual eight-rank full restore')
    now = trainer.actor_rollout_wg.checkpoint_pause_identity()
    if (len(now) != 8 or {v['rank'] for v in now} != set(range(8))
            or len({v['pid'] for v in now}) != 8):
        raise RuntimeError('Actual restored worker identity coverage is incomplete')
    expected = {v['pause_identity']['rank']: v['pause_identity'] for v in rank_records}
    for value in now:
        # The new process has an explicitly different PID/start time. All saved
        # tensor, buffer, optimizer, scheduler and RNG identities must match.
        saved = expected[value['rank']]
        if {k: v for k, v in value.items() if k not in ('pid', 'start_ticks')} != {
                k: v for k, v in saved.items() if k not in ('pid', 'start_ticks')}:
            raise RuntimeError('Restored worker model/Adam/scheduler/RNG differs from75')
    trainer.recovery_sampler.load_state_dict(driver['sampler'])
    if not equal_state(trainer.recovery_sampler.state_dict(), driver['sampler']):
        raise RuntimeError('Saved data permutation/cursor restore differs')
    if vars(trainer.kl_ctrl).keys() != driver['kl_controller'].keys():
        raise RuntimeError('KL controller fields changed')
    vars(trainer.kl_ctrl).update(driver['kl_controller'])
    controller.stage.load_state_dict(algorithm['bapo'])
    if controller.stage.state_dict() != algorithm['bapo']:
        raise RuntimeError('Saved controller stage differs after restore')
    restore_rng(driver['rng'])
    if not equal_state(rng_state(), driver['rng']):
        raise RuntimeError('Actual driver RNG restoration differs')
    trainer.actual_outer_updates = 0
    trainer.instruct_restore_manifest = manifest
    atomic_json_save(dict(status=f'RESTORED_FULL_STATE{step}', source_manifest=manifest,
        retained_logical_step=step, resumed_actual_updates=0, discarded_failed_branch_updates=origin['discarded'],
        migration=differences, workers=now, driver_cursor=step * 512, engine_restart_exact=False),
        Path(trainer.config.trainer.instruct300.approval_path).with_name(f'resume{step}-restored.json'))
