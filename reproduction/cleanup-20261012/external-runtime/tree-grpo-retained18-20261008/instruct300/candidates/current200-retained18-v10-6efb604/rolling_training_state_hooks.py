"""Save the last two completed Fresh200 boundaries with the existing native API.

HF milestone saves remain owned by the original trainer. These native snapshots
preserve real completed updates, without claiming automatic or exact restart.
"""
import functools
import json
import os
from pathlib import Path
import shutil
import time


def save_completed_boundary(trainer):
    from search_r1.baselines.tree_fresh200 import is_fresh200
    if trainer.config.get('baselines') is None or not is_fresh200(trainer.config):
        return
    step = getattr(trainer, 'actual_outer_updates', 0)
    if step < 1 or step != getattr(trainer, 'global_steps', None):
        return
    if getattr(trainer, '_rolling_native_saved_outer', 0) == step:
        return
    if not trainer.config.trainer.get('training_state_enabled', False):
        raise ValueError('Fresh200 rolling completed state requires native training state')

    from search_r1.baselines.training_state import atomic_save, driver_snapshot, digest
    from search_r1.baselines.tree_guard_continue import accounting
    identity = dict(job_id=os.environ['SLURM_JOB_ID'],
        run_id=os.environ['SEARCH_R1_BASELINE_RUN_ID'],
        code_commit=os.environ['SEARCH_R1_BASELINE_COMMIT'],
        config_sha256=os.environ['SEARCH_R1_BASELINE_CONFIG_SHA256'],
        data_identity=os.environ['SEARCH_R1_BASELINE_DATA_IDENTITY'],
        model_identity=os.environ['SEARCH_R1_MODEL_IDENTITY'],
        completed_step=step, actual_update_limit=trainer.config.trainer.actual_update_limit,
        guard_continue_accounting=accounting(trainer))
    root = Path(trainer.config.trainer.default_local_dir)/'training_state'/'rolling-completed'
    directory = root/f'global_step_{step}'
    workers = trainer.actor_rollout_wg.save_training_state(str(directory), identity, verify=False)
    world = trainer.actor_rollout_wg.world_size
    if len(workers) != world or {r['identity']['rank'] for r in workers} != set(range(world)):
        raise ValueError('Rolling native worker snapshot coverage is incomplete')
    saved = driver_snapshot(trainer, identity)
    driver_path = directory/'driver.pt'
    logical = digest(saved)
    driver_sha = atomic_save(saved, driver_path)
    manifest = dict(status='COMPLETE_NATIVE_ROLLING_STATE_SAVED',
        snapshot_kind='rolling_completed_outer', identity=identity, worker_states=workers,
        driver=dict(path=str(driver_path), sha256=driver_sha, logical_state_sha256=logical,
                    next_step=step+1, exact_reload_verified=False),
        hf_checkpoint=None, hf_identity=None, automatic_resume_supported=False,
        exact_executor_resume_verified=False,
        saved_components=dict(worker_model_shards=True, worker_parameters=True,
            worker_gradients=True, worker_buffers=True, adam_optimizer=True, scheduler=True,
            worker_python_numpy_torch_cuda_rng=True, driver_python_numpy_torch_rng=True,
            sampler_order_cursor_generator=True, tree_rng_and_algorithm_state=True,
            source_bytes=False, config_bytes=False, source_config_identity=True,
            executor_vllm_state=False),
        retention=dict(keep_completed=2, permanent_milestones_unchanged=[5,50,100,150,200]),
        limits='Native same-world boundary snapshot only; no reload performed, no automatic restart or executor/vLLM equivalence claim')
    manifest_path = directory/'manifest.json'
    manifest_sha = atomic_save(manifest, manifest_path, json_format=True)
    latest = dict(status='COMPLETE_NATIVE_ROLLING_STATE_SAVED', identity=identity,
        completed_outer_updates=step, next_outer_update=step+1,
        manifest_path=str(manifest_path), manifest_sha256=manifest_sha,
        automatic_resume_supported=False, exact_executor_resume_verified=False)
    pending = root/f'latest.json.pending-{os.getpid()}-{time.time_ns()}'
    try:
        with pending.open('x') as stream:
            json.dump(latest, stream, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, root/'latest.json')
        fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        pending.unlink(missing_ok=True)
    trainer._rolling_native_saved_outer = step

    # Publish the complete new boundary before pruning any older complete state.
    complete = []
    for candidate in root.iterdir():
        if candidate.is_symlink() or not candidate.is_dir():
            continue
        suffix = candidate.name.removeprefix('global_step_')
        if not candidate.name.startswith('global_step_') or not suffix.isdecimal():
            continue
        candidate_manifest = candidate/'manifest.json'
        if candidate_manifest.is_symlink() or not candidate_manifest.is_file():
            continue
        value = json.loads(candidate_manifest.read_text())
        prior_identity = value.get('identity', {})
        if (value.get('snapshot_kind') == 'rolling_completed_outer'
                and value.get('status') == 'COMPLETE_NATIVE_ROLLING_STATE_SAVED'
                and prior_identity.get('completed_step') == int(suffix)
                and all(prior_identity.get(key) == identity[key] for key in
                    ('job_id', 'run_id', 'code_commit', 'config_sha256', 'data_identity', 'model_identity'))):
            complete.append((int(suffix), candidate))
    for _, candidate in sorted(complete, reverse=True)[2:]:
        shutil.rmtree(candidate)
    print('TREE_ROLLING_NATIVE_STATE_SAVED=' + str(manifest_path), flush=True)


def install():
    from search_r1.baselines.controller import Controller
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    if getattr(RayPPOTrainer, '_rolling_native_state_installed', False):
        raise ValueError('Rolling native state hook already installed')
    original_validate = RayPPOTrainer._validate
    original_stage = Controller.stage_validation

    @functools.wraps(original_validate)
    def validate(self, *args, **kwargs):
        # A fifth-outer dev failure must not erase that completed update.
        # Install after the cost observer, so saving is outside validation cost.
        save_completed_boundary(self)
        return original_validate(self, *args, **kwargs)

    @functools.wraps(original_stage)
    def stage_validation(self, *args, **kwargs):
        # The original trainer calls this at every fully credited actor boundary.
        save_completed_boundary(self.trainer)
        return original_stage(self, *args, **kwargs)

    RayPPOTrainer._validate = validate
    Controller.stage_validation = stage_validation
    RayPPOTrainer._rolling_native_state_installed = True
