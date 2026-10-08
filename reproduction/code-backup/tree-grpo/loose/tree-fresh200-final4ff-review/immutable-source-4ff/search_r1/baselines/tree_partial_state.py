"""Retain a stopped outer as evidence, without credit, rollback, or auto-resume."""
import os
import time
from pathlib import Path

from .training_state import atomic_save, cpu_copy, digest, file_sha, rng_state
from .policy_safety_runtime import PolicySafetyRefusal

PROTOCOL = 'tree_partial_state_v1'


def partial_context(trainer):
    outer, completed = trainer.global_steps, trainer.actual_outer_updates
    if type(outer) is not int or type(completed) is not int or outer != completed + 1:
        raise ValueError('Partial context requires the real attempted/completed outer boundary')
    identity = dict(job_id=os.environ['SLURM_JOB_ID'], run_id=os.environ['SEARCH_R1_BASELINE_RUN_ID'],
                    code_commit=os.environ['SEARCH_R1_BASELINE_COMMIT'],
                    config_sha256=os.environ['SEARCH_R1_BASELINE_CONFIG_SHA256'],
                    data_identity=os.environ['SEARCH_R1_BASELINE_DATA_IDENTITY'],
                    model_identity=os.environ['SEARCH_R1_MODEL_IDENTITY'],
                    training_protocol=trainer.config.baselines.training_protocol,
                    attempted_outer=outer, completed_outer=completed)
    directory = Path(trainer.config.trainer.default_local_dir) / 'training_state' / f'partial_outer_{outer:06d}'
    return dict(protocol=PROTOCOL, identity=identity, directory=str(directory))


def partial_driver_snapshot(trainer, context):
    """Different format from complete boundary snapshots: consumed batch is kept."""
    identity = context['identity']
    if (context['protocol'] != PROTOCOL or identity['attempted_outer'] != trainer.global_steps
            or identity['completed_outer'] != trainer.actual_outer_updates):
        raise ValueError('Partial driver provenance differs from the actual controller')
    controller = trainer.baseline_controller
    return cpu_copy(dict(version=3, snapshot_kind=PROTOCOL, identity=identity,
        attempted_outer=trainer.global_steps, completed_outer_updates=trainer.actual_outer_updates,
        outer_completed=False, automatic_rollback=False, automatic_resume_supported=False,
        consumed_sampler=trainer.checkpoint_sampler.state_dict(), rng=rng_state(),
        algorithm=dict(method=controller.method, stage=controller.stage.state_dict(),
                       tree_rng=None if controller.tree_rng is None else controller.tree_rng.getstate()),
        fresh_initialization_provenance=getattr(trainer, 'fresh_initialization_provenance', None),
        recovery_provenance=getattr(trainer, 'recovery_provenance', None)))


def _preserve_driver_and_batch(trainer, batch, context):
    directory = Path(context['directory'])
    saved = partial_driver_snapshot(trainer, context)
    driver_path = directory / 'driver.pt'
    driver = dict(path=str(driver_path), sha256=atomic_save(saved, driver_path),
                  logical_state_sha256=digest(saved), completed_outer_updates=trainer.actual_outer_updates,
                  consumed_cursor=saved['consumed_sampler']['cursor'],
                  consumed_epoch=saved['consumed_sampler']['epoch'])
    # Preserve every tensor, binding, mask, and diagnostic field, including invalid
    # numeric values. Never truncate/filter the failed batch into a healthy batch.
    actor_inputs = cpu_copy(dict(batch={key: batch.batch[key] for key in batch.batch.keys()},
                                non_tensor_batch=batch.non_tensor_batch, meta_info=batch.meta_info))
    batch_path = directory / 'actor-inputs.pt'
    inputs = dict(path=str(batch_path), sha256=atomic_save(actor_inputs, batch_path),
                  rows=len(batch), fields=list(actor_inputs['batch']),
                  bytes=batch_path.stat().st_size)
    return driver, inputs


def _validate_workers(envelope, context, world_size):
    if (envelope.get('protocol') != PROTOCOL or envelope.get('identity') != context['identity']
            or envelope.get('status') != 'PARTIAL_WORKER_STATE_SAVED'
            or envelope.get('outer_completed') is not False or envelope.get('automatic_rollback') is not False):
        raise ValueError('Partial worker envelope identity/control mismatch')
    workers = envelope['worker_states']
    if (not isinstance(workers, list) or len(workers) != world_size
            or {r['rank'] for r in workers} != set(range(world_size))):
        raise ValueError('Partial worker rank coverage is incomplete')
    root = Path(context['directory']).resolve()
    for r in workers:
        if (type(r['rank']) is not int or type(r['world_size']) is not int or r['world_size'] != world_size
                or r['identity'] != dict(context['identity'], rank=r['rank'], world_size=world_size)
                or type(r['committed_optimizer_calls']) is not int or not 0 <= r['committed_optimizer_calls'] <= 20
                or type(r['scheduler_step_before_actual']) is not int
                or type(r['scheduler_step_after_actual']) is not int
                or r['scheduler_step_before_actual'] != context['identity']['completed_outer']
                or r['scheduler_step_after_actual'] not in (r['scheduler_step_before_actual'],
                                                            r['scheduler_step_before_actual'] + 1)
                or type(r['state_finite']) is not bool or r['status'] != 'PARTIAL_WORKER_STATE_SAVED'):
            raise ValueError('Partial worker state/count provenance is incomplete or invalid')
        path = Path(r['path'])
        if (path.is_symlink() or not path.is_file() or path.resolve().parent != root
                or path.name != f"worker-{r['rank']:02d}.pt"
                or type(r['bytes']) is not int or path.stat().st_size != r['bytes']
                or file_sha(path) != r['sha256']):
            raise ValueError('Partial worker persisted bytes do not match the direct receipt')
    return workers


def stop_on_partial_actor_result(trainer, batch, actor_output):
    """Called before metric reduction, after_actor, scheduler/outer credit on driver."""
    envelope = actor_output.meta_info.get('tree_partial_state')
    if envelope is None:
        return
    started = time.monotonic()
    context = partial_context(trainer)
    error, workers = None, []
    # Even an incomplete/invalid worker envelope retains actual driver/batch
    # evidence. Its missing worker states must never be claimed as saved.
    driver, inputs = _preserve_driver_and_batch(trainer, batch, context)
    try:
        workers = _validate_workers(envelope, context, trainer.actor_rollout_wg.world_size)
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    manifest = dict(status=('PARTIAL_OUTER_STATE_SAVED_CONTROLLED_STOP' if error is None
                            else 'PARTIAL_OUTER_STATE_INCOMPLETE_CONTROLLED_STOP'),
        protocol=PROTOCOL, identity=context['identity'], worker_outcome=envelope,
        worker_states_verified=(error is None), verification_error=error,
        actual_optimizer_calls_per_rank={str(r['rank']): r['committed_optimizer_calls'] for r in workers},
        actual_scheduler_steps_per_rank={str(r['rank']): r['scheduler_step_after_actual'] for r in workers},
        all_saved_worker_states_finite=(all(r['state_finite'] for r in workers) if error is None else None),
        outer_completed=False, completed_outer_updates=trainer.actual_outer_updates,
        attempted_outer=trainer.global_steps, driver=driver, actor_inputs=inputs,
        automatic_rollback=False, automatic_resume_supported=False,
        driver_preservation_and_verification_seconds=time.monotonic()-started,
        limits='Actual stopped model/Adam/scheduler/RNG/consumed batch are evidence; no executor resume or HF reload claim')
    path = Path(context['directory']) / 'manifest.json'
    sha = atomic_save(manifest, path, json_format=True)
    trainer.partial_outer_evidence = dict(path=str(path), sha256=sha, status=manifest['status'])
    raise PolicySafetyRefusal(f"Stopped outer {trainer.global_steps}; complete outer remains {trainer.actual_outer_updates}; "
                              f"actual partial state: {path}; status {manifest['status']}")


def preserve_unmatched_actor_failure(trainer, batch, error):
    """Driver-only best effort; never start collectives after an unpaired failure."""
    context = partial_context(trainer)
    context['directory'] = str(Path(context['directory']) / 'unmatched-worker-failure')
    driver, inputs = _preserve_driver_and_batch(trainer, batch, context)
    path = Path(context['directory']) / 'manifest.json'
    manifest = dict(status='UNMATCHED_WORKER_FAILURE_DRIVER_ONLY', protocol=PROTOCOL,
        identity=context['identity'], driver=driver, actor_inputs=inputs,
        failure=f'{type(error).__name__}: {error}'[:8192],
        actual_optimizer_calls_per_rank=None, worker_states_verified=False,
        outer_completed=False, automatic_rollback=False, automatic_resume_supported=False,
        limits='Worker failure was not a synchronized controlled brake; partial worker state/counts remain unknown')
    atomic_save(manifest, path, json_format=True)
    return str(path)
