"""Preserve a synchronized refused batch without completing or rolling it back.

The files are failure evidence, not an automatically resumable full checkpoint.
Only the known all-rank policy rejection path invokes the worker snapshot.
"""
import json
import os
import time
from pathlib import Path

from .recovery import (atomic_json_save, atomic_torch_save, cpu_snapshot,
                       file_sha256, rng_state, worker_training_state)


def _folder(outer_step):
    root = os.environ.get('SEARCH_R1_BASELINE_AUDIT_DIR')
    if not root or type(outer_step) is not int or outer_step < 1:
        raise RuntimeError('Partial state requires a bound real outer identity')
    path = Path(root) / 'partial-updates' / f'outer_{outer_step:06d}'
    path.mkdir(parents=True, exist_ok=True)
    return path


def _adam_steps(state):
    steps = []
    for slot in state['optimizer']['state'].values():
        if not slot:
            continue  # A fresh optimizer may contain inspected but still empty slots.
        value = slot.get('step')
        if value is None:
            raise RuntimeError('Partial Adam slot has no actual step count')
        number = float(value)
        if not number.is_integer() or number < 0:
            raise RuntimeError('Partial Adam step is unreadable/nonfinite')
        steps.append(int(number))
    if steps and len(set(steps)) != 1:
        raise RuntimeError('Partial Adam slots have different actual write counts')
    return steps[0] if steps else 0


def preserve_worker(worker, data, rejection):
    """Called by all actors only after the original matched rejection collective."""
    import torch
    outer = data.meta_info.get('baseline_outer_step')
    completed = outer - 1
    record = rejection.record
    if (record.get('actual_optimizer_writes') is None or type(record['actual_optimizer_writes']) is not int
            or not 0 <= record['actual_optimizer_writes'] <= 20
            or record.get('outer_step') != outer):
        raise RuntimeError('Matched rejection has no exact partial optimizer/scheduler boundary')
    folder = _folder(outer)
    state = worker_training_state(worker.actor_module_fsdp, worker.actor_optimizer,
        worker.actor_lr_scheduler, worker.rank, worker.world_size, completed,
        worker.rollout_sharding_manager, complete_internal=True)
    step_error = None
    try:
        steps = _adam_steps(state)
    except BaseException as counter_error:
        steps = None
        step_error = repr(counter_error)
    state['pending_gradients'] = {name: cpu_snapshot(parameter.grad)
        for name, parameter in worker.actor_module_fsdp.named_parameters()}
    state['partial_update'] = dict(status='INCOMPLETE_OUTER_SAVED_NOT_COUNTED',
        outer_step=outer, in_progress_outer_id=outer, mini_index=record.get('mini_index'),
        phase=record.get('phase'), completed_outer_updates=completed,
        actual_partial_adam_steps=record['actual_optimizer_writes'],
        actual_adam_total=steps, adam_counter_error=step_error,
        prior_actual_adam_total=steps-record['actual_optimizer_writes'] if steps is not None else None,
        actual_extra_adam_beyond_complete_outers=(steps-completed*20
            if steps is not None else None), scheduler_step=worker.actor_lr_scheduler.last_epoch, rejection=record,
        automatic_rollback=False, automatic_resume=False, exact_resume_accepted=False)
    receipt = atomic_torch_save(state, folder / f'rank_{worker.rank}.pt')
    result = dict(status='PARTIAL_WORKER_STATE_SAVED', rank=worker.rank,
        pid=os.getpid(), state_receipt=receipt, partial_update=state['partial_update'],
        job_id=os.environ.get('SLURM_JOB_ID'), run_id=os.environ.get('SEARCH_R1_BASELINE_RUN_ID'),
        source_commit=os.environ.get('SEARCH_R1_BASELINE_COMMIT'),
        config_sha256=os.environ.get('SEARCH_R1_BASELINE_CONFIG_SHA256'))
    atomic_json_save(result, folder / f'rank_{worker.rank}.json')
    return result


def preserve_driver(trainer, batch, error, wait_seconds=300):
    """Save the actual advanced cursor and batch; never rewind or label it complete."""
    from omegaconf import OmegaConf
    outer = trainer.global_steps
    folder = _folder(outer)
    completed = getattr(trainer, 'actual_outer_updates', 0)
    sampler = trainer.recovery_sampler.state_dict()
    if completed != outer - 1 or sampler['cursor'] != outer * 512:
        raise RuntimeError('Failed outer progress differs from the real data cursor')
    controller = trainer.baseline_controller
    algorithm = dict(method=controller.method, gigpo_parameters=OmegaConf.to_container(
        trainer.config.baselines.gigpo, resolve=True), stage_state=controller.stage.state_dict(),
        groups='stateless_per_rollout_with_step_ids_and_advantage_broadcast',
        loss_reduction=trainer.config.baselines.loss_reduction)
    state = dict(schema=1, status='INCOMPLETE_OUTER_SAVED_NOT_COUNTED', failed_outer_step=outer,
        actual_completed_outer_updates=completed, source=trainer.config.actor_rollout_ref.model.path,
        source_revision='aa8e72537993ba99e69dfaafa59ed015b17504d1', sampler=sampler,
        rng=rng_state(), kl_controller=dict(vars(trainer.kl_ctrl)), algorithm_state=algorithm,
        config=OmegaConf.to_container(trainer.config, resolve=True), error=repr(error),
        batch=cpu_snapshot(dict(tensors=dict(batch.batch), non_tensors=batch.non_tensor_batch,
            meta_info=batch.meta_info)), automatic_rollback=False, automatic_resume=False)
    driver_receipt = atomic_torch_save(state, folder / 'driver.pt')
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        if all((folder / f'rank_{rank}.json').is_file() or (folder / f'rank_{rank}-save-failed.json').is_file()
                for rank in range(8)):
            break
        time.sleep(1)
    # Read available successes AND failures even at timeout. Never report an
    # empty list when seven real rank snapshots already exist.
    coverage = collect_worker_receipts(folder, completed, outer)
    result = dict(status='PARTIAL_DRIVER_SAVED_WORKER_COVERAGE_UNVERIFIED', failed_outer_step=outer,
        actual_completed_outer_updates=completed, data_cursor=sampler['cursor'],
        driver_receipt=driver_receipt, workers=coverage['workers'],
        worker_save_failures=coverage['failures'], worker_receipt_errors=coverage['errors'],
        missing_worker_ranks=coverage['missing'], error=repr(error),
        automatic_rollback=False, automatic_resume=False, exact_resume_accepted=False)
    records = coverage['workers']
    if len(records) == 8 and not any(coverage[key] for key in ('failures', 'errors', 'missing')):
        totals = {r['partial_update']['actual_adam_total'] for r in records}
        writes = {r['partial_update']['actual_partial_adam_steps'] for r in records}
        readable = (len(totals) == len(writes) == 1 and None not in totals
            and all(r['partial_update']['scheduler_step'] == completed for r in records)
            and all(r['partial_update']['adam_counter_error'] is None for r in records))
        if readable:
            result.update(status='ALL8_PARTIAL_STATES_SAVED_NOT_COMPLETED_NOT_RESUME_ACCEPTED',
                actual_partial_adam_steps=next(iter(writes)), actual_adam_total=next(iter(totals)),
                actual_extra_adam_beyond_complete_outers=next(iter(totals))-completed*20)
        else:
            result.update(status='ALL8_PARTIAL_STATES_SAVED_COUNTERS_OR_SCHEDULER_UNACCEPTED',
                actual_adam_totals=list(totals), actual_partial_writes=list(writes))
    atomic_json_save(result, folder / 'partial-result.json')
    return result


def collect_worker_receipts(folder, completed, outer):
    """Bind every available original receipt to this job/run/config and rank path."""
    folder = Path(folder)
    binding = dict(job_id=os.environ.get('SLURM_JOB_ID'), run_id=os.environ.get('SEARCH_R1_BASELINE_RUN_ID'),
        source_commit=os.environ.get('SEARCH_R1_BASELINE_COMMIT'),
        config_sha256=os.environ.get('SEARCH_R1_BASELINE_CONFIG_SHA256'))
    if any(not value for value in binding.values()):
        raise RuntimeError('Partial receipt binding is incomplete')
    workers, failures, errors, missing = [], [], [], []
    for rank in range(8):
        success_path = folder / f'rank_{rank}.json'
        failure_path = folder / f'rank_{rank}-save-failed.json'
        if not success_path.is_file() and not failure_path.is_file():
            missing.append(rank)
        for path, success in ((success_path, True), (failure_path, False)):
            if not path.is_file():
                continue
            try:
                record = json.loads(path.read_text())
                if record.get('rank') != rank or any(record.get(key) != value for key, value in binding.items()):
                    raise ValueError('Worker receipt current job/run/source/config differs')
                if success:
                    part = record['partial_update']
                    receipt = record['state_receipt']
                    expected = folder / f'rank_{rank}.pt'
                    if (record['status'] != 'PARTIAL_WORKER_STATE_SAVED'
                            or part['completed_outer_updates'] != completed or part['outer_step'] != outer
                            or part['in_progress_outer_id'] != outer
                            or type(part['actual_partial_adam_steps']) is not int
                            or not 0 <= part['actual_partial_adam_steps'] <= 20
                            or Path(receipt['path']) != expected or expected.is_symlink()
                            or expected.stat().st_size != receipt['bytes']
                            or file_sha256(expected) != receipt['sha256']):
                        raise ValueError('Partial worker snapshot boundary/path/bytes differ')
                    workers.append(record)
                else:
                    if (record['status'] != 'PARTIAL_WORKER_STATE_SAVE_FAILED'
                            or record.get('outer_step') != outer):
                        raise ValueError('Partial failure receipt boundary differs')
                    failures.append(record)
            except Exception as read_error:
                errors.append(dict(rank=rank, path=str(path), error=repr(read_error)))
    return dict(workers=workers, failures=failures, errors=errors, missing=missing)
