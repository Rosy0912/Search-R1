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


def _folder(outer_step, attempt_id=None, refusal_index=None):
    root = os.environ.get('SEARCH_R1_BASELINE_AUDIT_DIR')
    if not root or type(outer_step) is not int or outer_step < 1:
        raise RuntimeError('Partial state requires a bound real outer identity')
    path = Path(root) / 'partial-updates' / f'outer_{outer_step:06d}'
    if attempt_id is not None:
        if type(attempt_id) is not int or attempt_id < 1:
            raise ValueError('Invalid actual unique attempt identity')
        path = Path(root) / 'partial-updates' / f'attempt_{attempt_id:06d}'
        if refusal_index is not None:
            if type(refusal_index) is not int or refusal_index < 1:
                raise ValueError('Invalid unique refusal identity')
            path = path / f'refusal_{refusal_index:02d}'
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


def preserve_worker(worker, data, rejection, refusal_index=None):
    """Called by all actors only after the original matched rejection collective."""
    import torch
    outer = data.meta_info.get('baseline_outer_step')
    completed = data.meta_info.get('fresh200_completed_before', outer - 1)
    attempt_id = data.meta_info.get('fresh200_attempt_id')
    record = rejection.record
    if (record.get('actual_optimizer_writes') is None or type(record['actual_optimizer_writes']) is not int
            or not 0 <= record['actual_optimizer_writes'] <= 20
            or record.get('outer_step') != outer):
        raise RuntimeError('Matched rejection has no exact partial optimizer/scheduler boundary')
    folder = _folder(outer, attempt_id, refusal_index)
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
    state['runtime_metadata'] = worker.checkpoint_runtime_metadata()
    state['partial_update'] = dict(status='INCOMPLETE_OUTER_SAVED_NOT_COUNTED',
        outer_step=outer, in_progress_outer_id=outer, mini_index=record.get('mini_index'),
        phase=record.get('phase'), completed_outer_updates=completed,
        actual_partial_adam_steps=record['actual_optimizer_writes'],
        actual_adam_total=steps, adam_counter_error=step_error,
        prior_actual_adam_total=steps-record['actual_optimizer_writes'] if steps is not None else None,
        scheduler_step=worker.actor_lr_scheduler.last_epoch, rejection=record,
        attempt_id=attempt_id, refusal_index=refusal_index,
        resume_point='diagnostic_before_gradient_clear; never resume rejected gradients',
        automatic_rollback=False, automatic_resume=False, exact_resume_accepted=False)
    if (folder / f'rank_{worker.rank}.pt').exists() or (folder / f'rank_{worker.rank}.json').exists():
        raise ValueError('Refusing to overwrite a real partial attempt snapshot')
    receipt = atomic_torch_save(state, folder / f'rank_{worker.rank}.pt')
    result = dict(status='PARTIAL_WORKER_STATE_SAVED', rank=worker.rank,
        pid=os.getpid(), state_receipt=receipt, partial_update=state['partial_update'],
        job_id=os.environ.get('SLURM_JOB_ID'), run_id=os.environ.get('SEARCH_R1_BASELINE_RUN_ID'),
        source_commit=os.environ.get('SEARCH_R1_BASELINE_COMMIT'),
        config_sha256=os.environ.get('SEARCH_R1_BASELINE_CONFIG_SHA256'))
    atomic_json_save(result, folder / f'rank_{worker.rank}.json')
    return result


def rejection_disposition(record):
    """Expired old-policy support always abandons the remaining rollout reuse."""
    # Diagnostic evidence failure is not a numerical opportunity refusal. The
    # original matched IO decision takes precedence even when phase/reason
    # still describes an otherwise skippable monitor or expired old rollout.
    if (record.get('global_diagnostic_io_failed') is True
            or record.get('matched_guard') == 'ORIGINAL_ALL_REDUCE_MAX_MONITOR_IO_REFUSAL'):
        return 'fail_closed'
    phase = record.get('phase', '')
    if type(record.get('mini_index')) is not int or not 0 <= record['mini_index'] < 20:
        return 'fail_closed'
    if record.get('rejection_reason') in ('OLD_POLICY_TOKEN_KL',
            'CALIBRATION_CANDIDATE_OLD_POLICY_TOKEN_KL', 'NEGATIVE_JOINT_ACTION_TAIL_COUNT'):
        return 'abandon' if phase in ('before', 'after') else 'fail_closed'
    phases = {'before', 'after', 'global_gradient_before_adam',
              'gradient_action_prefix_schema', 'gradient_action_prefix_outputs'}
    phases.update(f'gradient_micro{i}{suffix}' for i in range(8) for suffix in ('', '_loss'))
    # Probe enter/exit catches may contain unknown original state/collective
    # failures. Their typed envelope does not authorize a safe numerical skip.
    phases.update(f'{side}_state' for side in ('before', 'after'))
    phases.update(f'{side}_{kind}{i}' for side in ('before', 'after')
                  for kind in ('schema', 'stats') for i in range(16))
    phases.update(f'{side}_action_prefix_micro{i}{suffix}' for side in ('before', 'after')
                  for i in range(16) for suffix in ('_schema', '_outputs'))
    if phase in phases:
        return 'skip_opportunity'
    return 'fail_closed'


def continue_worker_after_rejection(worker, data, rejection, refusal_index):
    """Only a typed original all-rank refusal may enter these new matched local stages."""
    import torch
    from .policy_safety import PolicySafetyRejected
    from verl.workers.fsdp_workers import _checkpoint_matched_stage
    if not isinstance(rejection, PolicySafetyRejected):
        raise TypeError('Unknown exceptions cannot become a safe skip')
    receipt, error = None, None
    try:
        receipt = preserve_worker(worker, data, rejection, refusal_index)
    except Exception as local_error:
        error = local_error
    # Unknown failure of this original collective propagates, with no retry MAX.
    _checkpoint_matched_stage('safe_skip/preserve_actual_partial', error)
    error, actual_total = None, None
    try:
        before = worker.checkpoint_pause_identity()
        worker.actor_optimizer.zero_grad(set_to_none=True)
        if any(parameter.grad is not None for parameter in worker.actor_module_fsdp.parameters()):
            raise ValueError('Rejected pending gradients were not cleared')
        tensors = list(worker.actor_module_fsdp.parameters()) + list(worker.actor_module_fsdp.buffers())
        tensors += [parameter._local_shard for parameter in worker.actor_module_fsdp.parameters()
                    if hasattr(parameter, '_local_shard')]
        tensors += [value for slot in worker.actor_optimizer.state.values()
                    for value in slot.values() if isinstance(value, torch.Tensor)]
        if not all(bool(torch.isfinite(value).all()) for value in tensors):
            raise ValueError('Written model/Adam is nonfinite; fresh continuation forbidden')
        actual_total = receipt['partial_update']['actual_adam_total']
        if (actual_total is None or actual_total != data.meta_info['fresh200_adam_total_before'] +
                rejection.record['actual_optimizer_writes']
                or worker.actor_lr_scheduler.last_epoch != data.meta_info['fresh200_completed_before']):
            raise ValueError('Actual partial Adam or untouched scheduler boundary differs')
        after = worker.checkpoint_pause_identity()
        if before != after:
            raise ValueError('Gradient clear changed actual model/Adam/scheduler/RNG/PID')
    except Exception as local_error:
        error = local_error
    _checkpoint_matched_stage('safe_skip/clear_gradients_and_verify_finite', error)
    # Verify all ranks really stand at the same write/opportunity boundary. This
    # is reached only after the known refusal and successful local-state stages.
    signature = torch.tensor([data.meta_info['fresh200_attempt_id'], actual_total,
        rejection.record['actual_optimizer_writes'], rejection.record['mini_index'],
        worker.actor_lr_scheduler.last_epoch], dtype=torch.int64, device='cuda')
    lowest, highest = signature.clone(), signature.clone()
    torch.distributed.all_reduce(lowest, op=torch.distributed.ReduceOp.MIN)
    torch.distributed.all_reduce(highest, op=torch.distributed.ReduceOp.MAX)
    if not torch.equal(lowest, highest):
        raise RuntimeError('All-rank partial opportunity/write boundary differs')
    result, error = None, None
    try:
        folder = _folder(data.meta_info['baseline_outer_step'], data.meta_info['fresh200_attempt_id'], refusal_index)
        # Gradient clear was verified to leave all model/Adam/RNG tensor
        # identities unchanged. Keep those immutable original bytes by SHA,
        # rather than writing a second multi-GB tensor set for every refusal.
        state = dict(schema='FRESH200_POSTCLEAR_CONTINUATION_REFERENCE_V1',
            logical_step=data.meta_info['fresh200_completed_before'],
            tensor_payload_from=receipt['state_receipt'], pause_identity=after,
            rng=rng_state(), runtime_metadata=worker.checkpoint_runtime_metadata())
        state['prior_completed_attempt_ledger'] = state['runtime_metadata']['fresh200_worker_ledger']
        state['continuation_boundary'] = dict(attempt_id=data.meta_info['fresh200_attempt_id'],
            refusal_index=refusal_index, gradients_cleared=True, model_and_adam_finite=True,
            actual_adam_total=actual_total, rejected_opportunity=rejection.record['mini_index'],
            resume_point='next_opportunity_or_fresh_batch; rejected_gradients_are_discarded',
            automatic_rollback=False, automatic_resume=False, complete_checkpoint=False,
            same_worker_pid=os.getpid())
        path = folder / f'rank_{worker.rank}-continue.pt'
        if path.exists():
            raise ValueError('Refusing to overwrite a safe continuation state')
        safe_state = atomic_torch_save(state, path)
        result = dict(receipt, safe_to_continue=True, continuation_state=safe_state,
                      gradients_cleared=True, model_and_adam_finite=True)
        atomic_json_save(result, folder / f'rank_{worker.rank}-continue.json')
    except Exception as local_error:
        error = local_error
    _checkpoint_matched_stage('safe_skip/save_postclear_continuation_state', error)
    return result


def _read_bound_torch_state(path, expected_sha256, expected_bytes=None, label='snapshot'):
    """Hash and parse the same regular no-follow FD, with stable fstat identity."""
    import hashlib
    import stat
    import torch
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, 'rb') as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or (expected_bytes is not None and before.st_size != expected_bytes):
            raise ValueError(label + ' bytes changed')
        identity = lambda value: (value.st_dev, value.st_ino, value.st_size,
                                   value.st_mtime_ns, value.st_ctime_ns)
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1048576), b''):
            digest.update(chunk)
        if digest.hexdigest() != expected_sha256:
            raise ValueError(label + ' bytes changed')
        stream.seek(0)
        state = torch.load(stream, map_location='cpu', weights_only=False)
        if identity(os.fstat(stream.fileno())) != identity(before):
            raise ValueError(label + ' changed while hashing/parsing the same FD')
    return state


def materialize_continuation_state(path, expected_sha256):
    """Read a postclear snapshot and its immutable tensor payload; never resume it."""
    reference = _read_bound_torch_state(path, expected_sha256, label='Postclear continuation snapshot')
    boundary = reference['continuation_boundary']
    if (reference.get('schema') != 'FRESH200_POSTCLEAR_CONTINUATION_REFERENCE_V1'
            or boundary.get('gradients_cleared') is not True
            or boundary.get('model_and_adam_finite') is not True
            or boundary.get('complete_checkpoint') is not False
            or boundary.get('automatic_resume') is not False):
        raise ValueError('Postclear snapshot is not a verified safe evidence boundary')
    receipt = reference['tensor_payload_from']
    state = _read_bound_torch_state(receipt['path'], receipt['sha256'], receipt['bytes'],
                                   label='Immutable original model/Adam payload')
    if (state['pause_identity'] != reference['pause_identity']
            or state['logical_step'] != reference['logical_step']
            or state['partial_update']['attempt_id'] != boundary['attempt_id']
            or state['partial_update']['refusal_index'] != boundary['refusal_index']):
        raise ValueError('Original tensor payload is not this unchanged postclear state')
    state.pop('pending_gradients')
    state.update(rng=reference['rng'], runtime_metadata=reference['runtime_metadata'],
                 continuation_boundary=boundary, prior_completed_attempt_ledger=
                 reference['prior_completed_attempt_ledger'])
    return state


def save_attempt_driver(trainer, batch, outcome):
    """Bind actual finished/skipped attempt to its driver/batch/RNG/algorithm state."""
    from omegaconf import OmegaConf
    from .fresh200 import driver_flow_state
    from .instruct300 import _live_gigpo_state
    attempt = outcome['attempt']
    folder = _folder(attempt['completed_before'] + 1, attempt['attempt_id'])
    target = folder / 'driver.pt'
    if target.exists():
        raise ValueError('Refusing to overwrite a unique attempt driver state')
    state = dict(schema=1, status='COUNTED_COMPLETE_OUTER' if outcome['counted_complete'] else
        'ATTEMPT_NOT_COUNTED_FRESH_BATCH_NEXT', source=trainer.config.actor_rollout_ref.model.path,
        source_revision='aa8e72537993ba99e69dfaafa59ed015b17504d1', driver_pid=os.getpid(),
        attempt=attempt, fresh200_attempt_ledger=outcome['ledger'],
        sampler=trainer.recovery_sampler.state_dict(), rng=rng_state(),
        fresh200_driver_flow=driver_flow_state(trainer), kl_controller=dict(vars(trainer.kl_ctrl)),
        algorithm_state=_live_gigpo_state(trainer, trainer.actual_outer_updates),
        config=OmegaConf.to_container(trainer.config, resolve=True),
        batch=cpu_snapshot(dict(tensors=dict(batch.batch), non_tensors=batch.non_tensor_batch,
                               meta_info=batch.meta_info)), automatic_rollback=False)
    receipt = atomic_torch_save(state, target)
    atomic_json_save(dict(attempt=attempt, ledger=outcome['ledger'], driver_receipt=receipt), folder / 'attempt-result.json')
    return receipt


def preserve_driver(trainer, batch, error, wait_seconds=300):
    """Save the actual advanced cursor and batch; never rewind or label it complete."""
    from omegaconf import OmegaConf
    outer = trainer.global_steps
    attempt_id = batch.meta_info.get('fresh200_attempt_id')
    folder = _folder(outer, attempt_id)
    completed = getattr(trainer, 'actual_outer_updates', 0)
    sampler = trainer.recovery_sampler.state_dict()
    if completed != outer - 1:
        raise RuntimeError('Failed outer progress differs from the real data cursor')
    if attempt_id is not None:
        from .fresh200 import initialize_attempt_ledger, _sampler_position
        ledger = initialize_attempt_ledger(trainer)
        active = ledger['active_attempt']
        if (active is None or active['attempt_id'] != attempt_id
                or active['sampler_position'] != _sampler_position(trainer.recovery_sampler)):
            raise RuntimeError('Failed attempt is not the actual advanced sampler/ledger position')
    elif sampler['cursor'] != outer * 512:
        raise RuntimeError('Legacy failed outer progress differs from real data cursor')
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
        attempt_id=attempt_id, fresh200_attempt_ledger=ledger if attempt_id is not None else None,
        batch=cpu_snapshot(dict(tensors=dict(batch.batch), non_tensors=batch.non_tensor_batch,
            meta_info=batch.meta_info)), automatic_rollback=False, automatic_resume=False)
    if (folder / 'driver.pt').exists():
        raise ValueError('Refusing to overwrite an actual failed attempt driver state')
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
                actual_partial_adam_steps=next(iter(writes)), actual_adam_total=next(iter(totals)))
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
