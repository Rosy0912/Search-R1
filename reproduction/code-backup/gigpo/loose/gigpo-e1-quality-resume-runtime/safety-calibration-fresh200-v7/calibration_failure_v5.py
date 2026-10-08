"""Local failure journal for actual partial attempts; no worker RPC or collective."""
import copy


def preserve_attempt_failure(trainer, run, progress, error, *, driver_snapshot, json_save):
    """Retain available driver/rank returns without masking an unknown failure.

    A successful source receipt is the only continuation authority. This failure
    journal authorizes neither rollback nor automatic replay of an active attempt.
    """
    from calibration_failure_v2 import snapshot_resolved_rank_outputs
    from search_r1.baselines.recovery import atomic_torch_save, cpu_snapshot, rng_state
    report = dict(status='ATTEMPT_FAILED_DRIVER_RETAINED_NOT_COUNTED_NOT_RESUME_ACCEPTED',
        attempt_id=progress.get('attempt_id'), failed_stage=progress.get('stage'), original_error=repr(error),
        actual_driver_cursor=getattr(trainer.recovery_sampler, 'cursor', None),
        completed_outer_updates=getattr(trainer, 'actual_outer_updates', None),
        update_actor_execute_started=progress.get('first_adam_possible'),
        last_all_rank_await_successful=progress.get('workers_safe_to_snapshot', False),
        worker_recovery_rpc_dispatched=False, automatic_rollback=False,
        automatic_resume_allowed=False, exact_resume_accepted=False, preservation_errors=[])
    saved = dict(report)
    operations = (
        ('rng', lambda: cpu_snapshot(rng_state())),
        ('sampler', lambda: cpu_snapshot(trainer.recovery_sampler.state_dict())),
        ('attempt_ledger', lambda: copy.deepcopy(getattr(trainer, 'fresh200_attempt_ledger', None))),
        ('attempt_outcome', lambda: copy.deepcopy(progress.get('attempt_outcome'))),
        ('complete_available_driver', lambda: driver_snapshot(trainer, progress.get('batch'))),
        ('fetched_batch', lambda: cpu_snapshot(progress.get('fetched_batch'))),
        ('available_batch', lambda: None if progress.get('batch') is None else dict(
            tensors=cpu_snapshot(dict(progress['batch'].batch.items())),
            non_tensors=copy.deepcopy(progress['batch'].non_tensor_batch),
            metadata=copy.deepcopy(progress['batch'].meta_info))),
        ('resolved_worker_rank_outputs', lambda: snapshot_resolved_rank_outputs(progress, cpu_snapshot)),
        ('worker_rpc_boundary_v4', lambda: copy.deepcopy(progress.get('worker_rpc_boundary_v4'))),
        ('available_sources', lambda: copy.deepcopy(progress.get('sources'))),
        ('available_turns', lambda: copy.deepcopy(progress.get('turns'))),
        ('active_rollout_states', lambda: None if progress.get('active_rollout_states') is None else
            cpu_snapshot([vars(state) for state in progress['active_rollout_states']])))
    for name, operation in operations:
        try:
            saved[name] = operation()
        except BaseException as save_error:
            report['preservation_errors'].append(name + ': ' + repr(save_error))
    folder = run / 'attempt-failures' / ('attempt_' + str(progress.get('attempt_id')))
    try:
        report['driver'] = atomic_torch_save(saved, folder / 'driver.pt')
    except BaseException as save_error:
        report['preservation_errors'].append('driver_save: ' + repr(save_error))
    try:
        json_save(folder / 'result.json', report)
    except BaseException:
        pass
    return report
