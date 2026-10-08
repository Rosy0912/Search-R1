"""Best-effort honest failure evidence; never performs model/distributed computation."""
import copy
import json

def preserve_first_refusal(trainer, run, progress, error, *, driver_snapshot, atomic_torch_save,
                           json_save, cpu_snapshot, rng_state):
    """Record actual partial driver state for every refusal, without masking it.

    An unknown original model/collective error never dispatches worker recovery.
    A known matched refusal has all ranks stopped at the same boundary and the
    rank save implementation only reads local shards; each rank reports failure.
    """
    from calibration_boundary_v2 import is_matched_refusal
    report = dict(status='REFUSED_BEFORE_FIRST_ADAM', original_error=repr(error),
        failed_stage=progress.get('stage'), driver_update_actor_calls=0, actual_adam_writes=0,
        expected_scheduler_step=0, actual_scheduler_steps_by_rank={},
        all_rank_zero_adam_scheduler0_verified=False,
        retained_training_updates_contributed=0, exact_resume_accepted=False,
        validated_resume_checkpoint=False, automatic_resume_allowed=False,
        actual_driver_cursor=getattr(trainer.recovery_sampler, 'cursor', None),
        worker_snapshots=[], missing_ranks=list(range(8)), duplicate_ranks=[], preservation_errors=[])
    safe = progress.get('workers_safe_to_snapshot', False) or is_matched_refusal(error)
    report['known_matched_or_completed_worker_boundary'] = safe
    # Driver save is first and unconditional; no rank observer can replace the refusal.
    driver = dict(actual_cursor=report['actual_driver_cursor'], failed_stage=report['failed_stage'],
                  original_error=report['original_error'], exact_resume_accepted=False)
    for name, operation in (
        ('rng', lambda: cpu_snapshot(rng_state())),
        ('sampler', lambda: cpu_snapshot(trainer.recovery_sampler.state_dict())),
        ('complete_available_driver', lambda: driver_snapshot(trainer, progress.get('batch'))),
        ('fetched_batch', lambda: cpu_snapshot(progress.get('fetched_batch'))),
        ('partial_generated', lambda: None if progress.get('generated') is None else dict(
            tensors=cpu_snapshot(dict(progress['generated'].batch.items())),
            non_tensors=copy.deepcopy(progress['generated'].non_tensor_batch),
            metadata=copy.deepcopy(progress['generated'].meta_info))),
        ('available_sources', lambda: copy.deepcopy(progress.get('sources'))),
        ('available_turns', lambda: copy.deepcopy(progress.get('turns')))):
        try:
            driver[name] = operation()
        except BaseException as save_error:
            report['preservation_errors'].append(name + ': ' + repr(save_error))
    try:
        report['driver'] = atomic_torch_save(driver, run / 'probability-rejection-state/driver.pt')
    except BaseException as save_error:
        report['preservation_errors'].append('driver_save: ' + repr(save_error))
    if safe:
        try:
            rows = trainer.actor_rollout_wg.calibration_preserve_zero_adam()
            report['worker_snapshots'] = rows
            ranks = [row['rank'] for row in rows]
            report['duplicate_ranks'] = sorted({rank for rank in ranks if ranks.count(rank) > 1})
            report['actual_scheduler_steps_by_rank'] = {str(row['rank']): row.get('scheduler_step') for row in rows}
            covered = {row['rank'] for row in rows if row.get('saved') and row.get('zero_adam_verified')}
            report['missing_ranks'] = sorted(set(range(8)) - covered)
            report['all_rank_zero_adam_scheduler0_verified'] = (len(rows) == 8 and not report['missing_ranks']
                and {row['rank'] for row in rows} == set(range(8)))
        except BaseException as save_error:
            report['preservation_errors'].append('rank_save_rpc: ' + repr(save_error))
    else:
        report['worker_preservation_omitted_reason'] = 'UNKNOWN_UNSYNCHRONIZED_ORIGINAL_FAILURE_NO_NEW_COLLECTIVE_OR_RPC'
        report['status'] = 'UNKNOWN_FAILURE_DRIVER_RETAINED_WORKER_COVERAGE_INCOMPLETE'
    try:
        json_save(run / 'probability-rejection-state/result.json', report)
    except BaseException as save_error:
        # Last local fallback journals the original refusal and missing coverage.
        try:
            (run / 'probability-rejection-state').mkdir(parents=True, exist_ok=True)
            (run / 'probability-rejection-state/emergency-result.json').write_text(
                json.dumps(dict(report, journal_error=repr(save_error)), indent=2, allow_nan=False))
        except BaseException:
            pass  # Original refusal remains the driver failure, never a fabricated successful save.
    return report

