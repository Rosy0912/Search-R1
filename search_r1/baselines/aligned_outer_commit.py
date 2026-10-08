"""Production aligned optimizer writes await the prescribed stage before LR commit."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import torch.distributed as dist


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _consensus(worker, value):
    values = [None] * worker.world_size
    dist.all_gather_object(values, value, group=worker.actor_module_fsdp.process_group)
    if any(v.get('error') for v in values):
        raise RuntimeError('Aligned outer collective admission failed: ' + repr(values))
    bodies = [{k:v for k,v in row.items() if k != 'rank'} for row in values]
    if any(body != bodies[0] for body in bodies[1:]):
        raise RuntimeError('Aligned outer collective state differs: ' + repr(values))
    return values


def check_ready(worker, outer, attempt):
    error = None
    try:
        if (type(outer) is not int or type(attempt) is not int or outer < 1 or attempt < 1
                or getattr(worker, '_aligned_pending_outer', None) is not None
                or worker.actor_lr_scheduler.last_epoch != outer - 1):
            raise ValueError('Pending/uncommitted outer or scheduler identity differs')
    except Exception as exc:
        error = repr(exc)
    _consensus(worker, dict(rank=worker.rank, error=error, actual_outer=outer,
                           attempt_id=attempt, scheduler_before=worker.actor_lr_scheduler.last_epoch))


def prepare(worker, events):
    from .fresh200_contract import validate_rank_updates
    outer, attempt = events[0]['actual_outer'], events[0]['attempt_id']
    check_ready(worker, outer, attempt)
    if not validate_rank_updates(events, outer) or not all(e.get('baseline_aligned') is True for e in events):
        raise ValueError('Aligned provisional outer requires twenty complete writes on all ranks')
    worker._aligned_pending_outer = dict(actual_outer=outer, attempt_id=attempt,
        scheduler_before=worker.actor_lr_scheduler.last_epoch, write_ledger_sha256=_digest(events))
    return worker._aligned_pending_outer


def stage_evidence(outer, attempt, metrics, bindings=None):
    count, score = metrics.get('bapo/stage_evaluated_count'), metrics.get('bapo/stage_validation')
    if type(count) is not int or count != 64 or type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError('Aligned outer requires a finite complete64-question prescribed stage')
    result = dict(actual_outer=outer, attempt_id=attempt, evaluated_count=64, score=float(score))
    if bindings is not None:
        if set(bindings) & set(result):
            raise ValueError('Stage bindings cannot replace evaluation identity')
        result.update(bindings)
    result['sha256'] = _digest(result)
    return result


def commit(worker, outer, attempt, stage):
    """Called on all actor ranks. No retry/rollback on partial collective failure."""
    error = None
    pending = getattr(worker, '_aligned_pending_outer', None)
    try:
        if (type(outer) is not int or type(attempt) is not int
                or not worker.config.actor.get('aligned_stage_commit_enabled', False)
                or not pending or pending['actual_outer'] != outer or pending['attempt_id'] != attempt
                or worker.actor_lr_scheduler.last_epoch != pending['scheduler_before']
                or pending['scheduler_before'] != outer - 1
                or stage.get('actual_outer') != outer or stage.get('attempt_id') != attempt
                or type(stage.get('evaluated_count')) is not int or stage.get('evaluated_count') != 64
                or type(stage.get('score')) not in (int,float) or not math.isfinite(stage.get('score', float('nan')))
                or not 0 <= stage['score'] <= 1
                or _digest({k:v for k,v in stage.items() if k != 'sha256'}) != stage.get('sha256')):
            raise ValueError('Absent, duplicate or mismatched aligned stage commit')
    except Exception as exc:
        error = repr(exc)
    _consensus(worker, dict(rank=worker.rank,error=error,actual_outer=outer,attempt_id=attempt,
        pending=pending,stage_validation=stage,scheduler_before=worker.actor_lr_scheduler.last_epoch))
    before = worker.actor_lr_scheduler.last_epoch
    lr_before = worker.actor_lr_scheduler.get_last_lr()
    event = copy.deepcopy(worker._fresh200_last_update_event)
    try:
        worker.actor_lr_scheduler.step()
        lr_after = worker.actor_lr_scheduler.get_last_lr()
        if worker.actor_lr_scheduler.last_epoch != outer or not all(math.isfinite(float(lr)) and lr >= 0 for lr in lr_after):
            raise ValueError('Committed aligned scheduler state invalid')
        event.update(outer_commit_phase='STAGE_COMMITTED', accepted_complete_outer=True,
            stage_validation=stage,scheduler_before=before,scheduler_after=worker.actor_lr_scheduler.last_epoch,
            scheduler_lr_before=lr_before,scheduler_lr_after=lr_after)
        error = None
    except Exception as exc:
        error = repr(exc)
        event.update(outer_commit_phase='COMMIT_FAULT',accepted_complete_outer=False,
                     status='INFRASTRUCTURE_FAULT',fatal=True,update_complete=False,attempt_complete=False,
                     postabort=True,reason='aligned_scheduler_commit_fault',
                     confirmed_previous_optimizer_steps=event['optimizer_steps'],
                     current_state_requires_native_save=True,
                     scheduler_before=before,scheduler_after=worker.actor_lr_scheduler.last_epoch)
    worker._fresh200_last_update_event = event
    try:
        _consensus(worker,dict(rank=worker.rank,error=error,actual_outer=outer,attempt_id=attempt,
            stage_validation=stage,scheduler_before=before,scheduler_after=worker.actor_lr_scheduler.last_epoch,
            scheduler_lr_before=lr_before,scheduler_lr_after=worker.actor_lr_scheduler.get_last_lr()))
    except Exception:
        worker._aligned_pending_outer['phase']='COMMIT_FAULT'
        raise
    from .policy_safety import persist_rank_update_event
    path = Path(worker.config.actor.fresh200_fault_dir) / f'rank_{worker.rank}-outer_{outer}-attempt_{attempt}.json'
    event, faults = persist_rank_update_event(path,event,worker.actor_module_fsdp.process_group)
    if faults is not None:
        event.update(outer_commit_phase='COMMIT_FAULT',accepted_complete_outer=False)
        worker._aligned_pending_outer['phase']='COMMIT_FAULT'
        worker._fresh200_last_update_event = event
        raise RuntimeError('Aligned final rank commit ledger persistence failed')
    worker._fresh200_last_update_event = event
    _consensus(worker,dict(rank=worker.rank,error=error,actual_outer=outer,attempt_id=attempt,
        stage_validation=stage,scheduler_before=before,scheduler_after=worker.actor_lr_scheduler.last_epoch,
        scheduler_lr_before=lr_before,scheduler_lr_after=worker.actor_lr_scheduler.get_last_lr()))
    worker._aligned_pending_outer = None
    print('ACTOR_RANK_OUTER_COMMIT_JSON='+json.dumps(dict(rank=worker.rank,actual_outer=outer,
        attempt_id=attempt,outer_commit_phase='STAGE_COMMITTED',scheduler_before=before,
        scheduler_after=worker.actor_lr_scheduler.last_epoch,stage_proof_sha256=stage['sha256']),sort_keys=True),flush=True)
    return dict(rank=worker.rank,pid=os.getpid(),actual_outer=outer,attempt_id=attempt,
                scheduler_before=before,scheduler_after=worker.actor_lr_scheduler.last_epoch,event=event)


def validate_commits(reports, outer, attempt, stage):
    if len(reports) != 8 or {r.get('rank') for r in reports} != set(range(8)):
        raise ValueError('Aligned scheduler commit requires eight ranks')
    events = []
    for report in reports:
        event=report['event']
        if (report.get('actual_outer') != outer or report.get('attempt_id') != attempt
                or report.get('scheduler_before') != outer-1 or report.get('scheduler_after') != outer
                or event.get('rank') != report['rank'] or event.get('outer_commit_phase') != 'STAGE_COMMITTED'
                or event.get('accepted_complete_outer') is not True or event.get('stage_validation') != stage
                or event.get('scheduler_before') != outer-1 or event.get('scheduler_after') != outer):
            raise ValueError('Aligned stage/scheduler commit identity differs')
        events.append(event)
    from .fresh200_contract import validate_rank_updates
    validate_rank_updates(events,outer)
    return events
