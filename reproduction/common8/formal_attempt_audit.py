"""Audit every real completed or fully skipped formal attempt, without fake writes.

This reader checks the owned driver, scheduler and worker receipt prefix. It
does not execute a model or authenticate physical GPU execution from JSON.
"""
import copy
import math
import os
import re
from pathlib import Path

from reproduction.common8.formal_contract import FORMAL_SAFEGUARDS, consumed_lr
from reproduction.common8.policy_calibration_plan import _strict_json, document_sha256
from reproduction.common8.topology import (
    actor_local_mini, actor_local_micro, formal_monitor_geometry, local_trajectories, monitor_scope, monitor_trajectories_per_rank, ranks, world_size,
)
from reproduction.common8.safety_calibration_evidence import (
    EvidenceError, GLOBAL_FIELDS, GRADIENT_CLIP_FLOAT32_TOLERANCE, GRADIENT_CLIP_LIMIT,
    OBJECTIVE_WEIGHT_DEFINITION, QUANTILE_METHOD,
    _adjacent, _enforcement, _finite, _hash, _identity, _integer, _require,
    _row, _same_arithmetic, _statistics,
)


NUMERICAL_GUARD_STAGES = frozenset((
    'before_gradient_clipping', 'before_optimizer_step', 'before_dual_clip_exp',
    'before_backward', 'before_fp64_gradient_clipping', 'before_fp64_clip_write',
    'after_fp64_gradient_clipping', 'measured_after_gradient_clipping',
))
INCOMPLETE_REASONS = frozenset((
    'nonfinite_global_totals', 'nonfinite_or_negative_effective_input',
    'delta_above_kl_log_bound', 'empty_effective_samples',
))


def _owned_rows(base, relative):
    path = base / relative
    _require(path.is_file() and not path.is_symlink() and path.resolve() == path
             and path.stat().st_uid == os.getuid(), 'Receipt is not an owned canonical file: ' + relative)
    return [_strict_json(line) for line in path.read_bytes().splitlines()]


def _attempt_identity(row):
    context = row.get('context') or row
    index = _integer(context['attempt_index'], 'attempt_index', minimum=1)
    attempt = context['attempt_id']
    _require(isinstance(attempt, str) and re.fullmatch('[0-9a-f]{32}', attempt) is not None,
             'Malformed actual attempt UUID')
    return attempt, index


def _monitor_statistics(row):
    _require(type(row['allowed']) is bool and type(row['kl_estimate_complete']) is bool,
             'Malformed actual monitor decision')
    _require(isinstance(row['reasons'], list) and all(isinstance(x, str) and x for x in row['reasons']),
             'Malformed actual monitor reasons')
    if row['kl_estimate_complete']:
        statistics = _statistics(dict(row, allowed=True, reasons=[]))
        reasons = []
        if statistics['sampled_old_policy_kl'] > FORMAL_SAFEGUARDS['target_old_policy_kl']:
            reasons.append('old_policy_kl')
        if statistics['negative_tail_weight_fraction'] > FORMAL_SAFEGUARDS['negative_tail_limit']:
            reasons.append('negative_advantage_tail')
        _require(row['reasons'] == reasons and row['allowed'] == (not reasons),
                 'Actual monitor differs from the frozen prospective brake decision')
        statistics.update(allowed=row['allowed'], reasons=copy.deepcopy(reasons))
        return statistics
    _require(row['allowed'] is False and row['sampled_old_policy_kl'] is None and row['reasons']
             and any(reason in INCOMPLETE_REASONS for reason in row['reasons']),
             'Incomplete numerical estimate cannot claim true KL or an unexplained skip')
    for name in ('effective_token_count', 'negative_token_count', 'negative_tail_count', 'invalid_count', 'overflow_count'):
        _integer(row[name], name)
    count = row['effective_token_count']
    _require(0 <= row['negative_tail_count'] <= row['negative_token_count'] <= count
             and row['invalid_count'] <= count and row['overflow_count'] <= count,
             'Inconsistent incomplete numerical counts')
    nonfinite_totals = 'nonfinite_global_totals' in row['reasons']
    for name in ('effective_weight_mass', 'weighted_k3_numerator', 'negative_effective_weight_mass',
                 'tail_effective_weight_mass'):
        if nonfinite_totals:
            _require(row[name] is None, 'Nonfinite global total must remain an explicit unavailable diagnostic')
        else:
            _finite(row[name], name)
    for name in ('negative_tail_count_fraction', 'negative_tail_weight_fraction'):
        _finite(row[name], name)
    _same_arithmetic(row['negative_tail_count_fraction'],
        row['negative_tail_count'] / row['negative_token_count'] if row['negative_token_count'] else 0.,
        'incomplete negative tail count fraction')
    if nonfinite_totals:
        _require(row['negative_tail_weight_fraction'] == 0., 'Unavailable tail mass must remain a diagnostic fraction')
    else:
        mass, negative_mass, tail_mass = (row[name] for name in
            ('effective_weight_mass', 'negative_effective_weight_mass', 'tail_effective_weight_mass'))
        _require((negative_mass <= mass or math.isclose(negative_mass, mass, rel_tol=1e-12, abs_tol=1e-12)) and
                 (tail_mass <= negative_mass or math.isclose(tail_mass, negative_mass, rel_tol=1e-12, abs_tol=1e-12)),
                 'Inconsistent incomplete global masses')
        _same_arithmetic(row['negative_tail_weight_fraction'], tail_mass / negative_mass if negative_mass else 0.,
                         'incomplete negative tail weight fraction')
    saturated = row['dual_clip_saturated_objective_weight_fraction']
    if nonfinite_totals or row['effective_weight_mass'] == 0:
        _require(saturated is None, 'Unavailable saturation cannot claim a finite complete ratio')
    else:
        _finite(saturated, 'dual_clip_saturated_objective_weight_fraction')
        _same_arithmetic(saturated, row['tail_effective_weight_mass'] / row['effective_weight_mass'],
                         'incomplete saturated objective fraction')
    reasons = []
    if nonfinite_totals: reasons.append('nonfinite_global_totals')
    if row['invalid_count']: reasons.append('nonfinite_or_negative_effective_input')
    if row['overflow_count']: reasons.append('delta_above_kl_log_bound')
    if count <= 0 or (not nonfinite_totals and row['effective_weight_mass'] <= 0): reasons.append('empty_effective_samples')
    if row['negative_tail_weight_fraction'] > FORMAL_SAFEGUARDS['negative_tail_limit']:
        reasons.append('negative_advantage_tail')
    _require(row['reasons'] == reasons, 'Incomplete numerical guard differs from its actual diagnostic counts/ratios')
    _require(row['quantile_method'] == QUANTILE_METHOD, 'Unexpected incomplete quantile method')
    for name in ('negative_log_ratio_p95', 'negative_log_ratio_p99', 'negative_log_ratio_max'):
        if row[name] is not None:
            _finite(row[name], name, nonnegative=False)
    quantiles = [row[name] for name in ('negative_log_ratio_p95', 'negative_log_ratio_p99', 'negative_log_ratio_max')]
    _require((row['negative_token_count'] == 0 and quantiles == [None, None, None]) or
             (row['negative_token_count'] > 0 and all(value is not None for value in quantiles)
              and quantiles == sorted(quantiles)), 'Incomplete numerical quantiles differ from actual negative samples')
    return {name: copy.deepcopy(row[name]) for name in GLOBAL_FIELDS}


def _audit_monitor_rows(rows_by_rank, binding, attempt, rejection_before, *, run):
    """Require40 true pre/post monitors and20 dispositions for this real UUID."""
    try:
        geometry = formal_monitor_geometry(run)
    except (AssertionError, KeyError, TypeError) as failure:
        raise EvidenceError('Invalid bound formal monitor geometry: ' + str(failure)) from failure
    observations = {}; dispositions = {}; identities = {}; final_rejections = []
    outer = attempt['logical_update']; start = attempt['optimizer_writes_before']
    successes = attempt['optimizer_writes_this_attempt']; complete = attempt['complete_outer']
    for rank in ranks(run):
        rows = rows_by_rank[rank]; cursor = 0; counter = start; identity = None; frozen = None
        rejections = rejection_before
        for mini in range(20):
            _require(cursor < len(rows), 'Missing actual attempt pre-monitor')
            before = rows[cursor]; cursor += 1; write = None
            if cursor < len(rows) and rows[cursor]['kind'] == 'OPTIMIZER_WRITE':
                write = rows[cursor]; cursor += 1
            _require(cursor + 1 < len(rows), 'Missing actual attempt post-monitor or opportunity')
            after, decision = rows[cursor:cursor + 2]; cursor += 2
            succeeded = write is not None
            context = dict(logical_update=outer, minibatch_index=mini, prior_optimizer_writes=counter,
                attempt_id=attempt['attempt_id'], attempt_index=attempt['attempt_index'],
                behavior_probability_gate_sha256=attempt['behavior_probability_gate_sha256'])
            for event in [before, after, decision] + ([] if write is None else [write]):
                _require(event['context'] == context, 'Actual attempt opportunity context or gate differs')
            _row(decision, rank, binding, 'OPPORTUNITY')
            for name in ('opportunity_index', 'optimizer_writes_before', 'optimizer_writes_after'):
                _integer(decision[name], name)
            _require(decision['processed'] is True and decision['opportunity_index'] == mini,
                     'Actual attempt opportunities are missing or reordered')
            _require(type(decision['optimizer_write_succeeded']) is bool and
                     decision['optimizer_write_succeeded'] == succeeded,
                     'Actual opportunity success differs from the real write receipt')
            _require(decision['optimizer_writes_before'] == counter and
                     decision['optimizer_writes_after'] == counter + int(succeeded),
                     'Actual opportunity write counters differ')
            guards = decision['guard_events']
            _require(isinstance(guards, list), 'Missing actual opportunity guard events')
            for event in guards:
                _require(isinstance(event, dict) and set(event) == {'stage', 'reasons'} and
                         isinstance(event['reasons'], list) and event['reasons'] and
                         all(isinstance(x, str) and x for x in event['reasons']), 'Malformed actual guard event')
                _require(event['stage'] in NUMERICAL_GUARD_STAGES | {'before_minibatch', 'after_minibatch'},
                         'Fatal or unknown stage cannot claim a recoverable skip')
                if event['stage'] in NUMERICAL_GUARD_STAGES:
                    _require(event['reasons'] == ['nonfinite_or_empty_effective_numerical_input'],
                             'Unexpected actual numerical guard reason')
                _require(not succeeded or event['stage'] == 'after_minibatch',
                         'Rejected pre-write opportunity cannot claim an Adam write')
            expected_status = ('written_with_post_guard' if guards else 'written') if succeeded else 'skipped_guard'
            _require(decision['status'] == expected_status and (succeeded or bool(guards)),
                     'Actual skip/write disposition lacks its guard')
            _require(decision['guard_stage'] == (guards[0]['stage'] if guards else None) and
                     decision['guard_reasons'] == (guards[0]['reasons'] if guards else []),
                     'Actual primary guard differs from guard events')
            _require(decision['status'] == attempt['opportunity_statuses'][mini],
                     'Driver return differs from the real opportunity disposition')
            local_decision = {key: copy.deepcopy(value) for key, value in decision.items() if key != 'rank'}
            if mini in dispositions:
                _require(dispositions[mini] == local_decision, 'Actual ranks disagree on opportunity disposition')
            else:
                dispositions[mini] = local_decision
            for row, stage, writes in ((before, 'before_minibatch', counter),
                                      (after, 'after_minibatch', counter + int(succeeded))):
                _row(row, rank, binding, 'POLICY_MONITOR')
                _integer(row['optimizer_writes'], 'monitor optimizer_writes')
                _require(row['stage'] == stage and row['optimizer_writes'] == writes,
                         'Actual monitor/write order differs')
                _require(row['calibration_sha256'] == binding['calibration_sha256'] and
                         row['execution_role'] == 'production' and row['enforcement'] == _enforcement('production'),
                         'Actual monitor prospective plan or enforced role differs')
                _require(row['objective_weight_definition'] == OBJECTIVE_WEIGHT_DEFINITION and
                         row['monitoring_scope'] == monitor_scope(run), 'Actual monitor weight/scope differs')
                _identity(row['monitor_identity'], monitor_count=monitor_trajectories_per_rank(run))
                if identity is None:
                    identity = copy.deepcopy(row['monitor_identity']); identities[rank] = identity
                _require(identity == row['monitor_identity'], 'Actual frozen monitor identity changed')
                statistics = _monitor_statistics(row)
                rejections += int(not row['allowed'])
                _require(_integer(row['rejection_count'], 'rejection_count') == rejections,
                         'Actual cumulative policy rejection counter differs')
                if row['kl_estimate_complete']:
                    fixed = tuple(statistics[name] for name in ('effective_weight_mass', 'negative_effective_weight_mass',
                        'effective_token_count', 'negative_token_count'))
                    if frozen is None:
                        frozen = fixed
                    _require(fixed == frozen, 'Actual frozen monitor weights changed')
                if not row['allowed']:
                    _require(any(event['stage'] == stage and event['reasons'] == row['reasons'] for event in guards),
                             'Rejected actual policy monitor lacks its real disposition')
                    _require(stage != 'before_minibatch' or not succeeded,
                             'Rejected actual pre-monitor cannot enter Adam')
                key = (mini, stage)
                if key in observations:
                    _require(all(observations[key][name] == statistics[name] for name in GLOBAL_FIELDS),
                             'Actual rank replicas disagree on global attempt statistics')
                else:
                    observations[key] = statistics
            for event in guards:
                if event['stage'] in ('before_minibatch', 'after_minibatch'):
                    monitor_row = before if event['stage'] == 'before_minibatch' else after
                    _require(monitor_row['allowed'] is False and monitor_row['reasons'] == event['reasons'],
                             'Policy guard does not match its actual rejected monitor')
            if not succeeded:
                _adjacent(observations[(mini, 'before_minibatch')], observations[(mini, 'after_minibatch')])
            if write is not None:
                _row(write, rank, binding, 'OPTIMIZER_WRITE')
                _integer(write['write_count'], 'actual write_count', minimum=1)
                _require(write['write_count'] == counter + 1 and write['learning_rates'] and
                         all(type(rate) in (int, float) and rate == consumed_lr(outer) for rate in write['learning_rates']),
                         'Actual retained Adam write or consumed LR differs')
                _finite(write['gradient_norm_before'], 'gradient_norm_before')
                _require(_finite(write['gradient_norm_after'], 'gradient_norm_after') <=
                         GRADIENT_CLIP_LIMIT + GRADIENT_CLIP_FLOAT32_TOLERANCE, 'Actual global L2 clip differs')
                counter += 1
        _require(cursor == len(rows) - 1, 'Missing or extra actual attempt monitor/write/terminal receipts')
        terminal = rows[cursor]
        _row(terminal, rank, binding, 'OUTER_BUDGET' if complete else 'ABORTED_ATTEMPT')
        for key, value in dict(logical_update=outer, attempt_id=attempt['attempt_id'], attempt_index=attempt['attempt_index'],
            opportunities_processed=20, optimizer_writes=successes, cumulative_writes=counter).items():
            if type(value) is int:
                _integer(terminal[key], key)
            _require(terminal[key] == value, 'Actual attempt terminal counter differs: ' + key)
        if complete:
            for key, value in dict(local_trajectories=local_trajectories(run), local_mini=actor_local_mini(run), local_micro=actor_local_micro(run),
                optimizer_writes_before=start, optimizer_writes_after=counter).items():
                _require(terminal[key] == value, 'Actual completed budget differs: ' + key)
        else:
            _require(terminal['context'] == context and terminal['writes_before'] == start and
                     terminal['behavior_probability_gate_sha256'] == context['behavior_probability_gate_sha256'] and
                     terminal['guard_stage'] == attempt['guard_stage'] and terminal['guard_reasons'] == attempt['guard_reasons'],
                     'Actual skipped terminal identity or guard differs')
            _require(terminal['recoverable_guard'] is True and terminal['complete_outer'] is False and
                     type(terminal['scheduler_advances']) is int and terminal['scheduler_advances'] == 0 and
                     terminal['retained_partial_adam_writes'] is True and terminal['no_automatic_rollback'] is True,
                     'Actual skipped terminal cannot advance or claim rollback')
        _require(counter == attempt['optimizer_writes_after'], 'Actual retained attempt write total differs')
        final_rejections.append(rejections)
    _require(len(set(final_rejections)) == 1, 'Actual ranks disagree on policy rejection totals')
    token_count = sum(sum(identity['actor_token_lengths']) for identity in identities.values())
    for (mini, stage), observation in observations.items():
        _require(observation['effective_token_count'] == token_count, 'Actual global32 token count differs')
        if observation['kl_estimate_complete']:
            _same_arithmetic(observation['effective_weight_mass'], geometry['expected_complete_effective_weight_mass'],
                             'actual global masked-mean weight mass')
        if mini and stage == 'before_minibatch':
            _adjacent(observations[(mini - 1, 'after_minibatch')], observation)
    return dict(global_monitor_count=len(observations), rejection_count=final_rejections[0],
                guard_triggered_monitor_count=sum(not row['allowed'] for row in observations.values()),
                monitor_weight_binding=geometry)


def verify_attempt_history(base, run, complete_records, scheduler_receipts, *, last, plan_sha256):
    """Verify the sealed attempt prefix ending at the requested COMPLETE outer."""
    try:
        return _verify_attempt_history(base, run, complete_records, scheduler_receipts, last=last, plan_sha256=plan_sha256)
    except EvidenceError:
        raise
    except (KeyError, TypeError, OverflowError, ValueError, OSError) as failure:
        raise EvidenceError('Malformed actual formal attempt history: ' + str(failure)) from failure


def _verify_attempt_history(base, run, complete_records, scheduler_receipts, *, last, plan_sha256):
    from reproduction.common8.behavior_runtime import read_gate_receipt
    base = Path(base).resolve(); _integer(last, 'last', minimum=1); _hash(plan_sha256, 'prospective plan')
    _require(last <= 200 and run['policy_safety_mode'] == 'inline_first_five', 'Invalid formal attempt scope')
    from search_g1.full_state_restore import verified_runtime_prefix
    prefix=verified_runtime_prefix(base,run) if run.get('full_state_restore') else dict(completed_outer=0,writes=0,attempt_index=0,rejections=0,aborted=0)
    _require([row['step'] for row in complete_records] == list(range(prefix['completed_outer']+1, last + 1)), 'Incomplete actual completed selection')
    cutoff = _integer(complete_records[-1]['attempt_index'], 'cutoff attempt', minimum=1)
    _require(set(scheduler_receipts) == {str(rank) for rank in ranks(run)}, 'Need every actual scheduler receipt')
    all_attempts = _owned_rows(base, 'logs/actual-attempts.jsonl')
    attempts = [row for row in all_attempts if _attempt_identity(row)[1] <= cutoff]
    _require(last < 200 or len(attempts) == len(all_attempts), 'Actual driver contains attempts after final COMPLETE200')
    _require([row['attempt_index'] for row in attempts] == list(range(prefix['attempt_index']+1, cutoff + 1)),
             'Actual driver attempt prefix is omitted, duplicated or reordered')
    _require(len({row['attempt_id'] for row in attempts}) == cutoff-prefix['attempt_index'], 'Actual driver attempt UUID reused')
    rank_workers = {}; rank_schedulers = {}; selected_hashes = {}; scheduler_hashes = {}
    for rank in ranks(run):
        all_scheduler = _owned_rows(base, f'logs/scheduler-rank-{rank}.jsonl')
        scheduler = [row for row in all_scheduler if _attempt_identity(row)[1] <= cutoff]
        _require(last < 200 or len(scheduler) == len(all_scheduler), 'Actual scheduler continues after final COMPLETE200')
        _require(scheduler == scheduler_receipts[str(rank)]['selected_rows'],
                 'Owned scheduler prefix differs from independently checked receipts')
        all_workers = _owned_rows(base, f'logs/optimizer-rank-{rank}.jsonl')
        workers = [row for row in all_workers if _attempt_identity(row)[1] <= cutoff]
        _require(last < 200 or len(workers) == len(all_workers), 'Actual worker continues after final COMPLETE200')
        rank_workers[rank] = workers; rank_schedulers[rank] = scheduler
        selected_hashes[str(rank)] = document_sha256(workers)
        scheduler_hashes[str(rank)] = document_sha256(scheduler)
    binding = dict(run_id=run['run_id'], calibration_sha256=plan_sha256)
    completed=prefix['completed_outer'];writes=prefix['writes'];aborted=prefix['aborted'];rejections=prefix['rejections'];summaries=[]
    scheduler_cursors = {rank: 0 for rank in ranks(run)}; worker_cursors = {rank: 0 for rank in ranks(run)}
    job = rank_schedulers[0][0]['slurm_job_id']
    for ordinal, raw_attempt in enumerate(attempts, prefix['attempt_index']+1):
        attempt = copy.deepcopy(raw_attempt); outer = completed + 1
        for name in ('logical_update', 'optimizer_writes_before', 'optimizer_writes_after',
                     'completed_outer', 'cumulative_aborted_attempts'):
            _integer(attempt[name], name)
        _require(attempt['run_id'] == run['run_id'] and attempt['slurm_job_id'] == job and
                 attempt['logical_update'] == outer and attempt['all_ranks_agreed'] is True and
                 type(attempt['real_rank_count']) is int and attempt['real_rank_count'] == world_size(run),
                 'Actual driver attempt run/rank/outer identity differs')
        if run.get('execution_backend', 'slurm') == 'ssh_local':
            _require(job is None and attempt['execution_backend'] == 'ssh_local' and
                     attempt['execution_id'] == run['execution_id'], 'Actual driver local execution identity differs')
        count = _integer(attempt['optimizer_writes_this_attempt'], 'actual attempt writes')
        _require(count <= 20 and attempt['optimizer_writes_before'] == writes and
                 attempt['optimizer_writes_after'] == writes + count and
                 type(attempt['opportunities_processed']) is int and attempt['opportunities_processed'] == 20,
                 'Actual driver normal return is partial or has wrong retained write counters')
        complete = count > 0
        _require(type(attempt['complete_outer']) is bool and attempt['complete_outer'] is complete and
                 attempt['status'] == ('complete' if complete else 'recoverable_guard') and
                 type(attempt['scheduler_advances']) is int and attempt['scheduler_advances'] == int(complete) and
                 attempt['completed_outer'] == completed + int(complete), 'Actual driver completion or warmup differs')
        statuses = attempt['opportunity_statuses']
        _require(isinstance(statuses, list) and len(statuses) == 20 and
                 all(status in ('written', 'written_with_post_guard', 'skipped_guard') for status in statuses) and
                 sum(status != 'skipped_guard' for status in statuses) == count, 'Actual driver opportunity returns differ')
        _require(attempt['guard_stage'] == (None if complete else 'all_update_opportunities_skipped') and
                 attempt['guard_reasons'] == ([] if complete else ['no_successful_adam_write']) and
                 attempt['retained_partial_adam_writes'] is (not complete) and attempt['no_automatic_rollback'] is True,
                 'Actual driver skip/fatal/rollback semantics differ')
        aborted += int(not complete)
        _require(attempt['cumulative_aborted_attempts'] == aborted, 'Actual driver retry counter differs')
        # A paired restore supplies only the current execution's suffix rows;
        # `completed` includes the authenticated parent prefix. Index relative
        # to that boundary rather than skipping real suffix receipts.
        record = complete_records[completed - prefix['completed_outer']] if complete else None
        if record is not None:
            _require(record['run_id'] == run['run_id'] and record['attempt_id'] == attempt['attempt_id'] and
                     record['attempt_index'] == ordinal and record['optimizer_writes_before'] == writes and
                     record['optimizer_writes_after'] == writes + count and record['opportunities_processed'] == 20,
                     'Actual completed update differs from the real driver attempt')
            _require(type(record['actual_optimizer_writes']) is int and record['actual_optimizer_writes'] == count and
                     record['metrics']['actor/optimizer_steps_this_update'] == count and
                     record['metrics']['actor/update_opportunities_this_attempt'] == 20 and
                     record['metrics']['actor/skipped_update_opportunities'] == 20 - count,
                     'Actual scalar update ledger differs from retained Adam writes/opportunities')
            gate = read_gate_receipt(base, run, outer, update_record=record)
        else:
            gate = read_gate_receipt(base, run, outer, attempt_id=attempt['attempt_id'], attempt_index=ordinal)
        attempt['behavior_probability_gate_sha256'] = gate['proof_sha256']
        rows_by_rank = {}
        expected_kinds = ['BEGIN', 'COMMIT_PREPARED', 'COMPLETE'] if complete else ['BEGIN', 'ABORTED_ATTEMPT']
        for rank in ranks(run):
            cursor = scheduler_cursors[rank]; rows = rank_schedulers[rank]
            selected = rows[cursor:cursor + len(expected_kinds)]
            _require([row['kind'] for row in selected] == expected_kinds, 'Actual scheduler attempt is partial or reordered')
            for row in selected:
                _require(_attempt_identity(row) == (attempt['attempt_id'], ordinal) and row['run_id'] == run['run_id'] and
                         row['rank'] == rank and row['slurm_job_id'] == job and row['logical_update'] == outer and
                         row['batch_gate_sha256'] == gate['proof_sha256'], 'Actual scheduler attempt identity/gate differs')
                if run.get('execution_backend', 'slurm') == 'ssh_local':
                    _require(row['execution_backend'] == 'ssh_local' and row['execution_id'] == run['execution_id'],
                             'Actual scheduler local execution identity differs')
                end = row['kind'] != 'BEGIN'; advanced = row['kind'] == 'COMPLETE'
                _require(row['writes_before'] == writes and row['writes_after'] == writes + (count if end else 0) and
                         row['actual_optimizer_writes'] == (count if end else 0) and
                         row['opportunities_processed'] == (20 if end else 0) and row['optimizer_opportunities'] == 20 and
                         row['completed_outer_before'] == completed and row['completed_outer_after'] == completed + int(advanced) and
                         row['scheduler_advances'] == int(advanced), 'Actual scheduler differs from the real driver attempt')
            scheduler_cursors[rank] += len(selected)
            cursor = worker_cursors[rank]; rows = rank_workers[rank]; selected = []
            while cursor < len(rows) and _attempt_identity(rows[cursor]) == (attempt['attempt_id'], ordinal):
                selected.append(rows[cursor]); cursor += 1
            worker_cursors[rank] = cursor; rows_by_rank[rank] = selected
        audit = _audit_monitor_rows(rows_by_rank, binding, attempt, rejections, run=run)
        _require(audit['global_monitor_count'] == 40, 'Need40 actual global monitors for every attempt')
        rejections = audit['rejection_count']
        summaries.append(dict(attempt_id=attempt['attempt_id'], attempt_index=ordinal, logical_update=outer,
            complete_outer=complete, actual_optimizer_writes=count, global_monitor_count=40,
            guard_triggered_monitor_count=audit['guard_triggered_monitor_count']))
        writes += count; completed += int(complete)
    _require(completed == last and attempts[-1]['complete_outer'] is True, 'Actual attempt prefix does not end at the requested COMPLETE')
    for rank in ranks(run):
        _require(scheduler_cursors[rank] == len(rank_schedulers[rank]) and worker_cursors[rank] == len(rank_workers[rank]),
                 'Extra or unbound actual scheduler/worker attempt receipts')
    selection = dict(plan_sha256=plan_sha256, driver_attempts=attempts,
                     selected_rank_rows_sha256=selected_hashes, selected_scheduler_rows_sha256=scheduler_hashes)
    return dict(schema_version=1, role='actual_formal_all_attempt_monitor_audit', run_id=run['run_id'],
        structural_accepted=True, completed_outer_count=completed, cutoff_attempt_index=cutoff,
        actual_attempt_count=len(attempts), source_completed_outer=prefix['completed_outer'],
        source_actual_optimizer_writes_per_rank=prefix['writes'], fully_skipped_attempt_count=aborted-prefix['aborted'],
        processed_opportunities_per_rank=len(attempts) * 20, actual_optimizer_writes_per_rank=writes,
        completed_outer_global_monitor_count=(completed-prefix['completed_outer']) * 40, retry_global_monitor_count=(aborted-prefix['aborted']) * 40,
        unique_global_monitor_count=len(attempts) * 40, cumulative_policy_rejections=rejections,
        guard_triggered_monitor_count=sum(row['guard_triggered_monitor_count'] for row in summaries),
        attempts=summaries, selected_driver_attempts_sha256=document_sha256(attempts),
        selected_rank_rows_sha256=selected_hashes, selected_scheduler_rows_sha256=scheduler_hashes,
        selected_evidence_sha256=document_sha256(selection), production_admission=False,
        scope='sealed actual attempt prefix; global policy-monitor rank replicas count once; '
              'actor loss keeps the original first-rank collector semantics; structural audit is not accuracy success')
