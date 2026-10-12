"""Audit retained and discarded logical intervals without rewriting any ledger.

This is counter evidence only. It cannot replace rank/process/scheduler,
canonical tensor, probability, probe, evaluation or owned cleanup acceptance.
"""
import math


def account_segment(run_id, updates, attempts, *, retained_through,
                    start_outer=0, start_writes=0, start_attempt_index=0):
    """Inputs are the actual completed driver rows, never synthesized prefixes."""
    for value in (retained_through, start_outer, start_writes, start_attempt_index):
        assert type(value) is int and value >= 0
    assert 0 <= start_outer < retained_through <= 200
    assert start_attempt_index >= start_outer
    assert updates and [row['step'] for row in updates] == list(range(start_outer + 1, updates[-1]['step'] + 1))
    assert retained_through <= updates[-1]['step'] <= 200
    assert all(row['run_id'] == run_id for row in updates + attempts)
    assert [row['attempt_index'] for row in attempts] == list(range(start_attempt_index + 1,
                                                                  start_attempt_index + len(attempts) + 1))
    ids = [row['attempt_id'] for row in attempts]
    assert len(set(ids)) == len(ids)
    completed = start_outer; writes = start_writes; processed = 0
    by_step = {}; retained_attempts = []; discarded_attempts = []
    for row in attempts:
        assert type(row['logical_update']) is int and 1 <= row['logical_update'] <= 200
        assert row['logical_update'] == completed + 1
        assert row['all_ranks_agreed'] is True and row['real_rank_count'] == 4
        before, after, count = (row[key] for key in ('optimizer_writes_before',
            'optimizer_writes_after', 'optimizer_writes_this_attempt'))
        assert all(type(value) is int for value in (before, after, count))
        assert before == writes and after == before + count and 0 <= count <= 20
        assert row['opportunities_processed'] == 20
        statuses = row['opportunity_statuses']
        assert len(statuses) == 20 and all(value in ('written', 'written_with_post_guard', 'skipped_guard')
                                         for value in statuses)
        assert sum(value != 'skipped_guard' for value in statuses) == count
        complete = count > 0
        assert row['complete_outer'] is complete and row['scheduler_advances'] == int(complete)
        assert row['status'] == ('complete' if complete else 'recoverable_guard')
        writes = after; processed += 20
        completed += int(complete)
        assert row['completed_outer'] == completed
        (retained_attempts if row['logical_update'] <= retained_through else discarded_attempts).append(row)
        if complete:
            assert completed not in by_step
            by_step[completed] = row
    scored_steps = {row['step'] for row in updates}
    assert scored_steps.issubset(by_step)
    # A process may exit after the durable all-rank return, but before the
    # scalar writer (validation/model export occur between these two writes).
    # Preserve that returned tail as lost work, without inventing its reward
    # or admitting it into the saved checkpoint prefix. The scalar writer is
    # contiguous, so this can only be the final complete attempt.
    unscored = [row for step, row in by_step.items() if step not in scored_steps]
    if unscored:
        assert len(unscored) == 1 and unscored[0] == attempts[-1]
        assert completed == updates[-1]['step'] + 1
        assert unscored[0]['logical_update'] > retained_through
    else:
        assert completed == updates[-1]['step']
    retained = []; discarded = []
    for update in updates:
        from search_g1.reward_schedule import ACTOR_PROBE_SCHEDULE
        assert update['reward_schedule'] == ACTOR_PROBE_SCHEDULE
        assert type(update['reward_logical_update']) is int and update['reward_logical_update'] == update['step']
        assert update['reward_mode'] == ('search_g1' if update['step'] <= 100 else 'binary_em')
        row = by_step[update['step']]
        assert all(update[key] == row[key] for key in ('attempt_id', 'attempt_index',
            'optimizer_writes_before', 'optimizer_writes_after', 'opportunities_processed'))
        assert update['actual_optimizer_writes'] == row['optimizer_writes_this_attempt']
        reward = update['metrics']['critic/rewards/mean']
        assert type(reward) in (float, int) and math.isfinite(reward)
        (retained if update['step'] <= retained_through else discarded).append(update)
    checkpoint_row = retained[-1]
    return dict(role='actor_probe_counter_segment', evidence_scope='counters_only', run_id=run_id,
        start_outer=start_outer, retained_through=retained_through,
        start_writes=start_writes, start_attempt_index=start_attempt_index,
        checkpoint_writes=checkpoint_row['optimizer_writes_after'],
        checkpoint_attempt_index=checkpoint_row['attempt_index'],
        observed_completed_through=completed, observed_writes=writes,
        observed_processed_opportunities=processed,
        retained_updates=retained, discarded_updates=discarded,
        discarded_unscored_attempts=unscored,
        retained_attempts=retained_attempts, discarded_attempts=discarded_attempts,
        discarded_successful_adam=writes-checkpoint_row['optimizer_writes_after'],
        scope='durable fully returned driver attempts only; incomplete rank-local tail must be reported separately')


def assemble_counter_chain(segments, *, final_outer):
    assert segments and type(final_outer) is int and 1 <= final_outer <= 200
    seen_runs = set(); owners = {}; previous = None; discarded = []; phases = {'g1': [], 'binary_em': []}
    for segment in segments:
        assert segment['role'] == 'actor_probe_counter_segment' and segment['evidence_scope'] == 'counters_only'
        derived = account_segment(segment['run_id'],
            segment['retained_updates'] + segment['discarded_updates'],
            segment['retained_attempts'] + segment['discarded_attempts'],
            retained_through=segment['retained_through'], start_outer=segment['start_outer'],
            start_writes=segment['start_writes'], start_attempt_index=segment['start_attempt_index'])
        assert derived == segment, 'Segment summary differs from its actual supplied rows'
        assert segment['run_id'] not in seen_runs, 'Recovery chain cycle or duplicated namespace'
        seen_runs.add(segment['run_id'])
        if previous is None:
            assert (segment['start_outer'], segment['start_writes'], segment['start_attempt_index']) == (0,0,0)
        else:
            assert segment['start_outer'] == previous['retained_through']
            assert segment['start_writes'] == previous['checkpoint_writes'], 'Recovery inherited discarded Adam updates'
            assert segment['start_attempt_index'] == previous['checkpoint_attempt_index']
        for row in segment['retained_updates']:
            assert row['step'] not in owners
            owners[row['step']] = segment['run_id']
            phases['g1' if row['step'] <= 100 else 'binary_em'].append(row['metrics']['critic/rewards/mean'])
        discarded.append(dict(run_id=segment['run_id'],
            complete_outers=[row['step'] for row in segment['discarded_updates']] +
                [row['logical_update'] for row in segment['discarded_unscored_attempts']],
            unscored_complete_outers=[row['logical_update'] for row in segment['discarded_unscored_attempts']],
            successful_adam=segment['discarded_successful_adam'],
            attempt_indices=[row['attempt_index'] for row in segment['discarded_attempts']],
            semantics='Preserved original evidence; excluded from the retained logical chain, no rollback claimed'))
        previous = segment
    assert set(owners) == set(range(1, final_outer + 1)) and previous['retained_through'] == final_outer
    rewards = {mode: dict(count=len(values), mean=sum(values)/len(values) if values else None,
                         minimum=min(values) if values else None, maximum=max(values) if values else None)
               for mode, values in phases.items()}
    return dict(role='actor_probe_counter_chain', evidence_scope='counters_only',
        complete_outers=final_outer, successful_adam=previous['checkpoint_writes'],
        logical_owners=owners, discarded_by_namespace=discarded, reward_phases=rewards,
        full_training_or_final_acceptance=False,
        pending_evidence='Actual rank/process/scheduler/parameter/RNG/probe and checkpoint binding, partial tails, '
                         'final ordered51713 EM/F1/source counts and only-owned cleanup remain mandatory')
