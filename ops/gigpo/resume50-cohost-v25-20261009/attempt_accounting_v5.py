"""Actual source receipts bind counters; no20×completed or512×completed inference."""
import copy

from calibration_contract_v1 import require


def actual_attempt_totals(ledger, expected_completed=None):
    from search_r1.baselines.fresh200 import validate_attempt_ledger
    validate_attempt_ledger(ledger)
    require(ledger['active_attempt'] is None, 'An active attempt is not a completed diagnostic endpoint')
    history = ledger['history']
    completed = ledger['completed_outer_updates']
    require(type(completed) is int and 0 <= completed <= 200
            and (expected_completed is None or type(expected_completed) is int and completed == expected_completed)
            and sum(row['opportunities_processed'] for row in history if row['counted_complete']) == 20 * completed,
            'Counted complete opportunities differ from actual ledger; writes remain separate')
    return dict(actual_optimizer_steps_per_rank=ledger['actual_adam_total'],
        actual_gradient_micro_per_rank=ledger['gradient_micro_batches'],
        diagnostic_counted_optimizer_opportunities=20 * completed,
        actual_processed_optimizer_opportunities=sum(row['opportunities_processed'] for row in history),
        actual_attempts=ledger['attempts'], actual_skipped_attempts=ledger['skipped_attempts'],
        successful_adam_writes_completed_outers=ledger['successful_adam_writes_completed_outers'],
        successful_adam_writes_abandoned_attempts=ledger['successful_adam_writes_abandoned_attempts'],
        actual_skipped_opportunities=ledger['skipped_opportunities'],
        actual_consumed_training_questions=ledger['consumed_training_questions'],
        no_skip_adam_per_outer_per_rank=20, no_skip_gradient_micro_per_outer_per_rank=160,
        counters_are_no_skip_reference=True)


def validate_actual_worker_ledgers(states, ledger, historical_prefix=None):
    from search_r1.baselines.fresh200 import validate_attempt_ledger, validate_worker_attempt_ledger
    validate_attempt_ledger(ledger)
    require(len(states) == 8 and {row['identity']['rank'] for row in states} == set(range(8))
            and len({row['identity']['pid'] for row in states}) == 8, 'Missing actual eight worker states')
    inherited_attempts = 0
    parent_pids = {}
    if historical_prefix is not None:
        from resume50_v21 import binding, manifest, MANIFEST_SHA, PARENT_RUN
        binding(); accepted = manifest()
        require(historical_prefix['manifest_sha256'] == MANIFEST_SHA
                and historical_prefix['run'] == str(PARENT_RUN)
                and historical_prefix['retained_outer'] == 50, 'Unbound inherited ledger prefix')
        parent_pids = {r['pause_identity']['rank']: r['pause_identity']['pid']
                       for r in accepted['files'] if 'pause_identity' in r}
        require(set(parent_pids) == set(range(8)), 'Missing accepted parent worker identities')
        inherited_attempts = 50
        require(len(ledger['history']) >= inherited_attempts
                and all(a['counted_complete'] for a in ledger['history'][:inherited_attempts]),
                'Inherited accepted attempt prefix differs')
    receipts = []
    for state in states:
        rank, pid = state['identity']['rank'], state['identity']['pid']
        worker = state['metadata']['source_runtime_metadata']['fresh200_worker_ledger']
        validate_worker_attempt_ledger(worker)
        expected = [next(row for row in attempt['rank_receipts'] if row['rank'] == rank)
                    for attempt in ledger['history']]
        require(worker['history'] == expected and all(row['pid'] == (parent_pids[rank] if index < inherited_attempts else pid)
                    for index, row in enumerate(expected))
                and worker['attempts'] == len(expected)
                and worker['completed_outer_updates'] == ledger['completed_outer_updates']
                and worker['actual_adam_total'] == ledger['actual_adam_total']
                and state['identity']['scheduler_step'] == ledger['completed_outer_updates'],
                'Worker actual ledger/PID/scheduler differs from consumed all-rank receipts')
        receipts.append(dict(rank=rank, pid=pid, actual_adam_total=worker['actual_adam_total'],
            completed_outer_updates=worker['completed_outer_updates'], attempts=worker['attempts'],
            history=copy.deepcopy(worker['history'])))
    return sorted(receipts, key=lambda row: row['rank'])
