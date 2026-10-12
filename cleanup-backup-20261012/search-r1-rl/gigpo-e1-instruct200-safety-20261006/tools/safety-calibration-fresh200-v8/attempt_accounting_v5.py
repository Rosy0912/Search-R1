"""Actual source receipts bind counters; no20×completed or512×completed inference."""
import copy

from calibration_contract_v1 import require


def actual_attempt_totals(ledger):
    from search_r1.baselines.fresh200 import validate_attempt_ledger
    validate_attempt_ledger(ledger)
    require(ledger['active_attempt'] is None, 'An active attempt is not a completed diagnostic endpoint')
    history = ledger['history']
    require(ledger['completed_outer_updates'] == 2
            and sum(row['opportunities_processed'] for row in history if row['counted_complete']) == 40,
            'Diagnostic requires two countable outers and forty processed counted opportunities')
    return dict(actual_optimizer_steps_per_rank=ledger['actual_adam_total'],
        actual_gradient_micro_per_rank=ledger['gradient_micro_batches'],
        diagnostic_counted_optimizer_opportunities=40,
        actual_processed_optimizer_opportunities=sum(row['opportunities_processed'] for row in history),
        actual_attempts=ledger['attempts'], actual_skipped_attempts=ledger['skipped_attempts'],
        successful_adam_writes_completed_outers=ledger['successful_adam_writes_completed_outers'],
        successful_adam_writes_abandoned_attempts=ledger['successful_adam_writes_abandoned_attempts'],
        actual_skipped_opportunities=ledger['skipped_opportunities'],
        actual_consumed_training_questions=ledger['consumed_training_questions'],
        no_skip_adam_per_outer_per_rank=20, no_skip_gradient_micro_per_outer_per_rank=160,
        counters_are_no_skip_reference=True)


def validate_actual_worker_ledgers(states, ledger):
    from search_r1.baselines.fresh200 import validate_attempt_ledger, validate_worker_attempt_ledger
    validate_attempt_ledger(ledger)
    require(len(states) == 8 and {row['identity']['rank'] for row in states} == set(range(8))
            and len({row['identity']['pid'] for row in states}) == 8, 'Missing actual eight worker states')
    receipts = []
    for state in states:
        rank, pid = state['identity']['rank'], state['identity']['pid']
        worker = state['metadata']['source_runtime_metadata']['fresh200_worker_ledger']
        validate_worker_attempt_ledger(worker)
        expected = [next(row for row in attempt['rank_receipts'] if row['rank'] == rank)
                    for attempt in ledger['history']]
        require(worker['history'] == expected and all(row['pid'] == pid for row in expected)
                and worker['attempts'] == len(expected)
                and worker['completed_outer_updates'] == ledger['completed_outer_updates']
                and worker['actual_adam_total'] == ledger['actual_adam_total']
                and state['identity']['scheduler_step'] == ledger['completed_outer_updates'],
                'Worker actual ledger/PID/scheduler differs from consumed all-rank receipts')
        receipts.append(dict(rank=rank, pid=pid, actual_adam_total=worker['actual_adam_total'],
            completed_outer_updates=worker['completed_outer_updates'], attempts=worker['attempts'],
            history=copy.deepcopy(worker['history'])))
    return sorted(receipts, key=lambda row: row['rank'])
