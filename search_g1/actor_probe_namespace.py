"""Namespace-local ledger origins; never synthesize a parent's completed rows."""
from search_g1.actor_probe_contract import actor_probes


def namespace_origin(run):
    binding = run.get('actor_probe_recovery')
    if binding is None:
        return dict(completed_outer=0, actual_optimizer_writes=0,
                    attempt_index=0, aborted_attempts=0)
    assert actor_probes(run), 'Only the explicit actor-probe experiment may resume full state'
    assert binding['role'] == 'actor_probe_same_experiment_full_state_recovery'
    assert binding['state_semantics'] == 'full_model_optimizer_scheduler_rng_sampler_and_heads'
    origin = {key: binding[key] for key in ('completed_outer', 'actual_optimizer_writes',
                                          'attempt_index', 'aborted_attempts')}
    assert all(type(value) is int and value >= 0 for value in origin.values())
    assert origin['completed_outer'] in (5, 50, 100, 150)
    assert origin['attempt_index'] == origin['completed_outer'] + origin['aborted_attempts']
    assert origin['completed_outer'] <= origin['actual_optimizer_writes'] <= 20 * origin['completed_outer']
    assert binding['parent_run_id'] != run['run_id']
    assert binding['parent_execution_id'] != run['execution_id']
    return origin


def validate_driver_origin(run, payload):
    """Cross-check manifest counters against the actual saved driver payload."""
    origin = namespace_origin(run)
    progress = payload['attempt_progress']
    assert origin == dict(completed_outer=payload['completed_outer'],
        actual_optimizer_writes=progress['cumulative_writes'],
        attempt_index=progress['attempt_index'], aborted_attempts=progress['aborted_attempts'])
    assert progress['completed_outer'] == origin['completed_outer'] and progress['pending'] is None
    return origin


def validate_worker_origin(run, payload):
    origin = namespace_origin(run)
    assert payload['completed_outer'] == origin['completed_outer']
    assert payload['optimizer_writes'] == origin['actual_optimizer_writes']
    for key in ('actor_accounting', 'outer_accounting'):
        assert type(payload[key]['last_attempt_index']) is int
        assert payload[key]['last_attempt_index'] == origin['attempt_index']
    return origin
