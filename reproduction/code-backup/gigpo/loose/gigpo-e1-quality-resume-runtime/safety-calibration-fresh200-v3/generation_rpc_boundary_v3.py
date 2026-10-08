"""Observe the actual synchronous generation RPC boundary, with no extra call."""
from contextlib import contextmanager
from observer_safety_v2 import patches


@contextmanager
def track_generation_rpc(worker_group, progress):
    """Only an actual successful generate_sequences return proves rank quiescence.

    Controller preparation, postprocessing, retrieval and trace IO remain local
    driver stages. Unknown original RPC failures retain workers_safe_to_snapshot=False throughout
    cleanup and never dispatch another model or recovery call here.
    """
    original = worker_group.generate_sequences
    record = dict(observer_errors=[], calls_started=0, calls_returned=0, active_rpc=False)
    progress['generation_rpc_boundary'] = record

    def observed(*args, **kwargs):
        record['calls_started'] += 1
        record['active_rpc'] = True
        progress['stage'] = 'generate_sequences_rpc/' + str(record['calls_started'])
        progress['workers_safe_to_snapshot'] = False
        value = original(*args, **kwargs)  # Original synchronous worker API exactly once.
        progress['workers_safe_to_snapshot'] = True  # First action after genuine RPC return.
        record['active_rpc'] = False
        record['calls_returned'] += 1
        progress['last_generation_rpc_output'] = value
        progress['stage'] = 'rollout_driver_local_after_generation_rpc'
        return value

    with patches(record) as install:
        installed = install(worker_group, 'generate_sequences', observed)
        if not installed or worker_group.generate_sequences is not observed:
            raise RuntimeError('Cannot install exact generation RPC boundary observer; no RPC started')
        yield record
    if record['observer_errors'] or record.get('_observer_recording_failed'):
        raise RuntimeError('Generation RPC boundary observer restoration failed: ' + repr(record['observer_errors']))
