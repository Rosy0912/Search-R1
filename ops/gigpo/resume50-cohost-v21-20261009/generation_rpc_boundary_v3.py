"""Compatibility name: actual v4 frozen dispatch/await/collect observation."""
from contextlib import contextmanager
from worker_rpc_boundary_v4 import track_worker_group_rpcs

@contextmanager
def track_generation_rpc(worker_group, progress, *, _schema=None):
    with track_worker_group_rpcs(((worker_group, 'actor_rollout', ('generate_sequences',)),),
                                 progress, _schema=_schema) as record:
        yield record
