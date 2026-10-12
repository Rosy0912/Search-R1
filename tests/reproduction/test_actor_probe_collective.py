"""Real four-process CPU agreement prevents peers entering a later collective."""

import multiprocessing as mp
import tempfile
import time
import unittest
from pathlib import Path

import torch
import torch.distributed as dist


def worker(rank, init, queue):
    from datetime import timedelta
    from reproduction.common8.scheduler import collective_call
    from search_g1.actor_probe_features import detached_last_token_features
    from search_g1.actor_probe_model_state import validate_roundtrip_receipt
    from tests.reproduction.test_actor_probe_features import Actor
    dist.init_process_group('gloo', rank=rank, world_size=4, init_method=init,
                            timeout=timedelta(seconds=20))
    class CollectiveActor(Actor):
        calls = 0
        def forward(self, **kwargs):
            self.calls += 1
            tick = torch.tensor(1)
            dist.all_reduce(tick)
            return super().forward(**kwargs)
    actor = CollectiveActor().train()
    if rank == 0:
        with torch.no_grad(): actor.model.layers[1].linear.bias.fill_(float('nan'))
    error = None
    try:
        detached_last_token_features(actor, torch.ones(4, 2, dtype=torch.int64),
            torch.ones(4, 2, dtype=torch.int64), (1, 2), micro_batch_size=2, agree=collective_call)
    except Exception as exc:
        error = str(exc)
    run = dict(run_id='unit', execution_id='unit', source_commit='unit',
               execution_backend='ssh_local', training_world_size=4, profile='unit', slurm_job_id=None)
    receipt = dict(role='actor_probe_full_state_checkpoint', identity=run, completed_outer=5,
                   next_logical_update=6, actual_optimizer_writes=100,
                   workers=[dict(rank=r, completed_outer=5, optimizer_writes=100) for r in range(4)])
    if rank == 3: receipt['workers'].pop()
    preflight_error = release_error = None; model_loads = subsequent_calls = 0
    def preflight():
        validate_roundtrip_receipt(receipt, run, 5, rank)
        return dict(completed_outer=5)
    try:
        collective_call(preflight, stage='actor_probe_state_roundtrip_preflight')
        model_loads += 1
        dist.all_reduce(torch.tensor(1))
    except Exception as exc:
        preflight_error = str(exc)
    def release():
        if rank == 1: raise OSError('Injected rank-local offload failure')
        return dict(accepted=True)
    try:
        collective_call(release, stage='actor_probe_canonical_model_release')
        subsequent_calls += 1
        dist.all_reduce(torch.tensor(1))
    except Exception as exc:
        release_error = str(exc)
    queue.put(dict(rank=rank, calls=actor.calls, error=error, mode_restored=actor.training,
                   hooks_removed=all(not layer._forward_hooks for layer in actor.model.layers),
                   preflight_error=preflight_error, model_loads=model_loads,
                   release_error=release_error, subsequent_calls=subsequent_calls))
    dist.destroy_process_group()


class ActorProbeCollectiveTests(unittest.TestCase):
    def test_single_bad_rank_stops_all_before_second_forward(self):
        context = mp.get_context('spawn'); queue = context.Queue()
        with tempfile.TemporaryDirectory(prefix='probe-gloo-') as directory:
            init = 'file://' + str(Path(directory)/'store')
            children = [context.Process(target=worker, args=(rank, init, queue)) for rank in range(4)]
            for child in children: child.start()
            deadline = time.monotonic() + 35
            try:
                for child in children: child.join(max(.1, deadline - time.monotonic()))
                self.assertTrue(all(not child.is_alive() and child.exitcode == 0 for child in children))
                results = [queue.get(timeout=2) for _ in children]
                self.assertEqual(sorted(row['rank'] for row in results), [0, 1, 2, 3])
                self.assertTrue(all(row['calls'] == 1 and 'rejected' in row['error'] for row in results))
                self.assertTrue(all(row['mode_restored'] and row['hooks_removed'] for row in results))
                self.assertTrue(all(row['model_loads'] == 0 and
                    'roundtrip_preflight rejected' in row['preflight_error'] for row in results))
                self.assertTrue(all(row['subsequent_calls'] == 0 and
                    'canonical_model_release rejected' in row['release_error'] for row in results))
            finally:
                for child in children:
                    if child.is_alive(): child.terminate(); child.join(2)
