"""A real four-process CPU collective rejects a single-rank bad first micro."""

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
    finally:
        queue.put(dict(rank=rank, calls=actor.calls, error=error, mode_restored=actor.training,
                       hooks_removed=all(not layer._forward_hooks for layer in actor.model.layers)))
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
            finally:
                for child in children:
                    if child.is_alive(): child.terminate(); child.join(2)
