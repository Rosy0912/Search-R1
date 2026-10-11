"""Actual partition/reduction CPU tests; no GPU equality or speed claim."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import unittest
import threading
import queue
import numpy as np
import torch
from eight_gpu_streaming import (partition_tiles, merge_tile_original_order,
                                 eight_gpu_memory_plan, ScanJob, publish)
from pinned_overlap import tile_plan, Tile, DoubleBufferTiles
from .test_lifecycle_cpu import FakeBackend


class Checks(unittest.TestCase):
    def test_global_original_tiles_exact_unique_and_evenly_distributed(self):
        groups = partition_tiles(21015324)
        flat = sorted([tile for group in groups for tile in group], key=lambda x:x.number)
        self.assertEqual(flat, tile_plan(21015324))
        self.assertEqual([len(group) for group in groups], [81,81,80,80,80,80,80,80])
        for gpu, group in enumerate(groups):
            self.assertTrue(all(tile.number % 8 == gpu for tile in group))
        self.assertEqual(flat[-1].stop-flat[-1].start, 11036)
        memory = eight_gpu_memory_plan(21015324,768,40)
        self.assertEqual(sum(memory['per_gpu_vector_transfer_bytes']), 64559075328)
        self.assertEqual(memory['all_workers_pinned_cpu_buffers_bytes'], 1610612736)

    def test_original_order_merge_identical_even_with_exact_ties(self):
        torch.manual_seed(42)
        q, d, k = 5, 7, 3
        queries = torch.randint(-2,3,(q,d)).float()
        # Intentionally many equal scores and duplicate vectors.
        vectors = torch.randint(-1,2,(101,d)).float(); vectors[4:12] = vectors[0]
        baseline_s = torch.full((q,k),-float('inf')); baseline_i = torch.full((q,k),-1,dtype=torch.long)
        candidate_s, candidate_i = baseline_s.clone(), baseline_i.clone()
        computed = {}
        for number, start in enumerate(range(0,len(vectors),11)):
            distance = queries @ vectors[start:start+11].T
            scores, ids = torch.topk(distance,min(k,distance.shape[1]),dim=1,sorted=True)
            merged_s = torch.cat((baseline_s,scores),dim=1)
            merged_i = torch.cat((baseline_i,ids+start),dim=1)
            baseline_s, choices = torch.topk(merged_s,k,dim=1,sorted=True)
            baseline_i = torch.gather(merged_i,1,choices)
            computed[number] = (start,scores.clone(),ids.clone())
        # Workers may finish in another order; coordinator still reduces exactly
        # the original ordered tile local scores, with no per-shard aggregate.
        arrived = dict(reversed(list(computed.items())))
        for number in sorted(arrived):
            start,scores,ids = arrived[number]
            candidate_s,candidate_i = merge_tile_original_order(
                candidate_s,candidate_i,scores,ids,start,k,torch)
        self.assertTrue(torch.equal(candidate_s,baseline_s))
        self.assertTrue(torch.equal(candidate_i,baseline_i))

    def test_each_round_robin_worker_uses_local_alternating_buffer_slots(self):
        tiles = partition_tiles(32768*24+7)[0]
        backend = FakeBackend(); buffers = DoubleBufferTiles(backend)
        first = tiles[0]; buffers.stage(Tile(0,first.start,first.stop))
        for number, global_tile in enumerate(tiles):
            if number+1 < len(tiles):
                following = tiles[number+1]
                buffers.stage(Tile(number+1,following.start,following.stop))
            local = Tile(number,global_tile.start,global_tile.stop)
            self.assertEqual(buffers.acquire(local),local)
            buffers.consumed(local)
        buffers.close()
        self.assertEqual([event[1] for event in backend.events if event[0]=='stage'],list(range(len(tiles))))

    def test_bounded_producer_cancellation_does_not_strand_thread(self):
        job = ScanJob([], [], None, threading.Event())
        job.output.put('first'); job.output.put('second')
        outcome = []
        thread = threading.Thread(target=lambda:outcome.append(publish(job,'blocked')))
        thread.start(); job.cancel.set(); thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(outcome,[False])
        self.assertEqual(job.output.qsize(),2)


if __name__ == '__main__': unittest.main()
