"""Two real CPU/Gloo DDP ranks prove global token weighting, including empty rank."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from search_r1.baselines.distillation import global_token_denominator


def worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=2)
    try:
        model = torch.nn.parallel.DistributedDataParallel(torch.nn.Linear(1, 1, bias=False))
        for empty_rank in (False, True):
            model.zero_grad()
            length, value = (1, 1.) if rank == 0 else (3, 3.)
            mask = torch.ones(length)
            if empty_rank and rank == 0:
                mask.zero_()
            denominator = global_token_denominator(mask)
            loss = (model(torch.full((length, 1), value)).squeeze(-1) * mask).sum() / denominator
            loss.backward()
            expected = 3. if empty_rank else 2.5
            torch.testing.assert_close(model.module.weight.grad, torch.tensor([[expected]]))
    finally:
        dist.destroy_process_group()


class GlobalTokenMean(unittest.TestCase):
    def test_unequal_rank_lengths_and_zero_action_rank(self):
        with tempfile.TemporaryDirectory() as tmp:
            mp.spawn(worker, args=(os.path.join(tmp, 'gloo-init'),), nprocs=2, join=True)


if __name__ == '__main__':
    unittest.main()
