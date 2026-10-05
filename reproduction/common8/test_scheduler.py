"""Actual torch optimizer writes must use 1e-6, including the very first write."""
import unittest
from reproduction.common8.scheduler import constant_scheduler


class SchedulerTests(unittest.TestCase):
    def test_first_outer_twenty_real_adam_writes_use_requested_lr(self):
        import torch
        parameter=torch.nn.Parameter(torch.tensor([1.0],dtype=torch.float64))
        optimizer=torch.optim.AdamW([parameter],lr=1e-6)
        scheduler=constant_scheduler(optimizer)
        self.assertEqual(scheduler.get_last_lr(),[1e-6])
        previous=parameter.detach().clone()
        for _ in range(20):
            self.assertEqual(optimizer.param_groups[0]['lr'],1e-6)
            parameter.grad=torch.ones_like(parameter);optimizer.step()
            self.assertFalse(torch.equal(previous,parameter.detach()))
            previous=parameter.detach().clone()
        scheduler.step()
        self.assertEqual(optimizer.param_groups[0]['lr'],1e-6)

    def test_reject_different_base_lr(self):
        import torch
        p=torch.nn.Parameter(torch.ones(1));optimizer=torch.optim.AdamW([p],lr=2e-6)
        with self.assertRaises(AssertionError):constant_scheduler(optimizer)


if __name__=='__main__':unittest.main()
