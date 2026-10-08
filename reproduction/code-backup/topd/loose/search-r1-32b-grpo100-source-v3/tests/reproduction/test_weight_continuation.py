"""Actual path selection and real AdamW/LambdaLR continuation semantics on CPU."""
import os
import unittest
from unittest.mock import patch
from omegaconf import OmegaConf
import torch
from verl.utils.weight_continuation import (
    completed_updates, model_initialization_path, scheduler_from_completed)


def config(k=100):
    return OmegaConf.create({'trainer': {'weight_continuation_step': k, 'acceptance_step': k+5},
                            'actor_rollout_ref': {'model': {'path': '/base',
                                                          'actor_checkpoint_path': '/checkpoint'}}})


class WeightContinuation(unittest.TestCase):
    def test_actor_reference_are_separate_even_combined_worker(self):
        c = config().actor_rollout_ref.model
        self.assertEqual(model_initialization_path(c, True), '/checkpoint')
        self.assertEqual(model_initialization_path(c, False), '/base')
        c.actor_checkpoint_path = '/base'
        for actor in (False, True):
            with self.assertRaises(ValueError):
                model_initialization_path(c, actor)

    def test_no_checkpoint_preserves_original_model_for_every_role(self):
        c = OmegaConf.create({'path': '/base'})
        self.assertEqual(model_initialization_path(c, True), '/base')
        self.assertEqual(model_initialization_path(c, False), '/base')

    def test_counts_five_window_and_invalid_boundaries(self):
        for k in (5,100,495):
            self.assertEqual(completed_updates(config(k),501), k)
        for k in (-1,True,1.5,496,500):
            with self.assertRaises(ValueError):
                completed_updates(config(k),501)
        c = config(); c.trainer.acceptance_step=5
        with self.assertRaises(ValueError): completed_updates(c,501)
        c=config();c.actor_rollout_ref.model.actor_checkpoint_path=None
        with self.assertRaises(ValueError): completed_updates(c,501)
        c=config(0);c.actor_rollout_ref.model.actor_checkpoint_path=None
        self.assertEqual(completed_updates(c,501),0)

    def test_real_scheduler_next_lr_matches_uninterrupted_without_adam_state(self):
        for k in (0,1,5,100,141,142,200):
            with self.subTest(k=k):
                p=torch.nn.Parameter(torch.tensor(1.))
                old=torch.optim.AdamW([p],lr=1e-6)
                schedule=scheduler_from_completed(old,142,0)
                for _ in range(k):
                    p.grad=torch.ones_like(p);old.step();schedule.step()
                fresh=torch.optim.AdamW([torch.nn.Parameter(p.detach().clone())],lr=1e-6)
                restored=scheduler_from_completed(fresh,142,k)
                self.assertEqual(len(fresh.state),0)
                self.assertEqual(restored.last_epoch,k)
                self.assertEqual(fresh.param_groups[0]['lr'],old.param_groups[0]['lr'])
                self.assertAlmostEqual(fresh.param_groups[0]['lr'],1e-6*min(1,k/142))
                fresh.param_groups[0]['params'][0].grad=torch.tensor(1.)
                fresh.step();restored.step()
                self.assertAlmostEqual(fresh.param_groups[0]['lr'],1e-6*min(1,(k+1)/142))


if __name__ == '__main__':
    unittest.main()
