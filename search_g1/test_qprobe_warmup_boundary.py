"""Actual CPU RNG/data mutation detection; no native GPU acceptance claim."""
import copy
import random
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from search_g1.qprobe_warmup_boundary import capture, verify


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.runtime = SimpleNamespace(selection_rng=np.random.default_rng(20261013))
        self.state = dict(order=[4, 2, 0, 1, 3], cursor=0, epoch=0,
                          pending_source=None, pending_offset=0)
        self.stream = SimpleNamespace(state=lambda: self.state)
        self.before = capture(self.runtime, self.stream)

    def tearDown(self):
        rng = self.before['driver_rng']
        torch.set_rng_state(rng['torch_cpu']); np.random.set_state(rng['numpy'])
        random.setstate(rng['python'])

    def test_actual_driver_rng_restoration_and_unchanged_data_pass(self):
        torch.rand(4); np.random.rand(4); random.random()
        rng = self.before['driver_rng']
        torch.set_rng_state(rng['torch_cpu']); np.random.set_state(rng['numpy'])
        random.setstate(rng['python'])
        proof = verify(self.before, capture(self.runtime, self.stream))
        self.assertTrue(all(proof.values()))

    def test_unrestored_rng_each_data_field_and_selector_draw_fail(self):
        torch.rand(1)
        with self.assertRaises(AssertionError): verify(self.before, capture(self.runtime, self.stream))
        torch.set_rng_state(self.before['driver_rng']['torch_cpu'])
        for key, value in [('cursor', 1), ('order', [2, 4, 0, 1, 3]),
                           ('pending_source', torch.tensor([1])), ('epoch', 1)]:
            altered = copy.deepcopy(self.before); altered['stream'][key] = value
            with self.assertRaises(AssertionError): verify(self.before, altered)
        self.runtime.selection_rng.permutation(5)
        with self.assertRaises(AssertionError): verify(self.before, capture(self.runtime, self.stream))


if __name__ == '__main__': unittest.main()
