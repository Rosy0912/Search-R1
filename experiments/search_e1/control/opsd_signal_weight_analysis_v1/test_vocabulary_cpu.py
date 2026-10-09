"""Mandatory original-Node-env Torch CPU tests. No CUDA/model/GPU initialization.

Mac pure-Python fixtures run test_analysis separately; do not install Torch or
download an environment on Mac for these additional numerical checks.
"""
import unittest
try:
    from .test_analysis import source_module
    from . import replay_math as replay
except ImportError:
    from test_analysis import source_module
    import replay_math as replay


class VocabularyTests(unittest.TestCase):
    def test_full_vocabulary_matches_original_signed_component_clip(self):
        import torch
        objective = source_module('objective')
        s = torch.tensor([[0., 0.], [1., 2.]])
        t = torch.tensor([[100., -100.], [2., -1.]])
        b = torch.tensor([[1., 0.], [1., 2.]])
        values = replay.scalar_metrics(s, b, t, objective)
        sl = torch.log_softmax(s.float(), -1); tl = torch.log_softmax(t.float(), -1); bl = torch.log_softmax(b.float(), -1)
        comp = sl.exp() * (sl-tl)
        self.assertTrue(torch.equal(values['raw_teacher_reverse_kl'], comp.sum(-1)))
        self.assertTrue(torch.equal(values['post_clip_teacher_loss'], comp.clamp(max=10).sum(-1)))
        self.assertTrue(torch.equal(values['student_entropy'], -(sl.exp()*sl).sum(-1)))
        self.assertTrue(torch.equal(values['source100_ordinary_reverse_kl'], (sl.exp()*(sl-bl)).sum(-1)))
        self.assertTrue((comp < 0).any().item()); self.assertGreater(values['clipped_vocabulary_count'][0].item(), 0)
        self.assertFalse(torch.cuda.is_initialized())

    def test_full_vocabulary_no_B_keeps_ordinary_metrics(self):
        import torch
        objective = source_module('objective'); s = torch.tensor([[1., 2., 3.]])
        values = replay.scalar_metrics(s, s, None, objective)
        self.assertIsNone(values['teacher_logp']); self.assertIsNone(values['raw_teacher_reverse_kl'])
        self.assertAlmostEqual(values['source100_ordinary_reverse_kl'].item(), 0.)
        self.assertGreater(values['student_entropy'].item(), 0.)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__': unittest.main()
