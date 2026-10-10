import unittest
import torch
from search_g1.entropy_memory import entropy_metric
from verl.utils.torch_functional import entropy_from_logits, logprobs_of_labels_v2
from verl.trainer.ppo.core_algos import compute_policy_loss


class ZeroEntropyMemoryTest(unittest.TestCase):
    def compare(self, coefficient, magnitude=1.0):
        torch.manual_seed(73)
        source=torch.randn(4,11,29)*magnitude
        labels=torch.randint(29,(4,11))
        mask=(torch.rand(4,11)>.3).float()
        advantages=torch.randn(4,11)
        old=logprobs_of_labels_v2(source,labels).detach()+.02
        results=[]
        for bounded in (False,True):
            logits=source.clone().requires_grad_();saved=[]
            def pack(t): saved.append(t.numel()*t.element_size());return t
            with torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t):
                logp=logprobs_of_labels_v2(logits,labels)
                entropy=(entropy_metric(logits,coefficient,entropy_from_logits) if bounded
                         else entropy_from_logits(logits))
                pg,*_=compute_policy_loss(old,logp,advantages,mask,.2,dual_clip=3.)
                loss=pg-(entropy*mask).sum()/mask.sum()*coefficient
                loss.backward()
            results.append((entropy.detach(),loss.detach(),logits.grad,sum(saved)))
        for index in range(3):self.assertTrue(torch.equal(results[0][index],results[1][index]))
        if coefficient==0:self.assertLess(results[1][3],results[0][3])
        else:self.assertEqual(results[1][3],results[0][3])

    def test_zero_coefficient_exact_values_gradients_and_less_saved_memory(self):
        self.compare(0.0)

    def test_nonzero_entropy_retains_original_gradient(self):
        self.compare(.05)

    def test_extreme_finite_logits_and_zero_coefficient(self):
        self.compare(0.0,100.)
