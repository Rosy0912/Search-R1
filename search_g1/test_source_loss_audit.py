import unittest
import torch
from search_g1.source_loss_audit import contributions
from verl.trainer.ppo.core_algos import compute_policy_loss

class SourceLossTests(unittest.TestCase):
    def test_contributions_sum_to_original_loss_and_clip(self):
        old=torch.zeros(4,3);new=torch.tensor([[.1,.4,0.],[-.5,.1,0.],[.2,-.2,.3],[.1,.2,.1]])
        adv=torch.tensor([[1.]*3,[-1.]*3,[.3]*3,[-.4]*3]);mask=torch.tensor([[1.,1.,0.],[1.,0.,0.],[1.,1.,1.],[1.,1.,0.]])
        for fallback in [torch.tensor([False,True,True,False]),torch.zeros(4,dtype=torch.bool),torch.ones(4,dtype=torch.bool)]:
            for dual in [None,3.]:
                loss,clip,_=compute_policy_loss(old,new,adv,mask,.2,dual_clip=dual)
                proof=contributions(old,new,adv,mask,fallback,.2,dual)
                self.assertAlmostEqual(sum(v for k,v in proof.items() if k.endswith('/pg_loss_contribution')),float(loss),places=6)
                self.assertAlmostEqual(sum(v for k,v in proof.items() if k.endswith('/clip_fraction_contribution')),float(clip),places=6)
                self.assertAlmostEqual(sum(v for k,v in proof.items() if k.endswith('/policy_token_fraction')),1.,places=6)
                self.assertIsNone(new.grad)

if __name__=='__main__':unittest.main()
