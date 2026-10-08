import unittest
import math
import torch
import objective
from precision import PrecisionContract
from targets import SharedTargets
from test_targets import Toy,old_identity

class ReverseTests(unittest.TestCase):
    def test_three_asymmetric_full_gradient_and_difference_from_forward(self):
        for sv,tv in [([1.,-1.,.2],[0.,2.,-1.]),([4.,.5,-3.],[-2.,1.,2.]),([-.2,1.,1.5],[2.,-2.,.2])]:
            s=torch.tensor([sv],requires_grad=True);t=torch.tensor([tv],requires_grad=True)
            m=objective.vocabulary_metrics(s,t,None);m['raw'].sum().backward();p=s.detach().double().softmax(-1);logp=s.detach().double().log_softmax(-1);logt=t.detach().double().log_softmax(-1)
            ref=(p*(logp-logt)).sum(-1);grad=p*(logp-logt-ref[:,None])
            self.assertTrue(torch.allclose(m['raw'].detach().double(),ref,atol=2e-6));self.assertTrue(torch.allclose(s.grad.double(),grad,atol=2e-6));self.assertIsNone(t.grad)
            self.assertFalse(torch.allclose(s.grad,p.float()-t.detach().softmax(-1),atol=1e-4))
            eps=1e-5
            for i in range(3):
                z=s.detach().double();hi=z.clone();lo=z.clone();hi[0,i]+=eps;lo[0,i]-=eps
                f=lambda x:(x.softmax(-1)*(x.log_softmax(-1)-logt)).sum()
                self.assertAlmostEqual(((f(hi)-f(lo))/(2*eps)).item(),grad[0,i].item(),places=7)
    def test_equal_and_mixture_endpoints_and_initial_bound(self):
        s=torch.tensor([[1.,-2.,.2]],requires_grad=True);q=torch.tensor([[-2.,3.,.1]],requires_grad=True);o=s.detach().clone().requires_grad_()
        for a in (0.,.1,1.):
            m=objective.vocabulary_metrics(s,q,None,o,a)
            expected=a*q.detach().softmax(-1)+(1-a)*o.detach().softmax(-1)
            self.assertTrue(torch.allclose(m['target_logp'].exp(),expected,atol=1e-7))
            if a==.1:self.assertLessEqual(m['raw'].item(),-math.log(.9)+1e-6)
        eq=objective.vocabulary_metrics(s,s,None);g=torch.autograd.grad(eq['raw'].sum(),s)[0];self.assertTrue(torch.allclose(g,torch.zeros_like(g),atol=1e-7))
    def test_old_diagnostics_do_not_enable_mixture_in_control(self):
        z=torch.tensor([[1.,-1.]],requires_grad=True);q=torch.tensor([[-1.,1.]]);o=torch.tensor([[3.,-3.]])
        control=objective.vocabulary_metrics(z,q,10,o,.1,False)
        self.assertTrue(torch.equal(control['target_logp'],q.log_softmax(-1)))
        self.assertFalse(torch.equal(control['target_logp'],objective.vocabulary_metrics(z,q,10,o,.1,True)['target_logp']))
    def test_clip_only_upper_preserves_negative(self):
        s=torch.tensor([[0.,0.]],requires_grad=True);q=torch.tensor([[-100.,0.]])
        m=objective.vocabulary_metrics(s,q,10.);self.assertEqual(m['clipped_count'].item(),1)
        self.assertLess(m['post'].item(),10.);self.assertAlmostEqual(m['clipped_student_mass'].item(),.5)
    def test_detaching_student_coefficient_is_wrong(self):
        z=torch.tensor([[1.,-.2]],requires_grad=True);lp=z.log_softmax(-1);target=torch.tensor([[.2,.8]]).log()
        bad=(lp.exp().detach()*(lp-target)).sum();g=torch.autograd.grad(bad,z)[0]
        self.assertTrue(torch.allclose(g,torch.zeros_like(g),atol=1e-7))
        good=objective.vocabulary_metrics(z,target,None)['raw'];self.assertGreater(torch.autograd.grad(good.sum(),z)[0].abs().sum().item(),.01)
    def test_fp16_forward_and_recompute_have_fp32_master_gradients(self):
        m=Toy();c=PrecisionContract('fp32_master_fp16','cpu').install(m)
        t=SharedTargets(m,c,old_adapter='outer_old',old_identity=old_identity(m))
        enc={'prefixes':[[1,2],[3,4,5]],'response':[6,7,8], 'mask':[1,1,1], 'offsets':[(0,1),(1,2),(2,3)]}
        with t.current_graph():
            loss,_=objective.pair_objective(m,enc,10,32,3,precision=c,targets=t,use_mixture=True)
            self.assertEqual(loss.dtype,torch.float32);t.mark_backward();loss.backward()
        self.assertEqual(m.lora_A['default'].weight.grad.dtype,torch.float32)
        roles={e['role'] for e in c.events}
        self.assertTrue({'teacher','outer_old','current','current_head','teacher_head','outer_old_head'}<=roles)
        self.assertTrue(any(e['stage']=='backward_recompute' for e in c.events));c.close()
    def test_full_vocabulary_callback_once_through_checkpoint_recompute(self):
        m=Toy();c=PrecisionContract(device_type='cpu');t=SharedTargets(m,c,old_adapter='outer_old',old_identity=old_identity(m));seen=[];scalars=[]
        enc={'prefixes':[[1,2],[3,4,5]],'response':list(range(6,46)),'mask':[1]*35+[0]*5,'offsets':[(i,i+1) for i in range(40)]}
        with t.current_graph():
            loss,_=objective.pair_objective(m,enc,10,32,2,precision=c,targets=t,use_mixture=True,record=scalars.append,vocabulary_record=seen.append,vocabulary_positions=[0,33,39])
            t.mark_backward();loss.backward()
        self.assertEqual([x['token_index'] for x in seen],[0,33])
        for row in seen:
            for name in ('student','teacher','outer_old','target'):
                value=row[name+'_logp'];self.assertEqual(value.dtype,torch.float32);self.assertFalse(value.requires_grad)
                self.assertEqual(tuple(value.shape),(128,));self.assertAlmostEqual(value.exp().sum().item(),1.,places=6)
            scalar=next(x for x in scalars if x['token_index']==row['token_index'])
            pvalue=row['student_logp'].exp()
            for dist,field in [('teacher','student_to_teacher_raw_kl'),('outer_old','student_to_outer_old_raw_kl'),('target','student_to_mixture_raw_kl')]:
                actual=(pvalue*(row['student_logp']-row[dist+'_logp'])).sum().item()
                self.assertAlmostEqual(actual,scalar[field],places=6)
    def test_native_mask_chunk_recompute_top_branch_once_and_external_weight(self):
        m=Toy();c=PrecisionContract(device_type='cpu');t=SharedTargets(m,c,old_adapter='outer_old',old_identity=old_identity(m))
        enc={'prefixes':[[1,2],[3,4,5]],'response':list(range(6,46)),'mask':[1]*35+[0]*5,'offsets':[(i,i+1) for i in range(40)]};records=[]
        with t.current_graph():
            loss,stats=objective.pair_objective(m,enc,10,32,7,records.append,[0],precision=c,targets=t,pair_weight=2.)
            self.assertAlmostEqual(loss.item(),2*stats['post_loss'],places=6)
            t.mark_backward();(loss/7).backward()
        self.assertEqual(len(records),35);self.assertTrue(any(p.grad is not None for p in m.lora_A['default'].parameters()))
        self.assertAlmostEqual(sum(r['global_batch_contribution'] for r in records),loss.item()/7,places=6)
        self.assertIn('outer_old_observed_token_logp',records[34]);self.assertNotIn('student_top32_ids',records[34])

if __name__=='__main__':unittest.main()
