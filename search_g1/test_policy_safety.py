import math,unittest,tempfile,json
from pathlib import Path
import torch
import torch.multiprocessing as mp
from search_g1.policy_safety import dual_clip_terms,monitor


def collective_worker(rank,path,out):
    torch.distributed.init_process_group('gloo',init_method='file://'+path,rank=rank,world_size=2)
    d=torch.tensor([[.1,.2]] if rank==0 else [[.3,2.]],dtype=torch.float64)
    r=monitor(d,-torch.ones_like(d),torch.ones_like(d),torch.ones_like(d),c=3.,kl_log_bound=30.,target_kl=10.,negative_tail_limit=.1)
    Path(out+str(rank)).write_text(json.dumps(r))
    torch.distributed.destroy_process_group()


class SafetyTests(unittest.TestCase):
    def test_direct_formula_values_and_gradients(self):
        d=torch.tensor([-2.,-.1,.1,.3,1.,2.]*3,dtype=torch.float64,requires_grad=True)
        a=torch.tensor([1.]*6+[-1.]*6+[0.]*6,dtype=torch.float64)
        ratio=d.exp();ordinary=torch.maximum(-a*ratio,-a*ratio.clamp(.8,1.2))
        direct=torch.where(a<0,torch.minimum(ordinary,-a*3.),ordinary)
        stable=dual_clip_terms(d,a,torch.ones_like(a),.2,3.)
        self.assertTrue(torch.allclose(stable,direct))
        g1=torch.autograd.grad(stable.sum(),d,retain_graph=True)[0]
        g2=torch.autograd.grad(direct.sum(),d)[0]
        self.assertTrue(torch.allclose(g1,g2))

    def test_extreme_zero_padding_and_gradients(self):
        d=torch.tensor([800.,-800.,800.,float('nan')],requires_grad=True)
        a=torch.tensor([-1.,1.,0.,float('nan')]);m=torch.tensor([1.,1.,1.,0.])
        terms=dual_clip_terms(d,a,m,.2,3.);terms.sum().backward()
        self.assertTrue(torch.isfinite(terms).all() and torch.isfinite(d.grad).all())
        self.assertAlmostEqual(float(terms[0].detach()),3.);self.assertEqual(float(d.grad[0]),0.)
        with self.assertRaises(ValueError):dual_clip_terms(d,a,torch.ones_like(m),.2,3.)

    def test_statistics_reject_invalid_and_over_bound_without_fake_kl(self):
        for value in [800.,float('nan'),float('inf')]:
            d=torch.tensor([value]);r=monitor(d,-torch.ones(1),torch.ones(1),torch.ones(1),c=3.,kl_log_bound=30.,target_kl=.01,negative_tail_limit=.1)
            self.assertFalse(r['allowed']);self.assertIsNone(r['sampled_old_policy_kl'])
        r=monitor(torch.tensor([float('nan'),0.]),torch.tensor([float('nan'),1.]),torch.tensor([0.,1.]),torch.tensor([float('inf'),1.]),c=3.,kl_log_bound=30.,target_kl=.01,negative_tail_limit=.1)
        self.assertTrue(r['allowed']);self.assertEqual(r['sampled_old_policy_kl'],0.)

    def test_empty_and_negative_tail(self):
        d=torch.tensor([2.,0.]);a=-torch.ones(2);m=torch.ones(2)
        r=monitor(d,a,m,m,c=3.,kl_log_bound=30.,target_kl=10.,negative_tail_limit=.1)
        self.assertEqual(r['negative_tail_count_fraction'],.5);self.assertFalse(r['allowed'])
        r=monitor(d,a,torch.zeros(2),torch.zeros(2),c=3.,kl_log_bound=30.,target_kl=.01,negative_tail_limit=.1)
        self.assertIn('empty_effective_samples',r['reasons'])

    def test_existing_objective_weights_survive_partition(self):
        d=torch.tensor([[-2.,.1],[.2,1.],[2.,.3],[-.3,.4]],requires_grad=True)
        a=torch.tensor([[1.,1.],[-1.,-1.],[1.,-1.],[-1.,1.]])
        m=torch.tensor([[1.,1.],[1.,0.],[1.,1.],[0.,1.]])
        weights=m/torch.tensor([[3.],[3.],[3.],[3.]])
        full=(dual_clip_terms(d,a,m,.2,3.)*weights).sum()
        split=sum((dual_clip_terms(dd,aa,mm,.2,3.)*ww).sum() for dd,aa,mm,ww in zip(d.split(1),a.split(1),m.split(1),weights.split(1)))
        self.assertTrue(torch.equal(full,split))

    def test_actual_two_rank_global_tail_and_decision(self):
        with tempfile.TemporaryDirectory() as root:
            mp.spawn(collective_worker,args=(root+'/init',root+'/rank'),nprocs=2,join=True)
            a,b=[json.loads(Path(root+'/rank'+str(i)).read_text()) for i in range(2)]
        self.assertEqual(a,b);self.assertFalse(a['allowed'])
        self.assertEqual(a['negative_tail_count_fraction'],.25)
        self.assertEqual(a['negative_log_ratio_p99'],2.)
        expected=sum(math.expm1(x)-x for x in [.1,.2,.3,2.])/4
        self.assertAlmostEqual(a['sampled_old_policy_kl'],expected)


if __name__=='__main__':unittest.main()
