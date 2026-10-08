import unittest
import torch
from search_g1.gradient_clipping import clip_global_l2

class Clipping(unittest.TestCase):
    def model(self,dtype=torch.float32):
        return torch.nn.Linear(2,2,bias=False,dtype=dtype)
    def test_matches_upstream_moderate(self):
        a=self.model();b=self.model();g=torch.tensor([[1.,2.],[3.,4.]])
        a.weight.grad=g.clone();b.weight.grad=g.clone()
        expected=torch.nn.utils.clip_grad_norm_(b.parameters(),1.)
        actual=clip_global_l2(a,1.,chunk_size=2,diagnostic=False)
        self.assertAlmostEqual(actual.item(),expected.item(),places=6)
        torch.testing.assert_close(a.weight.grad,b.weight.grad)
    def test_extreme_finite_fp32_norm_overflows_but_clipping_is_real(self):
        a=self.model();a.weight.grad=torch.full_like(a.weight,1e20)
        self.assertFalse(torch.isfinite(torch.linalg.vector_norm(a.weight.grad)))
        actual=clip_global_l2(a,1.,diagnostic=False)
        self.assertTrue(torch.isfinite(actual));self.assertGreater(actual.item(),1e20)
        torch.testing.assert_close(a.weight.grad,torch.full_like(a.weight,.5))
    def test_bf16_scalar_coefficient_underflow_does_not_zero_gradients(self):
        a=self.model(torch.bfloat16);a.weight.grad=torch.full_like(a.weight,1e38)
        clip_global_l2(a,1.,chunk_size=1,diagnostic=False)
        torch.testing.assert_close(a.weight.grad,torch.full_like(a.weight,.5))
    def test_nonfinite_rejected_without_write(self):
        a=self.model();a.weight.grad=torch.full_like(a.weight,2.);a.weight.grad[0,0]=float('nan')
        old=a.weight.grad.clone()
        with self.assertRaises(RuntimeError):clip_global_l2(a,1.,diagnostic=False)
        self.assertTrue(torch.equal(a.weight.grad[~torch.isnan(old)],old[~torch.isnan(old)]))
        self.assertTrue(torch.isnan(a.weight.grad[0,0]))
    def test_unsupported_layout_rejected_before_mutation(self):
        a=self.model();a.weight.grad=torch.tensor([[1.,2.],[3.,4.]]).t()
        old=a.weight.grad.clone()
        with self.assertRaises(RuntimeError):clip_global_l2(a,1.,diagnostic=False)
        torch.testing.assert_close(a.weight.grad,old)
    def test_no_gradient_rejected(self):
        with self.assertRaises(RuntimeError):clip_global_l2(self.model(),1.,diagnostic=False)

if __name__=='__main__':unittest.main()
