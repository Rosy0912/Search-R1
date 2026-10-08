"""Independent precision oracle CPU checks; never native GPU acceptance."""
import importlib.util,sys,unittest
from pathlib import Path
import torch
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'bapo-e1-300-devicefix'))
from verl import DataProto
spec=importlib.util.spec_from_file_location('r3',Path(__file__).parent/'native-actor-fixture.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.DataProto=DataProto

def graph_gradient(data,micro,pg_on,kl_on):
 b=data.batch;local=len(data)//8;grads=[]
 for rank in range(8):
  total=torch.zeros(16,dtype=torch.float32)
  for start in range(rank*local,(rank+1)*local,micro):
   sl=slice(start,min(start+micro,(rank+1)*local));mask=b['loss_mask'][sl].bool();w=torch.linspace(-.04,.05,16,dtype=torch.float64).float().bfloat16().requires_grad_();current=-3+b['input_ids'][sl].bfloat16().matmul(w)
   # Separate valid-position branches reproduce actual actor PG/KL current edges.
   c=torch.where(mask,current,0.);old=torch.where(mask,b['old_log_probs'][sl],0.);adv=torch.where(mask,b['advantages'][sl],0.)if pg_on else torch.zeros_like(c);ratio=(c-old).exp();pg=torch.maximum(-adv*ratio,-adv*ratio.clamp(.8,1.2))
   x=torch.where(mask,b['ref_log_prob'][sl],0.)-torch.where(mask,current,0.);kl=(x.exp()-x-1).clamp(-10,10)
   denom=mask.sum(-1).clamp_min(1);loss=(torch.where(mask,pg+(.001 if kl_on else 0.)*kl,0.).sum(-1)/denom).sum()/local;loss.backward();total+=w.grad.float()
  grads.append(total)
 return torch.stack(grads).mean(0).double()
class Precision(unittest.TestCase):
 def test_real_HF_output_traversal_current_torch_namespace_counterexample(self):
  from transformers.modeling_outputs import CausalLMOutputWithPast
  from torch.distributed.utils import _apply_to_tensors
  from types import SimpleNamespace
  x=torch.tensor([1.],requires_grad=True);visited=[]
  _apply_to_tensors(lambda t:visited.append(t)or t,SimpleNamespace(logits=x));self.assertEqual(visited,[])
  out=_apply_to_tensors(lambda t:visited.append(t)or t,CausalLMOutputWithPast(logits=x));self.assertEqual(len(visited),1);self.assertIs(out.logits,x)
  text=(Path(__file__).parent/'tools/native-actor-fixture.py').read_text();self.assertIn('return CausalLMOutputWithPast(logits=logits)',text);self.assertNotIn('return SimpleNamespace(logits=logits)',text)
  self.assertLess(text.index('component-only-rank-'),text.index('production_forward_record = production_forward_precision_probe'))

 def test_independent_formula_equals_separate_autograd_all_masks_micro_pg_kl(self):
  for rows in (128,136):
   for kind in ('normal','emptyrank','all_empty','permutation','maskednan','saturation','ppo_ties'):
    data=m.make_data(rows,empty_rank=kind=='emptyrank',all_empty=kind=='all_empty');b=data.batch
    if kind=='permutation':data=m.select(data,torch.randperm(rows,generator=torch.Generator().manual_seed(7)));b=data.batch
    if kind=='maskednan':
     for key in ('old_log_probs','ref_log_prob','advantages'):b[key][~b['loss_mask'].bool()]=float('nan')
    if kind=='saturation':b['ref_log_prob'][:]=torch.where(torch.arange(rows)[:,None]%2==0,3.,-9.)
    if kind=='ppo_ties':
     cur=(-3+b['input_ids'].bfloat16().matmul(torch.linspace(-.04,.05,16,dtype=torch.float64).float().bfloat16())).double();b['old_log_probs']=cur.clone();b['advantages'][::2]=0.
    for micro in (1,2,4,16):
     for pg,kl in ((True,False),(False,True),(True,True)):
      with self.subTest(rows=rows,kind=kind,micro=micro,pg=pg,kl=kl):
       oracle=m.quantized_toy_reference_gradient(data,micro,pg,kl);actual=graph_gradient(data,micro,pg,kl);torch.testing.assert_close(actual,oracle,rtol=0.,atol=0.)
 def test_exact_3167_observed_mismatch_and_scaling_counterexample(self):
  data=m.make_data();actual=graph_gradient(data,1,True,True);fp64,_=m.reference_gradient(data);error=(actual-fp64).abs();self.assertAlmostEqual(float(error.max()),.0020527375424673634,places=15);self.assertEqual(int((error>2e-4+.03*fp64.abs()).sum()),10)
  oracle=m.quantized_toy_reference_gradient(data,1);torch.testing.assert_close(actual,oracle,rtol=0.,atol=0.)
  with self.assertRaises(AssertionError):torch.testing.assert_close(actual*8,oracle,rtol=.03,atol=2e-4)
 def test_original_micro_bounds_preserved(self):
  data=m.make_data();base=graph_gradient(data,1,True,True)
  for micro in (2,4,16):torch.testing.assert_close(graph_gradient(data,micro,True,True),base,rtol=.03,atol=2e-4)
 def test_clamp_at_upper_boundary_derivative_not_dropped(self):
  x=torch.tensor([9.,10.,11.],dtype=torch.float64,requires_grad=True);x.clamp(-10,10).sum().backward();torch.testing.assert_close(x.grad,torch.tensor([1.,1.,0.],dtype=torch.float64));self.assertIn('(x.exp() - x - 1) <= 10',(Path(__file__).parent/'native-actor-fixture.py').read_text())
if __name__=='__main__':unittest.main()
