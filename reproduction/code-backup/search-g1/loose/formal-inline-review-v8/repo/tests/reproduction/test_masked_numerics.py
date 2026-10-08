import ast,pathlib,types,unittest
import torch
ROOT=pathlib.Path(__file__).resolve().parents[2]
def function(path,name,namespace):
 node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
 exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace);return namespace[name]
mean=function(ROOT/'verl/utils/torch_functional.py','masked_mean',{'torch':torch})
kl=function(ROOT/'verl/trainer/ppo/core_algos.py','kl_penalty',{'torch':torch})
pg=function(ROOT/'verl/trainer/ppo/core_algos.py','compute_policy_loss',{'torch':torch,'verl_F':types.SimpleNamespace(masked_mean=mean)})
class MaskedNumerics(unittest.TestCase):
 def test_masked_nan_does_not_affect_mean_or_valid_gradient(self):
  values=torch.tensor([2.,float('nan')],requires_grad=True);result=mean(values,torch.tensor([1.,0.]));self.assertEqual(result.item(),2.);result.backward();self.assertEqual(values.grad.tolist(),[1.,0.])
 def test_masked_policy_overflow_excluded_before_exp(self):
  old=torch.tensor([[0.,float('-inf')]]);new=torch.tensor([[.1,float('inf')]],requires_grad=True);a=torch.tensor([[1.,float('nan')]]);mask=torch.tensor([[1.,0.]])
  result,_,_=pg(old,new,a,mask,.2);result.backward();self.assertTrue(torch.isfinite(result));self.assertTrue(torch.isfinite(new.grad).all());self.assertEqual(new.grad[0,1].item(),0.)
 def test_clipped_kl_equals_float64_objective_and_has_finite_gradient(self):
  diff=torch.tensor([-100.,-20.,-1.,0.,1.,2.,3.,20.,100.],requires_grad=True)
  result=kl(torch.zeros_like(diff),diff,'low_var_kl');reference_diff=diff.detach().double().requires_grad_(True);reference=(reference_diff.exp()-reference_diff-1).clamp(-10,10)
  self.assertTrue(torch.allclose(result.double(),reference,atol=1e-6));result.sum().backward();self.assertTrue(torch.isfinite(diff.grad).all());reference.sum().backward();self.assertTrue(torch.allclose(diff.grad.double(),reference_diff.grad,atol=1e-6))
if __name__=='__main__':unittest.main()
