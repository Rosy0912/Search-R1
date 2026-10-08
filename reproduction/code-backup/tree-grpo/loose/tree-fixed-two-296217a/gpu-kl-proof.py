"""Pinned saturated KL on real GPUs; analytical FP64 oracle, no model weights."""
import json,os
from pathlib import Path
import torch
from verl.trainer.ppo import core_algos
rank=int(os.environ['RANK']);device=torch.device('cuda',int(os.environ['LOCAL_RANK']))
torch.cuda.set_device(device)
assert torch.__version__.startswith('2.4.0')
assert core_algos.__file__.startswith('/dataset/shaoshitong/search-r1-rl/tree-grpo/candidates/tree-native-numeric-296217a/repo/')
records=[]
for dtype in (torch.float32,torch.bfloat16,torch.float16):
 delta=torch.cat([torch.linspace(-12,5,400,device=device),torch.tensor([-10000.,-100.,-11.,4.,80.,100.,10000.],device=device)]).to(dtype)
 current=(-delta).detach().requires_grad_();actual=core_algos.kl_penalty(current,torch.zeros_like(current),'low_var_kl')
 actual.sum().backward()
 oracle_current=(-delta.double()).clamp(min=-700).detach().requires_grad_()
 d=-oracle_current;oracle=(d.exp()-d-1).clamp(-10,10);oracle.sum().backward()
 tol=.04 if dtype==torch.bfloat16 else (.005 if dtype==torch.float16 else 1e-5)
 torch.testing.assert_close(actual.double(),oracle,rtol=tol,atol=tol)
 torch.testing.assert_close(current.grad.double(),oracle_current.grad,rtol=tol,atol=tol)
 assert torch.isfinite(actual).all() and torch.isfinite(current.grad).all()
 for bad in (float('nan'),float('inf'),float('-inf')):
  try:core_algos.kl_penalty(torch.tensor([bad],device=device,dtype=dtype),torch.zeros(1,device=device,dtype=dtype),'low_var_kl')
  except ValueError as error:assert 'Nonfinite' in str(error)
  else:raise AssertionError('Nonfinite KL input was not refused')
 records.append(dict(dtype=str(dtype),points=len(delta),max_abs_value_delta=float((actual.double()-oracle).abs().max()),max_abs_gradient_delta=float((current.grad.double()-oracle_current.grad).abs().max()),tolerance=tol,finite_backward=True,nonfinite_inputs_refused=True))
out=Path(os.environ['TREE_GPU_KL_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
(out/f'rank-{rank}.json').write_text(json.dumps(dict(status='PASS',rank=rank,torch=torch.__version__,core_algos_file=core_algos.__file__,records=records),indent=2))
