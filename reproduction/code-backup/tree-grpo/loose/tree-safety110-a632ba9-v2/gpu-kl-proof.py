"""Pinned saturated KL on real GPUs; analytical FP64 oracle, no model weights."""
import json,os
from pathlib import Path
import torch
from verl.trainer.ppo import core_algos
rank=int(os.environ['RANK']);device=torch.device('cuda',int(os.environ['LOCAL_RANK']))
torch.cuda.set_device(device)
assert torch.__version__.startswith('2.4.0')
assert Path(core_algos.__file__).resolve() == (Path(__file__).resolve().parent / 'source/repo/verl/trainer/ppo/core_algos.py').resolve(), core_algos.__file__
records=[]
for dtype in (torch.float32,torch.bfloat16,torch.float16):
 delta=torch.cat([torch.linspace(-12,5,400,device=device),torch.tensor([-10000.,-100.,-11.,4.,80.,100.,10000.],device=device)]).to(dtype)
 current=(-delta).detach().requires_grad_();actual=core_algos.kl_penalty(current,torch.zeros_like(current),'low_var_kl')
 actual.sum().backward()
 oracle_current=(-delta.double()).clamp(min=-700).detach().requires_grad_()
 d=-oracle_current;oracle=(d.exp()-d-1).clamp(-10,10);oracle.sum().backward()
 tol=.04 if dtype==torch.bfloat16 else (.005 if dtype==torch.float16 else 1e-5)
 torch.testing.assert_close(actual.double(),oracle,rtol=tol,atol=tol)
 # Compare the original finite expression in its actual computation dtype.
 # FP16/BF16 rounding at raw KL=10 can change the clamp derivative versus
 # FP64, even when old and candidate dtype operations agree exactly.
 safe=(delta>=-12)&(delta<=5)
 native_current=(-delta[safe]).detach().requires_grad_();native_delta=-native_current
 native=(native_delta.exp()-native_delta-1).clamp(-10,10);native.sum().backward()
 torch.testing.assert_close(actual[safe],native,rtol=0,atol=0)
 torch.testing.assert_close(current.grad[safe],native_current.grad,rtol=0,atol=0)
 raw64=d.exp()-d-1;stable=(raw64-10).abs()>.5
 torch.testing.assert_close(current.grad.double()[stable],oracle_current.grad[stable],rtol=tol,atol=tol)
 boundaries=[]
 for index in (~stable).nonzero().flatten().tolist():
  boundaries.append(dict(index=index,delta=float(delta[index]),unclamped_fp64=float(raw64[index]),
   actual_dtype_gradient=float(current.grad[index]),fp64_gradient=float(oracle_current.grad[index])))
 assert torch.isfinite(actual).all() and torch.isfinite(current.grad).all()
 for bad in (float('nan'),float('inf'),float('-inf')):
  try:core_algos.kl_penalty(torch.tensor([bad],device=device,dtype=dtype),torch.zeros(1,device=device,dtype=dtype),'low_var_kl')
  except ValueError as error:assert 'Nonfinite' in str(error)
  else:raise AssertionError('Nonfinite KL input was not refused')
 records.append(dict(dtype=str(dtype),points=len(delta),max_abs_value_delta=float((actual.double()-oracle).abs().max()),max_abs_gradient_delta=float((current.grad.double()-oracle_current.grad).abs().max()),tolerance=tol,native_dtype_points_compared=int(safe.sum()),native_dtype_values_and_gradients_exact=True,fp64_stable_gradient_points=int(stable.sum()),fp64_boundary_points=boundaries,finite_backward=True,nonfinite_inputs_refused=True))
out=Path(os.environ['TREE_GPU_KL_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
(out/f'rank-{rank}.json').write_text(json.dumps(dict(status='PASS',rank=rank,torch=torch.__version__,core_algos_file=core_algos.__file__,records=records),indent=2))
