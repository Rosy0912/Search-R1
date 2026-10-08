import torch, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'bapo-e1-300-devicefix'))
from search_r1.baselines.objectives import weighted_policy_loss
from verl.trainer.ppo.core_algos import kl_penalty
ids=torch.arange(128,dtype=torch.float64); token=torch.arange(7,dtype=torch.float64)
x=torch.stack([torch.sin(ids[:,None]*.17+token*.31+k*.13) for k in range(16)],-1)*.4
mask=token[None,:]<(ids.long()%8)[:,None]
old=-3+torch.cos(ids[:,None]*.21+token*.5)*.35
ref=-3+torch.cos(ids[:,None]*.1+token*.3)*.8
adv=torch.sin(ids[:,None]*.27+token*.4)*2
initial=torch.linspace(-.04,.05,16,dtype=torch.float64)
def objective(current, rows, denominator, pg=True, kl=True):
 m=mask[rows]; weights=m.double()/m.sum(-1,keepdim=True).clamp_min(1)
 loss,_,_=weighted_policy_loss(old[rows],current,adv[rows] if pg else adv[rows]*0,m,weights,.2,denominator)
 k=kl_penalty(torch.where(m,current,0.),torch.where(m,ref[rows],0.),'low_var_kl')
 return loss+(k*weights).sum()/denominator*(.001 if kl else 0)
refw=initial.clone().requires_grad_(); objective(-3+x.matmul(refw),slice(None),128).backward()
expected=refw.grad
results={}
for output_float in (False,True):
 for micro in (1,2,4,16):
  grads=[]; currents=[]
  for rank in range(8):
   g=torch.zeros(16,dtype=torch.float32)
   for start in range(rank*16,(rank+1)*16,micro):
    rows=slice(start,start+micro)
    w=initial.bfloat16().clone().requires_grad_()
    linear=x[rows].bfloat16().matmul(w)
    current=-3+(linear.float() if output_float else linear)
    objective(current,rows,16).backward()
    g+=w.grad.float(); currents.append(current.detach().double())
   grads.append(g)
  actual=torch.stack(grads).mean(0).double()
  c=torch.cat(currents)
  full=-3+x.matmul(initial); valid=mask
  ratio=(full-old).exp(); approx=(c-old).exp()
  flip=(((adv>0)&((ratio>1.2)!=(approx>1.2)))|((adv<0)&((ratio<.8)!=(approx<.8))))&valid
  err=(actual-expected).abs()
  key=f'BF16_linear_output_{"FP32" if output_float else "BF16"}_micro{micro}'
  results[key]=dict(gradient=actual.tolist(),max_abs=float(err.max()),max_rel=float((err/expected.abs()).max()),mismatch_at_frozen_tolerance=int((err>(expected.abs()*.03+.0002)).sum()),max_current_error=float((c-full)[valid].abs().max()),ppo_branch_flips=int(flip.sum()),loss=float(objective(c,slice(None),128)))
# Quantized forward exact-Jacobian diagnostic: isolate forward changes, no backward BF16 rounding.
w=initial.clone().requires_grad_(); ideal=-3+x.matmul(w)
bf16forward=-3+x.bfloat16().matmul(initial.bfloat16())
current=ideal+(bf16forward.double()-ideal).detach()
objective(current,slice(None),128).backward()
results['exact_jacobian_at_quantized_forward']=dict(gradient=w.grad.tolist(),max_abs=float((w.grad-expected).abs().max()))
out=dict(status='CPU_PRECISION_DIAGNOSTIC_NOT_NATIVE_ACCEPTANCE',torch=torch.__version__,expected=expected.tolist(),results=results)
Path(__file__).with_name('results.json').write_text(json.dumps(out,indent=2,allow_nan=False))
print(json.dumps({k:{a:b for a,b in v.items() if a!='gradient'} for k,v in results.items()},indent=2))
assert results['BF16_linear_output_BF16_micro1']['mismatch_at_frozen_tolerance']==10
assert abs(results['BF16_linear_output_BF16_micro1']['max_abs']-.0020527375424673634)<1e-15
assert abs(results['BF16_linear_output_BF16_micro1']['max_rel']-.5290988855198534)<1e-13
for micro in (1,2,4,16):
 key=f'BF16_linear_output_FP32_micro{micro}'
 assert results[key]['mismatch_at_frozen_tolerance']==0
 assert results[key]['ppo_branch_flips']==0
 base=torch.tensor(results['BF16_linear_output_FP32_micro1']['gradient'],dtype=torch.float64)
 actual=torch.tensor(results[key]['gradient'],dtype=torch.float64)
 torch.testing.assert_close(actual,base,rtol=.03,atol=.0002)
print('PASS_CPU_DIAGNOSIS: exact native error reproduced; unchanged bounds pass all four FP32-logprob micro sizes')
