import argparse,json,pathlib,time
import torch
from verl.utils.torch_functional import entropy_from_logits
p=argparse.ArgumentParser();p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args();assert not a.output.exists()
torch.manual_seed(42);start=time.perf_counter();results=[]
compiled=torch.compile(entropy_from_logits,dynamic=True)
for dtype in [torch.float32,torch.bfloat16]:
 x=torch.randn(3,17,device='cuda',dtype=dtype,requires_grad=True);y=compiled(x);expected=entropy_from_logits(x)
 assert torch.isfinite(y).all() and torch.allclose(y,expected,rtol=.01 if dtype==torch.bfloat16 else 1e-5,atol=.02 if dtype==torch.bfloat16 else 1e-6)
 y.sum().backward();assert torch.isfinite(x.grad).all() and torch.count_nonzero(x.grad)>0
 results.append({'dtype':str(dtype),'max_abs_delta':float((y-expected).abs().max()),'grad_norm':float(x.grad.float().norm()),'output_finite':True,'gradient_finite':True})
r={'status':'ACTOR_ENTROPY_COMPILE_CAPABILITY_OK','scope':'small GPU0 Torch compile/gradient capability; not real policy update acceptance','results':results,'wall_seconds':time.perf_counter()-start};a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
