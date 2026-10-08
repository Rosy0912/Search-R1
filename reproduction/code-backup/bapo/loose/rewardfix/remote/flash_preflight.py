"""Real per-rank BF16 FlashAttention varlen forward/backward check."""
import json,math,os
import torch
from flash_attn import flash_attn_varlen_func
rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);torch.manual_seed(42+rank)
q,k,v=[torch.randn(20,2,64,device='cuda',dtype=torch.bfloat16,requires_grad=True) for _ in range(3)]
cu=torch.tensor([0,8,20],device='cuda',dtype=torch.int32)
out=flash_attn_varlen_func(q,k,v,cu,cu,12,12,dropout_p=0.,causal=True)
refs=[]
for lo,hi in ((0,8),(8,20)):
 a,b,c=[x[lo:hi].transpose(0,1).float() for x in (q,k,v)]
 scores=a@b.transpose(-1,-2)/math.sqrt(64)
 scores=scores.masked_fill(torch.ones(hi-lo,hi-lo,device='cuda',dtype=torch.bool).triu(1),float('-inf'))
 refs.append((scores.softmax(-1)@c).transpose(0,1))
ref=torch.cat(refs)
assert torch.isfinite(out).all()
torch.testing.assert_close(out.float(),ref,rtol=.02,atol=.02)
out.float().square().mean().backward()
assert all(torch.isfinite(x.grad).all() and torch.count_nonzero(x.grad)>0 for x in (q,k,v))
print('FLASH_VARLEN_BF16_FORWARD_BACKWARD_JSON='+json.dumps({'rank':rank,'max_abs_reference_error':float((out.float()-ref).abs().max()),'finite_forward_backward':True}),flush=True)
