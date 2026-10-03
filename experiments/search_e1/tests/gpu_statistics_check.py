"""Real four-rank FSDP, compiled entropy and FlashAttention CE equivalence."""
import os,sys,json,ast
from pathlib import Path
import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP,MixedPrecision
sys.path.insert(0,str(Path(__file__).parents[1]/'control'))
from token_statistics import HELPER,patch_statistics
from verl.utils.torch_functional import entropy_from_logits,logprobs_from_logits
ns=dict(torch=torch,checkpoint=checkpoint,os=os,logprobs_from_logits=logprobs_from_logits);exec(HELPER,ns);statistics=ns['_e1_token_statistics']
rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);dist.init_process_group('nccl')
source=(Path(__file__).parents[1]/'archive/upstream/Search-R1/verl/workers/actor/dp_actor.py').read_text();ast.parse(patch_statistics(source))
entropy_fn=torch.compile(entropy_from_logits,dynamic=True)
torch.manual_seed(42+rank);hidden=torch.randn(37,8,device='cuda');labels=torch.randint(0,151936,(37,),device='cuda');weights=torch.rand(37,device='cuda');weights[:3]=0
for chunk in [7,16,128]:
 objects=[]
 for _ in range(2):
  torch.manual_seed(42);model=FSDP(torch.nn.Linear(8,151936,device='cuda'),device_id=rank,mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,reduce_dtype=torch.float32,buffer_dtype=torch.bfloat16));opt=torch.optim.AdamW(model.parameters(),lr=1e-3);objects.append((model,opt))
 outputs=[];gradients=[]
 for (model,opt),flag in zip(objects,[0,chunk]):
  os.environ['E1_LOGIT_CHUNK']=str(flag)
  logits=model(hidden).float();entropy,lp=statistics(logits,labels,entropy_fn);outputs.append((entropy.detach(),lp.detach()))
  ((-lp+.001*entropy)*weights).sum().div(weights.sum()).backward()
  gradients.append(torch.cat([p.grad.flatten().float() for p in model.parameters()]).clone());model.clip_grad_norm_(1.);opt.step()
 for left,right in zip(*outputs):torch.testing.assert_close(left,right,atol=1e-6,rtol=1e-6)
 torch.testing.assert_close(*gradients,atol=1e-6,rtol=1e-6)
 a,b=[model.state_dict() for model,_ in objects]
 for key in a:torch.testing.assert_close(a[key],b[key],atol=1e-6,rtol=1e-6)
 del objects,logits,entropy,lp,gradients,a,b;torch.cuda.empty_cache()
# Measure actual CUDA allocation, without saved-tensor hooks interacting with compile.
sizes=[]
for flag in [0,16]:
 os.environ['E1_LOGIT_CHUNK']=str(flag);torch.manual_seed(42)
 x=torch.randn(257,151936,device='cuda',requires_grad=True);y=torch.randint(0,151936,(257,),device='cuda')
 # Warm compilation and backward outside the measurement.
 entropy,lp=statistics(x,y,entropy_fn);(entropy.mean()*.001-lp.mean()).backward()
 del entropy,lp;x.grad=None;torch.cuda.synchronize();torch.cuda.empty_cache()
 baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
 entropy,lp=statistics(x,y,entropy_fn);(entropy.mean()*.001-lp.mean()).backward();torch.cuda.synchronize()
 sizes.append(torch.cuda.max_memory_allocated()-baseline)
 del x,y,entropy,lp;torch.cuda.empty_cache()
assert sizes[1]<sizes[0]*.75,sizes
print(json.dumps(dict(rank=rank,result='FSDP_ROW_STATISTICS_EQUIVALENCE_OK',chunks=[7,16,128],peak_additional_bytes=sizes)),flush=True)
dist.barrier();dist.destroy_process_group()
