"""Compare actual FSDP AdamW steps with the same states staged through CPU."""
import ast,os,sys,json
from pathlib import Path
from types import SimpleNamespace,MethodType
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
sys.path.insert(0,str(Path(__file__).parents[1]/'control'))
from jit_optimizer import patch_actor_optimizer,patch_worker_optimizer
from optimizer_chunks import patch_optimizer_chunks,HELPER
from functools import partial
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
up=Path(__file__).parents[1]/'archive/upstream/Search-R1/verl'
def extract(source,name,ns):
 node=next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name==name)
 exec(compile(ast.Module(body=[node],type_ignores=[]),'<actual-upstream>','exec'),ns);return ns[name]
ns=dict(torch=torch,FSDP=FSDP,os=os);exec(HELPER,ns)
utils=(up/'utils/fsdp_utils.py').read_text()
for name in ['load_fsdp_optimizer','offload_fsdp_optimizer']:extract(utils,name,ns)
source=(up/'workers/actor/dp_actor.py').read_text()
steps=[extract(source,'_optimizer_step',dict(ns)),extract(patch_optimizer_chunks(patch_actor_optimizer(source)),'_optimizer_step',dict(ns))]
ast.parse(patch_worker_optimizer((up/'workers/fsdp_workers.py').read_text()))
rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);dist.init_process_group('nccl')
def actor(step):
 torch.manual_seed(42)
 model=FSDP(torch.nn.Sequential(*[torch.nn.Linear(1024,1024,device='cuda') for _ in range(8)]),device_id=rank,auto_wrap_policy=partial(size_based_auto_wrap_policy,min_num_params=128))
 obj=SimpleNamespace(actor_module=model,actor_optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3),config=SimpleNamespace(grad_clip=1.,fsdp_config={'optimizer_offload':True}))
 obj._optimizer_step=MethodType(step,obj);return obj
a,b=[actor(step) for step in steps]
for iteration in range(4):
 torch.manual_seed(42+rank+iteration);inputs=torch.randn(32,1024,device='cuda')
 for obj,flag in [(a,'0'),(b,'1')]:
  os.environ['E1_JIT_OPTIMIZER']=flag;os.environ['E1_OPTIMIZER_CHUNK']=flag
  groups=obj.actor_optimizer.param_groups;ids=[id(g) for g in groups]
  obj.actor_optimizer.zero_grad(set_to_none=True)
  obj.actor_module(inputs).square().mean().backward();obj._optimizer_step()
  assert obj.actor_optimizer.param_groups is groups and [id(g) for g in groups]==ids
 sa=a.actor_module.state_dict();sb=b.actor_module.state_dict()
 for key in sa:torch.testing.assert_close(sa[key],sb[key],atol=0,rtol=0)
 states_a=list(a.actor_optimizer.state.values());states_b=list(b.actor_optimizer.state.values())
 for x,y in zip(states_a,states_b):
  for key in x:
   if isinstance(x[key],torch.Tensor):torch.testing.assert_close(x[key].cpu(),y[key].cpu(),atol=0,rtol=0);assert y[key].device.type=='cpu'
# Both optimizers have identical resident states for the memory comparison.
ns['load_fsdp_optimizer'](b.actor_optimizer,rank)
for obj in [a,b]:
 obj.actor_optimizer.zero_grad(set_to_none=True)
 obj.actor_module(inputs).square().mean().backward()
peaks=[]
for obj,flag in [(a,'0'),(b,'1')]:
 os.environ['E1_OPTIMIZER_CHUNK']=flag
 torch.cuda.synchronize();torch.cuda.empty_cache();baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
 ns['_e1_optimizer_step'](obj.actor_optimizer);torch.cuda.synchronize()
 peaks.append(torch.cuda.max_memory_allocated()-baseline)
assert peaks[1]<peaks[0]*.5,peaks
sa=a.actor_module.state_dict();sb=b.actor_module.state_dict()
for key in sa:torch.testing.assert_close(sa[key],sb[key],atol=0,rtol=0)
print(json.dumps(dict(rank=rank,result='FSDP_CHUNKED_ADAMW_BITWISE_EQUIVALENCE_OK',updates=4,parameter_tensors=len(list(a.actor_module.parameters())),additional_step_peak_bytes=peaks)),flush=True)
dist.barrier();dist.destroy_process_group()
