"""Compare actual FSDP AdamW steps with the same states staged through CPU."""
import ast,os,sys,json
from pathlib import Path
from types import SimpleNamespace,MethodType
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
sys.path.insert(0,str(Path(__file__).parents[1]/'control'))
from jit_optimizer import patch_actor_optimizer,patch_worker_optimizer
up=Path(__file__).parents[1]/'archive/upstream/Search-R1/verl'
def extract(source,name,ns):
 node=next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name==name)
 exec(compile(ast.Module(body=[node],type_ignores=[]),'<actual-upstream>','exec'),ns);return ns[name]
ns=dict(torch=torch,FSDP=FSDP,os=os)
utils=(up/'utils/fsdp_utils.py').read_text()
for name in ['load_fsdp_optimizer','offload_fsdp_optimizer']:extract(utils,name,ns)
source=(up/'workers/actor/dp_actor.py').read_text()
steps=[extract(source,'_optimizer_step',dict(ns)),extract(patch_actor_optimizer(source),'_optimizer_step',dict(ns))]
ast.parse(patch_worker_optimizer((up/'workers/fsdp_workers.py').read_text()))
rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);dist.init_process_group('nccl')
def actor(step):
 torch.manual_seed(42)
 model=FSDP(torch.nn.Linear(16,16,device='cuda'),device_id=rank)
 obj=SimpleNamespace(actor_module=model,actor_optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3),config=SimpleNamespace(grad_clip=1.,fsdp_config={'optimizer_offload':True}))
 obj._optimizer_step=MethodType(step,obj);return obj
a,b=[actor(step) for step in steps]
for iteration in range(4):
 torch.manual_seed(42+rank+iteration);inputs=torch.randn(32,16,device='cuda')
 for obj,flag in [(a,'0'),(b,'1')]:
  os.environ['E1_JIT_OPTIMIZER']=flag
  obj.actor_optimizer.zero_grad(set_to_none=True)
  obj.actor_module(inputs).square().mean().backward();obj._optimizer_step()
 sa=a.actor_module.state_dict();sb=b.actor_module.state_dict()
 for key in sa:torch.testing.assert_close(sa[key],sb[key],atol=0,rtol=0)
 states_a=list(a.actor_optimizer.state.values());states_b=list(b.actor_optimizer.state.values())
 for x,y in zip(states_a,states_b):
  for key in x:
   if isinstance(x[key],torch.Tensor):torch.testing.assert_close(x[key].cpu(),y[key].cpu(),atol=0,rtol=0);assert y[key].device.type=='cpu'
print(json.dumps(dict(rank=rank,result='FSDP_JIT_ADAMW_BITWISE_EQUIVALENCE_OK',updates=4)),flush=True)
dist.barrier();dist.destroy_process_group()
