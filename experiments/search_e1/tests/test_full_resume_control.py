import sys,tempfile,os,random,json
from pathlib import Path
import torch,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'control'))
from full_resume import restore_worker
from types import SimpleNamespace
class Model:
 def __init__(self,p):self.p=p
 def named_parameters(self):return [('flat',self.p)]
def worker(p):
 p._local_shard=p.detach().clone()
 opt=torch.optim.AdamW([p],lr=1e-6)
 sch=torch.optim.lr_scheduler.LambdaLR(opt,lambda n:min(1,n/142))
 return SimpleNamespace(rank=0,world_size=8,actor_module_fsdp=Model(p),actor_optimizer=opt,actor_lr_scheduler=sch)
a=worker(torch.nn.Parameter(torch.tensor([.07,.11],dtype=torch.float32)))
for iteration in range(2):
 for mini in range(10):
  a.actor_optimizer.zero_grad();a.actor_module_fsdp.p.grad=torch.tensor([.3+mini,.5]);a.actor_optimizer.step()
 a.actor_lr_scheduler.step()
with tempfile.TemporaryDirectory() as d:
 root=Path(d);(root/'receipts').mkdir();os.environ['E1_BASE']=d;os.environ['E1_RUN_ID']='cpu-restore-proof'
 torch.save(dict(model={'flat':a.actor_module_fsdp.p.detach().clone()},optimizer=a.actor_optimizer.state_dict(),scheduler=a.actor_lr_scheduler.state_dict(),rank=0,world_size=8,global_step=2,parameter_shapes=[('flat',[2],'torch.float32')],torch_rng=torch.get_rng_state(),cuda_rng=torch.empty(0,dtype=torch.uint8),numpy_rng=np.random.get_state(),python_rng=random.getstate()),root/'full-rank-0.pt')
 b=worker(torch.nn.Parameter(torch.tensor([.07,.11],dtype=torch.float32)))
 old=torch.cuda.set_rng_state;torch.cuda.set_rng_state=lambda state:None
 try:r=restore_worker(b,d)
 finally:torch.cuda.set_rng_state=old
 assert r['weights_exact'] and r['optimizer_exact']
 assert torch.equal(a.actor_module_fsdp.p,b.actor_module_fsdp.p._local_shard)
 for mini in range(10):
  for w in [a,b]:
   w.actor_optimizer.zero_grad();w.actor_module_fsdp.p.grad=torch.tensor([.3+mini,.5]);w.actor_optimizer.step()
 for w in [a,b]:w.actor_lr_scheduler.step()
 assert torch.equal(a.actor_module_fsdp.p,b.actor_module_fsdp.p)
 assert a.actor_lr_scheduler.state_dict()==b.actor_lr_scheduler.state_dict()
 for key,state in a.actor_optimizer.state_dict()['state'].items():
  for name,value in state.items():assert torch.equal(value,b.actor_optimizer.state_dict()['state'][key][name])
 print('EXACT_NEXT_ADAM_UPDATE_AND_SCHEDULER_MATCH; LOCAL_SHARD_ALIAS_UPDATED')
 torch.manual_seed(42);original=list(torch.utils.data.DataLoader(list(range(169103)),batch_size=512,shuffle=True,drop_last=True))
 torch.manual_seed(42);replayed=list(torch.utils.data.DataLoader(list(range(169103)),batch_size=512,shuffle=True,drop_last=True))[2:]
 assert all(torch.equal(x,y) for x,y in zip(original[2:],replayed));print('FIRST_EPOCH_DATA_ORDER_EXACT_AFTER_TWO_BATCH_SKIP')
