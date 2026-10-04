"""Actual archived PPO update, four-way versus paired eight-way CPU allreduce."""
import os,sys,types,tempfile
from pathlib import Path
import torch,torch.distributed as dist,torch.multiprocessing as mp
from tensordict import TensorDict
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).parent))
from test_physical_micro import method,patch_actor,UP,SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'control'))
import virtual_dp
package=types.ModuleType('search_e1');package.__path__=[]
sys.modules['search_e1']=package;sys.modules['search_e1.virtual_dp']=virtual_dp

def worker(rank,path):
 torch.set_num_threads(1);dist.init_process_group('gloo',init_method='file://'+path,rank=rank,world_size=8)
 oldgroup=dist.new_group([0,1,2,3]);torch.manual_seed(17)
 N=512;mask=(torch.rand(N,7)>.45).double();mask[:8]=0
 batch=TensorDict(dict(responses=torch.zeros(N,7,dtype=torch.long),input_ids=torch.randn(N,7,dtype=torch.double),attention_mask=mask,position_ids=torch.zeros(N,7,dtype=torch.long),loss_mask=mask,old_log_probs=torch.randn(N,7,dtype=torch.double)*.4,ref_log_prob=torch.randn(N,7,dtype=torch.double)*.4,advantages=torch.randn(N,7,dtype=torch.double)),batch_size=[N])
 source=(UP/'workers/actor/dp_actor.py').read_text()
 def run(fn,local,world,group,micro,mini,virtual):
  model=torch.nn.Linear(1,1,dtype=torch.double)
  with torch.no_grad():model.weight.fill_(.17);model.bias.fill_(-.2)
  opt=torch.optim.AdamW(model.parameters(),lr=1e-3);states=[];grads=[]
  def forward(micro_batch,temperature):
   lp=model(micro_batch['input_ids'].unsqueeze(-1)).squeeze(-1);return torch.sigmoid(lp),lp
  def step():
   for p in model.parameters():dist.all_reduce(p.grad,group=group);p.grad.div_(world)
   grads.append(torch.cat([p.grad.flatten() for p in model.parameters()]).clone())
   norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
   states.append(torch.cat([v.flatten() for p in model.parameters() for v in opt.state[p].values()]).clone());return norm
  actor=SimpleNamespace(config=SimpleNamespace(ppo_mini_batch_size=mini,ppo_micro_batch_size=micro,use_dynamic_bsz=False,state_masking=True,use_kl_loss=True,clip_ratio=.2,entropy_coeff=.001,kl_loss_type='low_var_kl',kl_loss_coef=.001),actor_module=model,actor_optimizer=opt,_forward_micro_batch=forward,_optimizer_step=step)
  data=SimpleNamespace(meta_info={'temperature':1.},select=lambda **kw:SimpleNamespace(batch=local))
  with patch.object(TensorDict,'cuda',lambda self:self),patch.dict(os.environ,{'E1_PHYSICAL_MICRO':'1' if virtual else '0','E1_VIRTUAL_DP_WORLD':'4' if virtual else '0','SEARCH_E1_TRAJECTORY_BATCHING':'0'}):metrics=fn(actor,data)
  weights=torch.cat([p.detach().flatten() for p in model.parameters()]);return grads,states,weights,metrics
 baseline=None
 if rank<4:
  baseline=run(method(source,'update_policy'),batch[rank*128:(rank+1)*128],4,oldgroup,16,64,False)
  means={}
  for key,values in baseline[3].items():
   v=torch.tensor(values,dtype=torch.double).mean();dist.all_reduce(v,group=oldgroup);means[key]=float(v/4)
  baseline=(*baseline[:3],means)
 dist.barrier()
 objects=[baseline if rank==0 else None];dist.broadcast_object_list(objects,src=0);reference=objects[0]
 order=virtual_dp.stripe_partitions([list(range(r*128,(r+1)*128)) for r in range(4)])
 actual=run(method(patch_actor(source),'update_policy'),batch[order[rank*64:(rank+1)*64]],8,None,8,32,True)
 for a,b in zip(actual[:3],reference[:3]):
  if isinstance(a,list):
   for x,y in zip(a,b):torch.testing.assert_close(x,y,atol=1e-12,rtol=1e-12)
  else:torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)
 for key,values in actual[3].items():
  v=torch.tensor(values,dtype=torch.double).mean();dist.all_reduce(v);torch.testing.assert_close(v/8,torch.tensor(reference[3][key],dtype=torch.double),atol=1e-7,rtol=1e-7)
 # A zero-token virtual pair must fail on every rank before training collectives.
 group=virtual_dp.make_pair_group()
 try:virtual_dp.pair_tokens(0 if rank<2 else 1,group)
 except FloatingPointError:pass
 else:raise AssertionError('Zero virtual pair accepted')
 dist.barrier()
 if rank==0:print('PAIRED_8_EQUALS_ACTUAL_4_GRADIENTS_ADAM_STATES_WEIGHTS_1e-12; ALL_RANKS_REJECT_ZERO_PAIR',flush=True)
 dist.destroy_process_group()
if __name__=='__main__':
 with tempfile.TemporaryDirectory() as d:mp.spawn(worker,args=(str(Path(d)/'gloo'),),nprocs=8,join=True)
