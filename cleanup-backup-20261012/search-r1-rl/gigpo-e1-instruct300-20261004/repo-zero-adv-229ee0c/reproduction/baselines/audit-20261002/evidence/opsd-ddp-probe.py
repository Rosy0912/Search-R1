import os,sys,tempfile,torch
sys.path.insert(0,'/Users/shaoshitong/Documents/Search-R1-RL-Baselines')
from search_r1.baselines.distillation import global_token_denominator
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
def run(rank,path):
 dist.init_process_group('gloo',init_method='file://'+path,rank=rank,world_size=2)
 model=DDP(torch.nn.Linear(1,1,bias=False)); model.module.weight.data.fill_(1.)
 counts=[3,1]; mask=torch.tensor([1.]*counts[rank]); d=global_token_denominator(mask)
 # Two backward calls emulate microaccumulation; DDP averages each one.
 for _ in range(2):
  value=2. if rank==0 else 8.
  loss=model(torch.tensor([[value/2.]])).sum()*counts[rank]/d
  loss.backward()
 # numerator = 3*2+1*8=14 / total mask=4
 torch.testing.assert_close(model.module.weight.grad,torch.tensor([[3.5]]))
 assert d.item()==2.; empty=global_token_denominator(torch.zeros(3)); assert empty.item()==.5
 print('rank',rank,'denominator',d.item(),'accumulated averaged gradient',model.module.weight.grad.item(),'zero-mask denominator',empty.item(),flush=True)
 dist.destroy_process_group()
if __name__=='__main__':
 with tempfile.TemporaryDirectory() as directory: mp.spawn(run,args=(directory+'/rendezvous',),nprocs=2,join=True)
