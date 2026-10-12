from pathlib import Path
import os,sys,tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from verl import DataProto
from search_r1.baselines.objectives import attach_policy_weights,weighted_policy_loss,row_denominator
from search_r1.baselines.distillation import global_token_count

def fixture(method, empty):
    mask=torch.tensor([[1,0,0],[1,1,1],[1,1,0],[0,0,0]],dtype=torch.float)
    if empty: mask[:2]=0
    d=DataProto.from_dict({'responses':torch.zeros(4,3,dtype=torch.long),'attention_mask':mask.long(),'info_mask':mask.long(),'step_ids':torch.zeros(4,3,dtype=torch.long)},non_tensors={'uid':np.asarray(['a','b','a','b'])})
    attach_policy_weights(d,method)
    a=torch.tensor([1.,2.,3.,4.])[:,None].expand(4,3)
    return d,mask,a

def worker(rank, init):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+init,rank=rank,world_size=2)
    try:
        model=torch.nn.parallel.DistributedDataParallel(torch.nn.Linear(1,1,bias=False))
        for method in ['tree_grpo','topd','gigpo']:
            for empty in [False,True]:
                d,mask,a=fixture(method,empty)
                p=torch.tensor(0.,requires_grad=True)
                ref=weighted_policy_loss(torch.zeros(4,3),p.expand(4,3),a,mask,d.batch['policy_weights'],.2,4.,d.batch['step_ids'] if method=='gigpo' else None)[0]
                ref.backward()
                for micro in [1,2]:
                    model.module.weight.data.zero_();model.zero_grad()
                    off=rank*2
                    denom=row_denominator(2,torch.device('cpu'))
                    for start in range(off,off+2,micro):
                        stop=start+micro
                        cur=model(torch.ones(micro*3,1)).view(micro,3)
                        loss=weighted_policy_loss(torch.zeros_like(cur),cur,a[start:stop],mask[start:stop],d.batch['policy_weights'][start:stop],.2,denom,d.batch['step_ids'][start:stop] if method=='gigpo' else None)[0]
                        loss.backward()
                    torch.testing.assert_close(model.module.weight.grad,p.grad.reshape(1,1),rtol=1e-5,atol=1e-6)
                    if rank==0: print(method,'empty_rank',empty,'micro',micro,'grad',p.grad.item(),flush=True)
        opt=torch.optim.AdamW(model.parameters(),lr=.1,weight_decay=.1)
        model(torch.ones(1,1)).sum().backward();opt.step();opt.zero_grad()
        before=model.module.weight.detach().clone()
        total,_=global_token_count(torch.zeros(2,3))
        assert total.item()==0
        # Same collective skip decision on both ranks; no optimizer step.
        torch.testing.assert_close(model.module.weight,before,rtol=0,atol=0)
    finally: dist.destroy_process_group()
if __name__=='__main__':
    with tempfile.TemporaryDirectory() as td: mp.spawn(worker,args=(os.path.join(td,'init'),),nprocs=2,join=True)
