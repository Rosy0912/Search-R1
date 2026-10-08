"""Real eight-rank Torch FSDP oracle; synthetic gradients, no training updates."""
import copy
import datetime
import json
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'repo'))
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, CPUOffload
from search_r1.baselines import gradient_norm
from search_r1.baselines.gradient_norm import clip_grad_norm64


class Tied(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a=torch.nn.Linear(16,16,bias=False)
        self.b=torch.nn.Linear(16,16,bias=False)
        self.b.weight=self.a.weight
    def forward(self,x):
        return self.a(x)+self.b(x)


class Mixed(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shard=torch.nn.Linear(16,16,bias=False)
        self.replica=Tied()
    def forward(self,x):
        return self.shard(x)+self.replica(x)


rank=int(os.environ['RANK']);device=torch.device('cuda',int(os.environ['LOCAL_RANK']))
torch.cuda.set_device(device)
dist.init_process_group('nccl',timeout=datetime.timedelta(seconds=120))
assert dist.get_world_size()==8 and torch.__version__.startswith('2.4.0')
records=[]

from search_r1.baselines.objectives import weighted_policy_loss
ratio_records=[]
for ratio_dtype in (torch.float32,torch.bfloat16):
    for advantage in (1.,0.,-1e-6):
        x=torch.full((1,500),.2,device=device,dtype=ratio_dtype,requires_grad=True)
        adv=torch.full_like(x,advantage);weight=torch.full_like(x,.002/500)
        loss,_,_=weighted_policy_loss(torch.zeros_like(x),x,adv,torch.ones_like(x),weight,.2,32.,torch.zeros_like(x,dtype=torch.long))
        oracle=x.detach().double().requires_grad_();summed=oracle.sum(dim=1,keepdim=True)
        summed=summed+(x.detach().sum(dim=1,keepdim=True).double()-summed.detach())
        r=summed.exp().expand_as(oracle)
        expected=(torch.maximum(-adv.double()*r,-adv.double()*r.clamp(.8,1.2))*weight.double()).sum()/32
        expected.backward();loss.backward()
        assert loss.dtype==torch.float64 and torch.isfinite(x.grad).all()
        torch.testing.assert_close(loss,expected,rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(x.grad,oracle.grad.to(ratio_dtype),rtol=1e-6,atol=1e-12)
        if advantage>=0:assert torch.count_nonzero(x.grad)==0
        ratio_records.append(dict(case='joint500_delta100_fp64_oracle',advantage=advantage,dtype=str(ratio_dtype),loss=float(loss.detach()),gradient_max=float(x.grad.abs().max())))
x=torch.full((1,500),.2,device=device,requires_grad=True)
loss,_,_=weighted_policy_loss(torch.zeros_like(x),x,-torch.ones_like(x),torch.ones_like(x),torch.full_like(x,.002/500),.2,32.,torch.zeros_like(x,dtype=torch.long))
assert torch.isfinite(loss);loss.backward();assert not torch.isfinite(x.grad).all()
ratio_records.append(dict(case='true_fp32_gradient_overflow_not_hidden'))
caught=False
try:
    x=torch.full((1,1),710.,device=device,requires_grad=True)
    weighted_policy_loss(torch.zeros_like(x),x,torch.ones_like(x),torch.ones_like(x),torch.ones_like(x),.2,1.,torch.zeros_like(x,dtype=torch.long))
except ValueError as e:
    caught='ratio_dtype' in str(e) and 'torch.float64' in str(e)
assert caught
ratio_records.append(dict(case='fp64_raw_ratio_overflow_rejected'))
from search_r1.baselines.runtime_audit import stats
raw_ratio=torch.tensor([100.],device=device,dtype=torch.float32).double().exp()
assert stats(raw_ratio)['finite']==1 and stats(raw_ratio)['maximum']>torch.finfo(torch.float32).max
ratio_records.append(dict(case='audit_raw_fp64_ratio_finite_not_clipped',maximum=stats(raw_ratio)['maximum']))
dist.barrier()
for orig in (False,True):
    for offload in (False,True):
        torch.manual_seed(42)
        base=Mixed();reference=copy.deepcopy(base).double()
        reference(torch.ones(2,16,dtype=torch.float64)).sum().backward()
        oracle=torch.linalg.vector_norm(torch.cat([p.grad.flatten() for p in reference.parameters()]))
        args=dict(device_id=device,use_orig_params=orig,cpu_offload=CPUOffload(offload_params=offload))
        base.shard=FSDP(base.shard,**args)
        base.replica=FSDP(base.replica,sharding_strategy=ShardingStrategy.NO_SHARD,**args)
        model=FSDP(base,**args)
        model(torch.ones(2,16,device=device)).sum().backward()
        params=[p for p in model.parameters() if p.grad is not None]
        saved=[p.grad.clone() for p in params]
        native=model.clip_grad_norm_(1.)
        native_grads=[p.grad.clone() for p in params]
        for p,g in zip(params,saved):p.grad.copy_(g)
        actual=clip_grad_norm64(model,1.)
        torch.testing.assert_close(actual.cpu(),oracle,rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(actual.float().cpu(),native.float().cpu(),rtol=1e-5,atol=1e-5)
        for p,g in zip(params,native_grads):torch.testing.assert_close(p.grad,g,rtol=1e-5,atol=1e-6)
        records.append(dict(case='normal_native_and_fullvector_oracle',orig=orig,offload=offload,norm=float(actual)))
        # 512 global coordinates:256sharded plus256replicated/tied once.
        for p in params:p.grad.fill_(2e20)
        expected=torch.tensor(2e20,dtype=torch.float32).double()*512**.5
        high=clip_grad_norm64(model,1.)
        torch.testing.assert_close(high.cpu(),expected,rtol=1e-12,atol=1e-12)
        for p in params:
            expected_element=(torch.tensor(2e20,dtype=torch.float32)*(1/(expected+1e-6)).float()).item()
            torch.testing.assert_close(p.grad,torch.full_like(p.grad,expected_element),rtol=1e-5,atol=1e-6)
        records.append(dict(case='finite_gradient_fp32_square_overflow',orig=orig,offload=offload,norm=float(high)))
        # One rank corrupts a replicated gradient. Every rank must reject.
        for p in params:p.grad.fill_(1.)
        if rank==0:
            handle=next(h for h in model._all_handles if not h.uses_sharded_strategy)
            p=next(p for p in (handle.flat_param._params if orig else [handle.flat_param]) if p.grad is not None)
            p.grad.flatten()[0]=float('nan')
        caught=False
        try:clip_grad_norm64(model,1.)
        except ValueError as e:
            caught='refused on all ranks' in str(e)
        all_caught=torch.tensor(int(caught),device=device)
        dist.all_reduce(all_caught)
        assert all_caught.item()==8
        records.append(dict(case='one_rank_nonsharded_nan_collective_rejection',orig=orig,offload=offload))
        # Empty local sharded gradients on rank0; other shards and replicas count.
        for p in params:p.grad.fill_(1.)
        if rank==0:
            for h in model._all_handles:
                if h.uses_sharded_strategy:
                    for p in (h.flat_param._params if orig else [h.flat_param]):p.grad=None
        empty=clip_grad_norm64(model,1.)
        torch.testing.assert_close(empty.cpu(),torch.tensor((256*7/8+256)**.5,dtype=torch.float64),rtol=1e-12,atol=1e-12)
        records.append(dict(case='empty_one_rank_sharded_gradients',orig=orig,offload=offload,norm=float(empty)))
        del model,base,reference,params,saved,native_grads
        import gc
        gc.collect();torch.cuda.empty_cache();dist.barrier()
out=Path(os.environ['NUMERIC_FIXTURE_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
(out/f'rank-{rank}.json').write_text(json.dumps(dict(status='PASS',rank=rank,world=8,torch=torch.__version__,gradient_norm_file=gradient_norm.__file__,sys_prefix=sys.prefix,records=records,ratio_records=ratio_records),indent=2))
dist.barrier();dist.destroy_process_group()
print('REAL_FSDP_NUMERIC_FIXTURE_PASS',rank,flush=True)
