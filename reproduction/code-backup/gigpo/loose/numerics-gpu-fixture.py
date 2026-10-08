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
(out/f'rank-{rank}.json').write_text(json.dumps(dict(status='PASS',rank=rank,world=8,torch=torch.__version__,gradient_norm_file=gradient_norm.__file__,sys_prefix=sys.prefix,records=records),indent=2))
dist.barrier();dist.destroy_process_group()
print('REAL_FSDP_NUMERIC_FIXTURE_PASS',rank,flush=True)
