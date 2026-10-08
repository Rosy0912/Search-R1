"""Real eight-rank Torch FSDP oracle; synthetic gradients, no training updates."""
import copy
import datetime
import json
import os
import sys
from pathlib import Path
if sys.flags.optimize or os.environ.get('PYTHONOPTIMIZE'):
    raise RuntimeError('Optimized Python disables GPU fixture checks')
assert sys.prefix=='/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-baseline-aligned-em200-20261009/env'
EXPECTED_SOURCE='080ebbc46b89d61567cd33e1a5e2f1b40afd8b84'
repo=Path(os.environ['SEARCH_R1_REPO']).resolve()
assert repo==Path('/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-baseline-aligned-em200-20261009/repo')
sys.path.insert(0,str(repo))
import subprocess
assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==EXPECTED_SOURCE
assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip()
run=Path(os.environ['BAPO_RUN_ID'])
root=repo.parent
assert run.parent==root/'runs' and run.resolve()==run and run.is_dir()
assert run.name.startswith('hardware-aligned-em200-') and run.name.endswith('-'+os.environ['SLURM_JOB_ID'])
out=Path(os.environ['NUMERIC_FIXTURE_OUTPUT'])
assert out==run/'numeric-fixture' and out.resolve()==out
assert not (out/f'rank-{int(os.environ["RANK"])}.json').exists()
from search_r1.baselines import preflight
assert Path(preflight.__file__).resolve()==repo/'search_r1/baselines/preflight.py'
preflight.validate_resource_guard()
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, CPUOffload
from search_r1.baselines import gradient_norm
from search_r1.baselines.gradient_norm import clip_grad_norm64
from verl.trainer.ppo import core_algos
assert Path(gradient_norm.__file__).resolve()==repo/'search_r1/baselines/gradient_norm.py'
assert Path(core_algos.__file__).resolve()==repo/'verl/trainer/ppo/core_algos.py'


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
# Pure replicated FSDP and all-empty gradients exercise the new MAX collective.
for orig in (False,True):
    model=FSDP(Tied(),device_id=device,use_orig_params=orig,sharding_strategy=ShardingStrategy.NO_SHARD)
    model(torch.ones(2,16,device=device)).sum().backward()
    for p in model.parameters():p.grad=None
    empty=clip_grad_norm64(model,1.)
    assert empty.item()==0
    records.append(dict(case='pure_NO_SHARD_all_empty_gradients',orig=orig,norm=float(empty)))
    for p in model.parameters():p.grad=torch.ones_like(p)
    if rank==0:next(model.parameters()).grad.flatten()[0]=float('inf')
    caught=False
    try:clip_grad_norm64(model,1.)
    except ValueError as error:caught='refused on all ranks' in str(error)
    all_caught=torch.tensor(int(caught),device=device);dist.all_reduce(all_caught)
    assert all_caught.item()==8
    records.append(dict(case='pure_NO_SHARD_one_rank_inf_rejection',orig=orig))
    del model
    import gc
    gc.collect();torch.cuda.empty_cache();dist.barrier()
# Actual GPU KL values and backward derivatives against the clipped FP64 function.
points=torch.cat((torch.linspace(-12,5,400,device=device),torch.tensor([-10000.,-11.,4.,80.,89.,90.,99.,10000.],device=device)))
current=(-points).requires_grad_()
actual=core_algos.kl_penalty(current,torch.zeros_like(current),'low_var_kl')
# Analytic saturated branches are constant10; bounded FP64 exp avoids oracle overflow.
oracle_current=(-points.double()).clamp(min=-700).detach().requires_grad_()
x=-oracle_current
oracle=(x.exp()-x-1).clamp(-10,10)
torch.testing.assert_close(actual.double(),oracle,rtol=1e-5,atol=1e-6)
actual.sum().backward();oracle.sum().backward()
torch.testing.assert_close(current.grad.double(),oracle_current.grad,rtol=1e-5,atol=1e-6)
assert torch.isfinite(current.grad).all()
records.append(dict(case='gpu_low_var_kl_saturated_forward_backward_fp64_oracle',points=points.numel(),max_value_error=float((actual.double()-oracle).abs().max()),max_grad_error=float((current.grad.double()-oracle_current.grad).abs().max())))
for bad in (float('nan'),float('inf'),float('-inf')):
 caught=False
 try:core_algos.kl_penalty(torch.tensor([bad],device=device),torch.zeros(1,device=device),'low_var_kl')
 except ValueError as error:caught='Nonfinite' in str(error)
 assert caught
records.append(dict(case='gpu_low_var_kl_nonfinite_inputs_refused',count=3))

# Ordinary PPO GPU oracle: the finite negative-advantage branch is not Dual-Clipped
# and log-ratio30 is not silently clamped to20. No real actor update is performed.
old=torch.zeros((2,4),device=device)
current=torch.tensor([[0.,.1,1.,30.],[-.2,.5,2.,-1.]],device=device,requires_grad=True)
adv=torch.tensor([[1.,-1.,1.,-1.],[-2.,2.,-3.,3.]],device=device)
mask=torch.tensor([[1.,1.,1.,1.],[1.,1.,1.,0.]],device=device)
loss,_,_=core_algos.compute_policy_loss(old,current,adv,mask,.2,log_ratio_clip=None)
x=current.detach().double().requires_grad_();ratio=x.exp()
expected=(torch.maximum(-adv.double()*ratio,-adv.double()*ratio.clamp(.8,1.2))*mask.double()).sum()/mask.sum()
loss.backward();expected.backward()
torch.testing.assert_close(loss.double(),expected,rtol=2e-6,atol=1e-6)
torch.testing.assert_close(current.grad.double(),x.grad,rtol=2e-6,atol=1e-6)
assert current.grad[0,3]>1e10 and torch.isfinite(current.grad).all()
for bad in (float('nan'),float('inf'),89.):
    caught=False
    try:core_algos.compute_policy_loss(torch.zeros((1,1),device=device),torch.tensor([[bad]],device=device),torch.ones((1,1),device=device),torch.ones((1,1),device=device),.2,None)
    except ValueError:caught=True
    assert caught
records.append(dict(case='ordinary_PPO_GPU_value_gradient_FP64_oracle_no_dualclip_no_ratio20_clamp',loss=float(loss),large_negative_advantage_grad=float(current.grad[0,3]),overflow_nonfinite_rejected=True))

assert out.resolve()==out and run.resolve()==run
out.mkdir(exist_ok=True)
assert out==Path(os.environ['BAPO_RUN_ID'])/'numeric-fixture'
with (out/f'rank-{rank}.json').open('x') as f:
    json.dump(dict(status='PASS',scope='ACTUAL_8RANK_SYNTHETIC_GPU_ORACLES_NOT_TRAINING',rank=rank,world=8,torch=torch.__version__,gradient_norm_file=gradient_norm.__file__,sys_prefix=sys.prefix,records=records,source_commit=EXPECTED_SOURCE,core_algos_file=core_algos.__file__),f,indent=2)
dist.barrier();dist.destroy_process_group()
print('REAL_FSDP_NUMERIC_FIXTURE_PASS',rank,flush=True)
