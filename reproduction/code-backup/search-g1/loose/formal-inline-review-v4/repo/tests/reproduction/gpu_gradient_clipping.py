"""Actual four-rank pinned FSDP clipping comparison; only isolated tiny models."""
import os,json
from pathlib import Path
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP,CPUOffload,ShardingStrategy
from search_g1.gradient_clipping import clip_global_l2
rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);dist.init_process_group('nccl')
results=[]
try:
 for offload,orig,no_shard in [(False,False,False),(True,False,False),(False,True,False),(False,False,True)]:
  torch.manual_seed(3)
  kwargs=dict(device_id=rank,use_orig_params=orig,cpu_offload=CPUOffload(offload_params=offload),sharding_strategy=ShardingStrategy.NO_SHARD if no_shard else ShardingStrategy.FULL_SHARD)
  inner=FSDP(torch.nn.Linear(4,4,bias=False),**kwargs)
  model=FSDP(torch.nn.Sequential(inner,torch.nn.Linear(4,2,bias=False)),**kwargs)
  model(torch.ones(2,4,device=rank)).sum().backward()
  params=[p for p in model.parameters() if p.grad is not None]
  originals=[p.grad.clone() for p in params]
  expected=model.clip_grad_norm_(1.)
  expected_clipped=[p.grad.clone() for p in params]
  for p,g in zip(params,originals):p.grad.copy_(g)
  actual=clip_global_l2(model,1.,chunk_size=3,diagnostic=False)
  torch.testing.assert_close(actual.float().cpu(),expected.float().cpu(),rtol=1e-5,atol=1e-6)
  for p,g in zip(params,expected_clipped):torch.testing.assert_close(p.grad,g,rtol=1e-5,atol=1e-6)
  for p in params:p.grad.fill_(1e20)
  huge=[p.grad.clone() for p in params]
  upstream=model.clip_grad_norm_(1.)
  assert not torch.isfinite(upstream),'Expected FP32 overflow reproduction'
  for p,g in zip(params,huge):p.grad.copy_(g)
  robust=clip_global_l2(model,1.,chunk_size=3,diagnostic=False)
  assert torch.isfinite(robust) and robust>1e20
  norm_after=clip_global_l2(model,100.,chunk_size=3,diagnostic=False)
  torch.testing.assert_close(norm_after.cpu(),torch.tensor(1.,dtype=torch.float64),rtol=1e-5,atol=1e-6)
  for p in params:p.grad.fill_(2.)
  if rank==0:params[0].grad.view(-1)[0]=float('nan')
  snapshots=[p.grad.clone() for p in params]
  rejected=False
  try:clip_global_l2(model,1.,diagnostic=False)
  except RuntimeError:rejected=True
  assert rejected
  for p,g in zip(params,snapshots):torch.testing.assert_close(p.grad,g,equal_nan=True)
  for p in params:p.grad=torch.full_like(p,2.)
  if rank==0:
   p=params[0];p.grad=torch.full((p.numel()*2,),2.,device=p.grad.device)[::2].reshape(p.shape)
  snapshots=[p.grad.clone() for p in params]
  rejected=False
  try:clip_global_l2(model,1.,diagnostic=False)
  except RuntimeError:rejected=True
  assert rejected
  for p,g in zip(params,snapshots):torch.testing.assert_close(p.grad,g)
  results.append(dict(offload=offload,orig_params=orig,no_shard=no_shard,moderate_matches=True,huge_raw_norm=float(robust),collective_nonfinite_rejected=True))
  del model,inner;torch.cuda.empty_cache();dist.barrier()
 Path(os.environ['CLIP_TEST_OUT']).joinpath(f'rank{rank}.json').write_text(json.dumps(dict(rank=rank,accepted=True,results=results)))
finally:dist.destroy_process_group()
