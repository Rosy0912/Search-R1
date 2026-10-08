"""Actual eight-rank guarded actor/FSDP fixture; no research-model training."""
import json, os
from pathlib import Path
from types import MethodType
from unittest.mock import patch
# Match the actual Ray actor: one selected physical GPU visible, local CUDA0.
rank=int(os.environ['LOCAL_RANK'])
visible=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
assert len(visible)==8 and 0 <= rank < 8
os.environ['CUDA_VISIBLE_DEVICES']=visible[rank]
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from omegaconf import OmegaConf
from tensordict import TensorDict
from verl import DataProto
from verl.workers.actor.dp_actor import DataParallelPPOActor
from search_r1.baselines.policy_safety_runtime import PolicySafetyRefusal
from search_r1.baselines.policy_safety import dual_clip_terms
from verl.utils.fsdp_utils import offload_fsdp_param_and_grad, load_fsdp_param_and_grad, offload_fsdp_optimizer, load_fsdp_optimizer
torch.cuda.set_device(0)
dist.init_process_group('nccl');assert dist.get_world_size()==8
root=Path(os.environ['NUMERIC_FIXTURE_OUTPUT']);root.mkdir(parents=True,exist_ok=True)
records=[]

for dtype in (torch.float16,torch.bfloat16,torch.float32):
 x=torch.tensor([[800.,-800.,0.]],device='cuda',dtype=dtype,requires_grad=True)
 a=torch.tensor([[-1.,1.,0.]],device='cuda',dtype=dtype)
 terms=dual_clip_terms(torch.zeros_like(x),x,a,torch.ones_like(x))
 terms.sum().backward();assert torch.isfinite(terms).all() and torch.isfinite(x.grad).all()
 records.append(dict(case='bounded_extreme',dtype=str(dtype),loss=terms.tolist(),gradient=x.grad.tolist()))

# Import-time rank audit is disabled only for this explicitly synthetic fixture;
# safety evidence and actual production actor/optimizer are exercised unchanged.
import search_r1.baselines.tree_rank_audit as audit
audit.enabled=lambda:False

def setup(case, enabled=True, lr=1e-6):
 torch.manual_seed(73)
 module=torch.nn.Linear(1,1,bias=False).cuda();module.weight.data.fill_(.01)
 module=FSDP(module,use_orig_params=False,device_id=0,
             mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,reduce_dtype=torch.float32,buffer_dtype=torch.float32))
 actor=DataParallelPPOActor.__new__(DataParallelPPOActor)
 actor.actor_module=module;actor.actor_optimizer=torch.optim.AdamW(module.parameters(),lr=lr);actor.use_ulysses_sp=False
 actor.config=OmegaConf.create(dict(ppo_mini_batch_size=48,ppo_micro_batch_size=2,ppo_epochs=1,
  state_masking=True,use_kl_loss=False,use_dynamic_bsz=False,baseline_method='tree_grpo',
  entropy_coeff=0.,clip_ratio=.2,grad_clip=1.,recovery_max_grad_norm=1000.,
  policy_safety=dict(enabled=enabled,dual_clip=3.,kl_log_bound=30.,monitor_rows_per_rank=32,
    target_old_policy_kl=10.,negative_tail_limit=1.,initial_old_logprob_max_abs_delta=1e-6)))
 def forward(self,micro_batch,temperature):
  x=self.actor_module(micro_batch['input_ids'][:,:1].to(torch.bfloat16)).float().expand(-1,7)
  return x*0,x
 actor._forward_micro_batch=MethodType(forward,actor)
 rows=480;features=(torch.arange(rows,device='cuda')%5+1).long()[:,None]
 batch=TensorDict(dict(responses=torch.ones(rows,7,device='cuda',dtype=torch.long),
  input_ids=torch.cat((features,torch.ones(rows,7,device='cuda',dtype=torch.long)),1),
  attention_mask=torch.ones(rows,8,device='cuda'),position_ids=torch.arange(8,device='cuda').expand(rows,-1),
  loss_mask=torch.ones(rows,7,device='cuda'),old_log_probs=torch.zeros(rows,7,device='cuda'),
  advantages=torch.ones(rows,7,device='cuda'),
  policy_weights=torch.full((rows,7),1/7,device='cuda')),batch_size=[rows],device=torch.device('cuda',0))
 with torch.no_grad():
  old=torch.cat([forward(actor,b,1.)[1] for b in batch.split(2)])
 batch['old_log_probs']=old
 os.environ['SEARCH_R1_TREE_RANK_AUDIT_DIR']=str(root/case)
 return actor,DataProto(batch=batch,meta_info=dict(temperature=1.,baseline_outer_step=111))

def run(case,enabled=True,modify=None,lr=1e-6):
 actor,data=setup(case,enabled,lr)
 if modify is not None:modify(actor,data)
 # Exercise actual CPU offload and reloading before the monitored actor call.
 offload_fsdp_param_and_grad(actor.actor_module,offload_grad=True);offload_fsdp_optimizer(actor.actor_optimizer)
 load_fsdp_param_and_grad(actor.actor_module,0,load_grad=True);load_fsdp_optimizer(actor.actor_optimizer,0)
 torch.manual_seed(91);before_rng=torch.get_rng_state().clone();before_cuda_rng=torch.cuda.get_rng_state().clone()
 refused=False;metrics=None
 try:metrics=actor.update_policy(data)
 except PolicySafetyRefusal:refused=True
 committed=getattr(getattr(actor,'_policy_safety',None),'committed',0)
 populated_state_reloaded=False
 if metrics is not None:
  saved=[{key:value.detach().cpu().clone() for key,value in state.items() if isinstance(value,torch.Tensor)}
         for state in actor.actor_optimizer.state.values()]
  assert saved and all(float(state['step'])==10 for state in saved)
  offload_fsdp_param_and_grad(actor.actor_module,offload_grad=True);offload_fsdp_optimizer(actor.actor_optimizer)
  assert all(value.device.type=='cpu' for state in actor.actor_optimizer.state.values()
             for value in state.values() if isinstance(value,torch.Tensor))
  load_fsdp_param_and_grad(actor.actor_module,0,load_grad=True);load_fsdp_optimizer(actor.actor_optimizer,0)
  for original,state in zip(saved,actor.actor_optimizer.state.values()):
   for key,value in original.items():assert torch.equal(value,state[key].detach().cpu())
  populated_state_reloaded=True
 with FSDP.summon_full_params(actor.actor_module):
  value=next(actor.actor_module.parameters()).detach().float().cpu().clone()
 same_rng=torch.equal(before_rng,torch.get_rng_state()) and torch.equal(before_cuda_rng,torch.cuda.get_rng_state())
 decision=torch.tensor([int(refused),committed],device='cuda');lo=decision.clone();hi=decision.clone()
 dist.all_reduce(lo,op=dist.ReduceOp.MIN);dist.all_reduce(hi,op=dist.ReduceOp.MAX);assert torch.equal(lo,hi)
 record=dict(case=case,refused=refused,committed=committed,rng_preserved=same_rng,
             populated_adam_offload_reload_equal=populated_state_reloaded,
             actual_optimizer_steps=None if metrics is None else metrics['actor/optimizer_steps'],parameter=value.tolist())
 assert same_rng;records.append(record)
 return value,record

baseline,off=run('off',False)
guarded,on=run('on');assert torch.equal(baseline,guarded) and on['actual_optimizer_steps']==10 and on['committed']==10

def pad(a,d):
 d.batch['attention_mask'][:,-1]=0;d.batch['policy_weights'][:,-1]=0
 d.batch['old_log_probs'][:,-1]=float('nan')
_,padding=run('padding',modify=pad);assert not padding['refused'] and padding['committed']==10

def bad(a,d):
 if rank==3:d.batch['old_log_probs'][0,0]=float('nan')
_,pre=run('one_rank_invalid',modify=bad);assert pre['refused'] and pre['committed']==0

def post(a,d):a.config.policy_safety.target_old_policy_kl=1e-12
_,brake=run('post_write_brake',modify=post,lr=.01);assert brake['refused'] and brake['committed']==1

def mismatch(a,d):
 if rank==3:a.config.policy_safety.monitor_rows_per_rank=30
_,startup=run('startup_mismatch',modify=mismatch);assert startup['refused'] and startup['committed']==0
(root/f'rank-{rank}.json').write_text(json.dumps(dict(status='PASS',rank=rank,world=8,
 scope='Synthetic actual actor FSDP/offload/NCCL; not3B or calibrated thresholds',selected_visible_gpu=os.environ['CUDA_VISIBLE_DEVICES'],cuda_local_device=torch.cuda.current_device(),records=records),allow_nan=False,indent=2))
dist.barrier();dist.destroy_process_group()
