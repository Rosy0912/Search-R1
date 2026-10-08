import sys,json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
root=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-3/work/bapo')
sys.path.insert(0,str(root))
import torch
from tensordict import TensorDict
from omegaconf import OmegaConf
from verl import DataProto
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
actor=DataParallelPPOActor.__new__(DataParallelPPOActor)
model=torch.nn.Linear(1,1,bias=False);optimizer=torch.optim.SGD(model.parameters(),lr=.1)
actor.actor_module=model;actor.actor_optimizer=optimizer;actor.gradient_accumulation=2
actor.config=OmegaConf.create(dict(ppo_mini_batch_size=2,ppo_micro_batch_size=1,ppo_epochs=1,state_masking=True,use_kl_loss=False,use_dynamic_bsz=False,baseline_method='bapo',clip_ratio=.2,entropy_coeff=0))
mask=torch.tensor([[1.,0.],[1.,1.]])
def forward(micro_batch,temperature):
 logp=model.weight.reshape(1,1).expand_as(micro_batch['old_log_probs'])
 return logp*0,torch.where(micro_batch['loss_mask'].bool(),logp,torch.full_like(logp,float('nan')))
actor._forward_micro_batch=forward
def step():
 norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();return norm
actor._optimizer_step=step
before=model.weight.detach().clone()
tensors=dict(responses=torch.zeros(2,2).long(),input_ids=torch.ones(2,3).long(),attention_mask=torch.ones(2,3).long(),position_ids=torch.zeros(2,3).long(),old_log_probs=torch.tensor([[0.,float('nan')],[0.,0.]]),advantages=torch.tensor([[1.,0.],[-1.,-1.]]),loss_mask=mask)
data=DataProto.from_dict(tensors,meta_info={'temperature':1.})
with patch.object(TensorDict,'cuda',lambda self,*a,**k:self): metrics=actor.update_policy(data)
assert metrics['bapo/old_logprob_nonfinite']==metrics['bapo/new_logprob_nonfinite']==0
assert metrics['bapo/loss_tokens']==3 and metrics['actor/optimizer_steps']==1
assert metrics['bapo/grad_norm_max']>=metrics['bapo/grad_norm_min']>=0
json.dumps(dict(step=1,metrics=metrics),default=lambda x:x.item(),allow_nan=False)
t=RayPPOTrainer.__new__(RayPPOTrainer);t.baseline_controller=SimpleNamespace(method='bapo')
tensors.update(info_mask=torch.tensor([[1,1,1],[1,0,0]]),step_ids=torch.tensor([[0,-1],[0,-1]]))
_,audit=t._create_loss_mask(DataProto.from_dict(tensors),{})
assert audit['bapo/mask_on_observation_tokens']==1
assert audit['bapo/model_tokens_missing_loss']==1
print('Production telemetry CPU probe passed: actual actor micro/optimizer loop, masked NaNs excluded, scalar stats and strict JSON, independent observation/missing-model mask violations detected.')
