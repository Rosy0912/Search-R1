"""Real actor/DataProto accumulation with fixed128 mini and low-var KL."""
from types import MethodType
from unittest.mock import patch
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from verl import DataProto
from verl.workers.actor.dp_actor import DataParallelPPOActor
from search_r1.baselines.objectives import attach_policy_weights


def run(micro,rows=3072,mini=128):
    torch.manual_seed(31)
    model=torch.nn.Linear(1,1,bias=False);model.weight.data.fill_(.015)
    length=7
    mask=(torch.arange(length)[None,:]<((torch.arange(rows)%8)[:,None])).float()
    prefix=torch.ones(rows,1)
    feature=(torch.arange(rows)%5+1).float()[:,None]
    old=feature.expand(-1,length)*.014
    data=DataProto.from_dict(dict(responses=torch.zeros(rows,length,dtype=torch.long),
        input_ids=torch.cat((feature,torch.ones(rows,length)),1).long(),
        attention_mask=torch.cat((prefix,mask),1).long(),info_mask=torch.cat((prefix,mask),1).long(),
        position_ids=torch.zeros(rows,length+1,dtype=torch.long),loss_mask=mask,
        old_log_probs=old,ref_log_prob=old-.01,
        advantages=(torch.arange(rows)%7-3).float()[:,None].expand(-1,length)),
        non_tensors={'uid':np.asarray([str(i//6) for i in range(rows)])},meta_info={'temperature':1.})
    attach_policy_weights(data,'tree_grpo')
    actor=DataParallelPPOActor.__new__(DataParallelPPOActor)
    actor.actor_module=model;actor.actor_optimizer=torch.optim.SGD(model.parameters(),lr=1e-6)
    actor.config=OmegaConf.create(dict(ppo_mini_batch_size=mini,ppo_micro_batch_size=micro,
        ppo_epochs=1,state_masking=True,use_kl_loss=True,use_dynamic_bsz=False,
        baseline_method='tree_grpo',entropy_coeff=0.,clip_ratio=.2,grad_clip=1.,
        kl_loss_type='low_var_kl',kl_loss_coef=.001))
    def forward(self,micro_batch,temperature):
        logp=self.actor_module(micro_batch['input_ids'][:,:1].float()).expand(-1,length)
        return logp*0,logp
    actor._forward_micro_batch=MethodType(forward,actor)
    gradients=[];step=actor._optimizer_step
    def capture():
        gradients.append(model.weight.grad.detach().clone());return step()
    actor._optimizer_step=capture
    with patch.object(TensorDict,'cuda',lambda self,*a,**kw:self):metrics=actor.update_policy(data)
    assert metrics['actor/optimizer_steps']==rows//mini
    return metrics,model.weight.detach().clone(),torch.stack(gradients)


def test_actual_actor_fixed128_micro2_16_128_same_target_and24_steps():
    reference=run(128)
    for micro in (2,16):
        metrics,weight,gradients=run(micro)
        torch.testing.assert_close(weight,reference[1],rtol=1e-6,atol=1e-8)
        torch.testing.assert_close(gradients,reference[2],rtol=2e-6,atol=2e-7)
        for key in ('actor/pg_loss','actor/kl_loss'):
            np.testing.assert_allclose(metrics[key],reference[0][key],rtol=2e-6,atol=2e-7)


def test_actual_recovery_rank480_mini48_micro2_performs10_updates():
    reference=run(48,rows=480,mini=48)
    metrics,weight,gradients=run(2,rows=480,mini=48)
    assert metrics['actor/optimizer_steps']==10 and len(gradients)==10
    torch.testing.assert_close(weight,reference[1],rtol=1e-6,atol=1e-8)
    torch.testing.assert_close(gradients,reference[2],rtol=2e-6,atol=2e-7)
