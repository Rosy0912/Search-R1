"""Real actor regression: microbatch changes must not rescale reported paper losses."""
from types import MethodType
from unittest.mock import patch
import numpy as np
import torch
from tensordict import TensorDict
from omegaconf import OmegaConf
from verl import DataProto
from verl.workers.actor.dp_actor import DataParallelPPOActor
from search_r1.baselines.objectives import attach_policy_weights


def run(micro, method='gigpo'):
    model=torch.nn.Linear(1,1,bias=False)
    with torch.no_grad():model.weight.zero_()
    actor=DataParallelPPOActor.__new__(DataParallelPPOActor)
    actor.actor_module=model
    actor.actor_optimizer=torch.optim.SGD(model.parameters(),lr=.1)
    actor.config=OmegaConf.create(dict(ppo_mini_batch_size=2,ppo_micro_batch_size=micro,ppo_epochs=1,
        state_masking=True,use_kl_loss=True,use_dynamic_bsz=False,baseline_method=method,
        entropy_coeff=0.,clip_ratio=.2,grad_clip=100.,kl_loss_type='kl',kl_loss_coef=.2))
    def forward(self,micro_batch,temperature):
        x=self.actor_module(torch.ones(len(micro_batch),1))
        return x*0+3,x
    actor._forward_micro_batch=MethodType(forward,actor)
    data=DataProto.from_dict(dict(responses=torch.zeros(2,1).long(),input_ids=torch.ones(2,2).long(),
        attention_mask=torch.ones(2,2).long(),info_mask=torch.ones(2,2).long(),position_ids=torch.zeros(2,2).long(),
        loss_mask=torch.ones(2,1),old_log_probs=torch.zeros(2,1),ref_log_prob=torch.full((2,1),-.1),
        advantages=torch.ones(2,1),step_ids=torch.zeros(2,1).long()),non_tensors={'uid':np.asarray(['q','q'])},meta_info={'temperature':1.})
    if method == 'tree_grpo':
        del data.batch['step_ids']
    attach_policy_weights(data,method)
    with patch.object(TensorDict,'cuda',lambda self,*a,**k:self):metrics=actor.update_policy(data)
    return metrics,model.weight.detach()


def test_paper_loss_metrics_and_update_invariant_to_microbatch():
    small,sw=run(1);large,lw=run(2)
    torch.testing.assert_close(sw,lw)
    for key in ('actor/pg_loss','actor/entropy_loss','actor/kl_loss','actor/pg_clipfrac','actor/ppo_kl'):
        np.testing.assert_allclose(small[key],large[key],atol=1e-7)
    np.testing.assert_allclose(small['actor/pg_loss'],[-1.])
    np.testing.assert_allclose(small['actor/entropy_loss'],[3.])
    np.testing.assert_allclose(small['actor/kl_loss'],[.1])


def test_tree_actor_updates_without_optional_step_ids():
    small, sw = run(1, 'tree_grpo')
    large, lw = run(2, 'tree_grpo')
    assert torch.isfinite(sw).all() and sw.abs().sum().item() > 0
    torch.testing.assert_close(sw, lw)
    for key in ('actor/pg_loss', 'actor/entropy_loss', 'actor/kl_loss', 'actor/pg_clipfrac', 'actor/ppo_kl'):
        np.testing.assert_allclose(small[key], large[key], atol=1e-7)
    np.testing.assert_allclose(small['actor/pg_loss'], [-1.])
