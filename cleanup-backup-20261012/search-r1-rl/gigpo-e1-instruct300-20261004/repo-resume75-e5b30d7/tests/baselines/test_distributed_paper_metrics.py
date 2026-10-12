"""Actual two-rank Gloo/DDP actor logs, including DataProto's rank-zero metadata."""
import os
from pathlib import Path
import tempfile
from types import MethodType
from unittest.mock import patch
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from tensordict import TensorDict
from verl import DataProto
from search_r1.baselines.objectives import attach_policy_weights
from search_r1.baselines.rollout import select_rows
from verl.workers.actor.dp_actor import DataParallelPPOActor

KEYS = ('actor/pg_loss', 'actor/entropy_loss', 'actor/kl_loss',
        'actor/pg_clipfrac', 'actor/ppo_kl')


def full_data(case):
    masks = {
        'unequal': [[1,0,0], [1,1,1], [1,1,0], [1,0,0]],
        'empty_leaf': [[0,0,0], [1,1,1], [1,1,0], [1,0,0]],
        'empty_rank': [[0,0,0], [1,1,1], [0,0,0], [1,0,0]],
    }
    mask = torch.tensor(masks[case], dtype=torch.float32)
    prefix = torch.ones(4,1)
    data = DataProto.from_dict(dict(
        responses=torch.zeros(4,3,dtype=torch.long),
        input_ids=torch.tensor([[1,1,1,1],[2,1,1,1],[3,1,1,1],[4,1,1,1]]),
        attention_mask=torch.cat((prefix,mask),1).long(),
        info_mask=torch.cat((prefix,mask),1).long(),
        position_ids=torch.zeros(4,4,dtype=torch.long), loss_mask=mask,
        old_log_probs=torch.tensor([[.1]*3,[-.2]*3,[.7]*3,[.6]*3]),
        ref_log_prob=torch.tensor([[.05]*3,[.2]*3,[.1]*3,[-.1]*3]),
        advantages=torch.tensor([[1.]*3,[-2.]*3,[3.]*3,[-.5]*3])),
        non_tensors={'uid':np.asarray(['q']*4)}, meta_info={'temperature':1.})
    return attach_policy_weights(data,'tree_grpo')


def actor_run(data, micro, distributed=False, reduce_logs=True):
    model = torch.nn.Linear(1,1,bias=False)
    with torch.no_grad(): model.weight.fill_(.15)
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model)
    actor = DataParallelPPOActor.__new__(DataParallelPPOActor)
    actor.actor_module = model
    actor.actor_optimizer = torch.optim.SGD(model.parameters(),lr=.02)
    actor.config = OmegaConf.create(dict(ppo_mini_batch_size=len(data),
        ppo_micro_batch_size=micro, ppo_epochs=1, state_masking=True,
        use_kl_loss=True, use_dynamic_bsz=False, baseline_method='tree_grpo',
        entropy_coeff=.03, clip_ratio=.2, grad_clip=100., kl_loss_type='kl', kl_loss_coef=.2))
    def forward(self,micro_batch,temperature):
        feature=micro_batch['input_ids'][:, :1].float()
        logp=self.actor_module(feature).expand(-1,3)
        return logp*.2+1., logp
    actor._forward_micro_batch=MethodType(forward,actor)
    gradients=[]
    original_step=actor._optimizer_step
    def step():
        gradients.append(next(model.parameters()).grad.detach().clone())
        return original_step()
    actor._optimizer_step=step
    from contextlib import nullcontext
    log_patch = nullcontext() if reduce_logs else patch(
        'search_r1.baselines.objectives.reduce_paper_metrics',lambda *args:None)
    with patch.object(TensorDict,'cuda',lambda self,*args,**kwargs:self), log_patch:
        metrics=actor.update_policy(data)
    return metrics,next(model.parameters()).detach().clone(),gradients[0]


def worker(rank,rendezvous,reference):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2)
    try:
        for case in ('unequal','empty_leaf','empty_rank'):
            for micro in (1,2):
                data=full_data(case)
                local=select_rows(data,[rank,rank+2])
                enabled,weight,gradient=actor_run(local,micro,True,True)
                disabled,old_weight,old_gradient=actor_run(local,micro,True,False)
                # Log-only reduction must leave every rank bitwise identical.
                torch.testing.assert_close(weight,old_weight,atol=0,rtol=0)
                torch.testing.assert_close(gradient,old_gradient,atol=0,rtol=0)
                expected,ref_weight,ref_gradient=reference[case]
                torch.testing.assert_close(weight,ref_weight,atol=1e-7,rtol=1e-6)
                torch.testing.assert_close(gradient,ref_gradient,atol=1e-6,rtol=1e-6)
                gathered=[None]*2
                dist.all_gather_object(gathered,DataProto(meta_info={'metrics':enabled}))
                combined=DataProto.concat(gathered).meta_info['metrics']
                for key in KEYS:
                    np.testing.assert_allclose(combined[key],expected[key],atol=1e-6,rtol=1e-6)
                    np.testing.assert_allclose(gathered[0].meta_info['metrics'][key],
                                               gathered[1].meta_info['metrics'][key],atol=0,rtol=0)
                assert combined['actor/optimizer_steps']==1
                if case=='empty_rank' and rank==0:
                    assert disabled['actor/pg_loss']==[0.]
                    assert abs(enabled['actor/pg_loss'][0])>0
    finally:
        dist.destroy_process_group()


def test_distributed_tree_metrics_concat_global_reference_and_unchanged_gradients():
    # Full four-leaf reference uses the real actor without any process group.
    reference={case:actor_run(full_data(case),4) for case in
               ('unequal','empty_leaf','empty_rank')}
    with tempfile.TemporaryDirectory() as temp:
        mp.spawn(worker,args=(os.path.join(temp,'gloo-init'),reference),nprocs=2,join=True)
