"""Real Gloo/DDP actor steps: unequal lengths, split questions and empty ranks."""
from pathlib import Path
import os
import sys
import tempfile
from types import MethodType
import unittest
from unittest.mock import patch
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from tensordict import TensorDict
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from verl import DataProto
from verl.workers.actor.dp_actor import DataParallelPPOActor
from search_r1.baselines.objectives import attach_policy_weights
from search_r1.baselines.rollout import select_rows


def worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=2)
    torch.cuda.current_device = lambda: rank  # CPU Gloo model/CUDA shim.
    try:
        for method in ('topd', 'tree_grpo', 'gigpo', 'cosearch'):
            for micro in (1, 2):
                model = torch.nn.parallel.DistributedDataParallel(torch.nn.Linear(1, 1, bias=False))
                with torch.no_grad(): model.module.weight.zero_()
                optimizer = torch.optim.SGD(model.parameters(), lr=.1)
                mask = torch.tensor([[1.,0.,0.], [1.,0.,0.], [1.,0.,0.], [1.,1.,1.]])
                tensors = dict(responses=torch.zeros(4,3).long(), input_ids=torch.ones(4,4).long(),
                    attention_mask=torch.cat((torch.ones(4,1),mask),1).long(),
                    info_mask=torch.cat((torch.ones(4,1),mask),1).long(), position_ids=torch.zeros(4,4).long(),
                    loss_mask=mask, old_log_probs=torch.zeros(4,3),
                    advantages=torch.tensor([[1.]*3,[1.]*3,[3.]*3,[3.]*3]), step_ids=torch.zeros(4,3).long())
                data = DataProto.from_dict(tensors,non_tensors={'uid':np.asarray(['a','a','b','b'])},meta_info={'temperature':1.})
                attach_policy_weights(data, method)
                # Every question spans ranks; each rank has one short a and one b.
                local = select_rows(data,[rank,rank+2])
                actor = DataParallelPPOActor.__new__(DataParallelPPOActor)
                actor.actor_module,actor.actor_optimizer = model,optimizer
                actor.config=OmegaConf.create(dict(ppo_mini_batch_size=2,ppo_micro_batch_size=micro,ppo_epochs=1,
                    state_masking=True,use_kl_loss=False,use_dynamic_bsz=False,baseline_method=method,
                    entropy_coeff=0.,clip_ratio=.2,grad_clip=100.))
                def forward(self,micro_batch,temperature):
                    logits=self.actor_module(torch.ones(len(micro_batch),1)).expand(-1,3)
                    return logits*0,logits
                actor._forward_micro_batch=MethodType(forward,actor)
                with patch.object(TensorDict,'cuda',lambda self,*args,**kwargs:self):
                    metrics=actor.update_policy(local)
                expected=.35 if method=='gigpo' else .2
                torch.testing.assert_close(model.module.weight,torch.tensor([[expected]]),atol=1e-6,rtol=0)
                assert metrics['actor/optimizer_steps']==1

        # Genuine main-actor DDP backward with rank0 entirely unsupervised.
        model=torch.nn.parallel.DistributedDataParallel(torch.nn.Linear(1,1,bias=False))
        with torch.no_grad(): model.module.weight.zero_()
        optimizer=torch.optim.AdamW(model.parameters(),lr=.1,weight_decay=.1)
        actor=DataParallelPPOActor.__new__(DataParallelPPOActor)
        actor.actor_module,actor.actor_optimizer=model,optimizer
        actor.config=OmegaConf.create(dict(ppo_mini_batch_size=2,ppo_micro_batch_size=1,ppo_epochs=1,
            state_masking=True,use_kl_loss=False,use_dynamic_bsz=False,baseline_method='topd',
            entropy_coeff=0.,clip_ratio=.2,grad_clip=100.))
        actor._forward_micro_batch=MethodType(forward,actor)
        local=select_rows(data,[rank,rank+2])
        local.batch['loss_mask']=torch.zeros_like(local.batch['loss_mask']) if rank==0 else local.batch['loss_mask']
        local.batch['policy_weights']=local.batch['loss_mask']/local.batch['loss_mask'].sum(-1,keepdim=True).clamp_min(1)
        with patch.object(TensorDict,'cuda',lambda self,*args,**kwargs:self): metrics=actor.update_policy(local)
        assert metrics['actor/optimizer_steps']==1
        torch.testing.assert_close(model.module.weight,torch.tensor([[.1]]),atol=1e-6,rtol=0)
        before=model.module.weight.detach().clone()
        state=optimizer.state[model.module.weight]['step'].clone()
        local.batch['loss_mask'].zero_();local.batch['policy_weights'].zero_()
        with patch.object(TensorDict,'cuda',lambda self,*args,**kwargs:self): metrics=actor.update_policy(local)
        assert metrics['actor/optimizer_steps']==0
        torch.testing.assert_close(model.module.weight,before,atol=0,rtol=0)
        torch.testing.assert_close(optimizer.state[model.module.weight]['step'],state,atol=0,rtol=0)
    finally:
        dist.destroy_process_group()


class PaperDDP(unittest.TestCase):
    def test_actor_micro_and_rank_invariance_and_no_unsupervised_decay(self):
        with tempfile.TemporaryDirectory() as temp:
            mp.spawn(worker,args=(os.path.join(temp,'gloo-init'),),nprocs=2,join=True)
