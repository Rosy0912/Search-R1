"""Common sequence measure and complete internal-state regression oracles."""
import copy
import numpy as np
import pytest
import torch
from verl import DataProto
from verl.trainer.ppo.core_algos import kl_penalty
from search_r1.baselines.objectives import attach_policy_weights, weighted_policy_loss
from search_r1.baselines.recovery import worker_training_state, load_worker_training_state, checkpoint_pause_identity
from search_r1.baselines.instruct300 import equal_state


def test_sequence_mean_keeps_joint_actions_and_masks():
    mask = torch.tensor([[1,0,1,1],[0,0,0,0]],dtype=torch.float32)
    data = DataProto.from_dict(dict(responses=torch.zeros(2,4).long(),attention_mask=mask.long(),
        info_mask=mask.long(),step_ids=torch.tensor([[0,-1,1,1],[-1,-1,-1,-1]])),non_tensors={'uid':np.array(['q','q'])})
    attach_policy_weights(data,'gigpo','sequence_mean')
    torch.testing.assert_close(data.batch['policy_weights'],mask/3)
    current = torch.zeros(2,4,requires_grad=True)
    loss,_,_=weighted_policy_loss(torch.zeros_like(current),current,torch.ones_like(current),mask,
        data.batch['policy_weights'],.2,2.,data.batch['step_ids'])
    loss.backward()
    # Second action's joint ratio sums two tokens; both inherit its whole 2/3 row mass.
    torch.testing.assert_close(current.grad,torch.tensor([[-1/6,0,-1/3,-1/3],[0,0,0,0]]))


def test_pg_kl_global128_dp8_micro2_matches_unsplit_with_empty_rank():
    gen=torch.Generator().manual_seed(37)
    mask=(torch.rand(128,8,generator=gen)>.25).float();mask[:16]=0
    ids=torch.where(mask.bool(),torch.arange(8)//2,-1).expand(128,8).clone()
    data=DataProto.from_dict(dict(responses=torch.zeros(128,8).long(),attention_mask=mask.long(),
        info_mask=mask.long(),step_ids=ids),non_tensors={'uid':np.array(['q'+str(i//5) for i in range(128)])})
    attach_policy_weights(data,'gigpo','sequence_mean');weights=data.batch['policy_weights']
    old=torch.randn(128,8,generator=gen)*.2
    reference=old-.07
    advantage=torch.randn(128,4,generator=gen).repeat_interleave(2,dim=1)
    features=torch.randn(128,8,generator=gen)*.02
    def objective(theta,start,stop,denom):
        current=old[start:stop]+theta*features[start:stop]
        pg,_,_=weighted_policy_loss(old[start:stop],current,advantage[start:stop],mask[start:stop],
            weights[start:stop],.2,denom,ids[start:stop])
        kl=(kl_penalty(current,reference[start:stop],kl_penalty='low_var_kl')*weights[start:stop]).sum()/denom
        return pg+.001*kl
    full=torch.tensor(.9,requires_grad=True);full_loss=objective(full,0,128,128);full_loss.backward()
    split=torch.tensor(.9,requires_grad=True);split_loss=0
    for rank in range(8):
        for row in range(rank*16,(rank+1)*16,2):
            value=objective(split,row,row+2,16)/8
            value.backward();split_loss=split_loss+value.detach()
    torch.testing.assert_close(split_loss,full_loss.detach(),atol=1e-10,rtol=1e-6)
    torch.testing.assert_close(split.grad,full.grad,atol=1e-10,rtol=1e-6)


def test_absolute57_outer_scheduler_not20_inner_steps():
    p=torch.nn.Parameter(torch.tensor(1.))
    optimizer=torch.optim.AdamW([p],lr=1e-6)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda outer:min(1.,outer/57))
    original=p.detach().clone()
    for outer in range(60):
        assert optimizer.param_groups[0]['lr']==pytest.approx(1e-6*min(1.,outer/57))
        for _ in range(20):
            optimizer.zero_grad();p.square().backward();optimizer.step()
        if outer==0:torch.testing.assert_close(p,original,rtol=0,atol=0)
        scheduler.step()
    assert optimizer.state[p]['step']==1200
    assert scheduler.last_epoch==60
    assert p.item()!=original.item()


def fixture():
    module=torch.nn.Linear(2,1)
    module.register_buffer('marker',torch.tensor([3.]))
    for p in module.parameters():p._local_shard=p.detach().clone()+9
    opt=torch.optim.AdamW(module.parameters(),lr=.01)
    sch=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
    module(torch.ones(1,2)).sum().backward();opt.step();opt.zero_grad();sch.step()
    return module,opt,sch


def test_distinct_internal_shards_buffers_restore_and_identity():
    module,opt,sch=fixture()
    state=worker_training_state(module,opt,sch,0,1,1,complete_internal=True)
    before=checkpoint_pause_identity(module,opt,sch,0,complete_internal=True)
    for p in module.parameters():p._local_shard.add_(42)
    module.marker.add_(1)
    assert before!=checkpoint_pause_identity(module,opt,sch,0,complete_internal=True)
    assert load_worker_training_state(state,module,opt,sch,0,1)==1
    assert before==checkpoint_pause_identity(module,opt,sch,0,complete_internal=True)
    for name,p in module.named_parameters():torch.testing.assert_close(p._local_shard,state['local_shards'][name])


@pytest.mark.parametrize('field',['local_shards','buffers'])
def test_bad_internal_state_rejected_before_any_mutation(field):
    module,opt,sch=fixture();state=worker_training_state(module,opt,sch,0,1,1,complete_internal=True)
    state[field].clear();before=checkpoint_pause_identity(module,opt,sch,0,complete_internal=True)
    with pytest.raises(ValueError,match='coverage'):load_worker_training_state(state,module,opt,sch,0,1)
    assert before==checkpoint_pause_identity(module,opt,sch,0,complete_internal=True)


def test_recursive_rng_identity_handles_numpy_and_torch():
    from search_r1.baselines.recovery import rng_state
    state=rng_state();assert equal_state(state,copy.deepcopy(state))
    bad=copy.deepcopy(state);bad['torch'][0]^=1
    assert not equal_state(state,bad)
