import math
import torch
import pytest
from omegaconf import OmegaConf
from search_r1.baselines.objectives import weighted_policy_loss
from search_r1.baselines.full_state import resolve_warmup_steps
from verl.trainer.ppo.core_algos import compute_policy_loss
from verl.utils.torch_functional import get_constant_schedule_with_warmup

@pytest.mark.parametrize('delta', [-30., -20., 0., 10., 20., 21., 30.])
@pytest.mark.parametrize('advantage', [-1., 1.])
def test_weighted_matches_author_clamped_surrogate_and_backward(delta, advantage):
    old=torch.tensor([[-40.]],dtype=torch.float64)
    current=(old+delta).requires_grad_(); mask=torch.ones_like(old)
    result=weighted_policy_loss(old,current,torch.full_like(old,advantage),mask,mask,.2,torch.tensor(1.),log_ratio_clip=20.)[0]
    oracle_current=current.detach().clone().requires_grad_()
    ratio=(oracle_current-old).clamp(-20,20).exp()
    expected=torch.maximum(-advantage*ratio,-advantage*ratio.clamp(.8,1.2)).mean()
    assert torch.allclose(result,expected)
    assert torch.allclose(torch.autograd.grad(result,current)[0],torch.autograd.grad(expected,oracle_current)[0])
    backup=compute_policy_loss(old,current,torch.full_like(old,advantage),mask,.2,log_ratio_clip=20.)[0]
    assert torch.allclose(backup,expected)

def test_masked_nonfinite_and_micro_partition_keep_full_row_measure():
    old=torch.full((4,3),-40.,dtype=torch.float64)
    current=torch.tensor([[-10.,-35.,float('nan')],[-50.,-20.,-19.],[-40.,-39.,-38.],[-80.,-41.,-21.]],dtype=torch.float64,requires_grad=True)
    mask=torch.ones_like(old); mask[0,2]=0; mask[2]=0
    weights=mask/mask.sum(-1,keepdim=True).clamp_min(1)
    adv=torch.tensor([-1.,1.,-1.,-1.],dtype=torch.float64)[:,None].expand_as(old)
    full=weighted_policy_loss(old,current,adv,mask,weights,.2,torch.tensor(4.),log_ratio_clip=20.)[0]
    split=sum(weighted_policy_loss(old[i:i+1],current[i:i+1],adv[i:i+1],mask[i:i+1],weights[i:i+1],.2,torch.tensor(4.),log_ratio_clip=20.)[0] for i in range(4))
    assert torch.allclose(full,split)
    grad_full=torch.autograd.grad(full,current,retain_graph=True)[0]
    grad_split=torch.autograd.grad(split,current)[0]
    assert torch.allclose(grad_full,grad_split) and torch.isfinite(grad_full).all()

def test_unrelated_objectives_keep_existing_ratio():
    old=torch.tensor([[-40.]],dtype=torch.float64); current=torch.tensor([[-10.]],dtype=torch.float64,requires_grad=True); one=torch.ones_like(old)
    result=weighted_policy_loss(old,current,-one,one,one,.2,torch.tensor(1.))[0]
    assert result.item()==pytest.approx(math.exp(30))

def test_actual_config_and_scheduler_warmup114():
    config=OmegaConf.load('verl/trainer/config/ppo_bapo_e1_300.yaml')
    optim_cfg=config.actor_rollout_ref.actor.optim
    assert optim_cfg.num_warmup_steps==114 and config.trainer.total_training_steps==301
    assert resolve_warmup_steps(dict(num_warmup_steps=114,total_training_steps=301,lr_warmup_steps_ratio=.285))==114
    parameter=torch.nn.Parameter(torch.tensor(1.)); opt=torch.optim.AdamW([parameter],lr=1e-6)
    schedule=get_constant_schedule_with_warmup(opt,114)
    for outer in range(1,301):
        assert schedule.get_last_lr()[0]==pytest.approx(1e-6*min(outer-1,114)/114)
        # Twenty Adam calls, only one outer scheduler step.
        for _ in range(20):
            opt.zero_grad(); parameter.square().backward(); opt.step()
        schedule.step()
        assert schedule.last_epoch==outer
        assert schedule.get_last_lr()[0]==pytest.approx(1e-6*min(outer,114)/114)

@pytest.mark.parametrize('warmup', [57,85,None,True,114.0,'114'])
def test_gate_rejects_old_or_unbound_actual_scheduler_config(warmup):
    from search_r1.baselines.checkpoint_gate import validate_gate_config
    config=dict(trainer=dict(total_training_steps=301,e1_gate_enabled=True,acceptance_step=5),actor_rollout_ref=dict(actor=dict(optim=dict(num_warmup_steps=warmup))))
    with pytest.raises(ValueError): validate_gate_config(config)

def test_actual_gate_config_derives_horizon_and_warmup():
    from search_r1.baselines.checkpoint_gate import validate_gate_config
    config=OmegaConf.to_container(OmegaConf.load('verl/trainer/config/ppo_bapo_e1_300.yaml'),resolve=False)
    assert validate_gate_config(config)==(300,114)
