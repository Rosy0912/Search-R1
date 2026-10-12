"""Strict joint PPO range checks against the original formula in FP64."""
import pytest
import torch
from search_r1.baselines.objectives import weighted_policy_loss


def call(delta, advantage, weight=.002, denominator=32., width=1):
    current=torch.full((1,width),delta/width,dtype=torch.float32,requires_grad=True)
    old=torch.zeros_like(current)
    adv=torch.full_like(current,advantage)
    weights=torch.full_like(current,weight/width)
    mask=torch.ones_like(current)
    ids=torch.zeros_like(current,dtype=torch.long)
    loss,clip,kl=weighted_policy_loss(old,current,adv,mask,weights,.2,denominator,ids)
    return current,loss,clip,kl


@pytest.mark.parametrize('advantage',[1.,0.,-1e-6])
def test_fp32_ratio_overflow_preserves_finite_value_and_gradient(advantage):
    current,loss,_,_=call(100.,advantage,width=500)
    oracle=current.detach().double().requires_grad_()
    adv=torch.full_like(current,advantage).double()
    # Production preserves native joint summation; only exponentiation widens.
    # Match that rounded value while retaining the sum's derivative in the oracle.
    summed=oracle.sum(dim=1,keepdim=True)
    summed=summed+(current.detach().sum(dim=1,keepdim=True).double()-summed.detach())
    ratio=summed.exp().expand_as(oracle)
    expected=(torch.maximum(-adv*ratio,-adv*ratio.clamp(.8,1.2))*torch.full_like(current,.002/500).double()).sum()/32
    expected.backward();loss.backward()
    assert loss.dtype==torch.float64
    assert torch.isfinite(current.grad).all()
    torch.testing.assert_close(loss,expected,rtol=1e-5,atol=1e-12)
    torch.testing.assert_close(current.grad,oracle.grad.float(),rtol=1e-5,atol=1e-12)
    if advantage>=0:assert torch.count_nonzero(current.grad)==0


def test_real_native_gradient_overflow_is_not_hidden():
    current,loss,_,_=call(100.,-1.,width=500)
    assert torch.isfinite(loss)
    loss.backward()
    assert not torch.isfinite(current.grad).all()


def test_fp64_ratio_overflow_still_rejected_with_failure_diagnostic():
    with pytest.raises(ValueError,match='ratio_dtype.*torch.float64'):
        call(710.,1.)


def test_normal_joint_values_and_clip_boundary_gradients_match_fp64_formula():
    values=torch.tensor([-.5,0.,.5],dtype=torch.float32)
    bounds=torch.log(torch.tensor([.8,1.2],dtype=torch.float32))
    values=torch.cat([values,bounds,torch.nextafter(bounds,torch.full_like(bounds,float('inf'))),torch.nextafter(bounds,torch.full_like(bounds,float('-inf')))])
    for advantage in [-1.,0.,1.]:
        current=values[:,None].clone().requires_grad_();old=torch.zeros_like(current)
        adv=torch.full_like(current,advantage);weight=torch.full_like(current,.125)
        loss,_,_=weighted_policy_loss(old,current,adv,torch.ones_like(current),weight,.2,32.,torch.zeros_like(current,dtype=torch.long))
        oracle=current.detach().double().requires_grad_();r=oracle.exp()
        expected=(torch.maximum(-adv.double()*r,-adv.double()*r.clamp(.8,1.2))*weight.double()).sum()/32
        expected.backward();loss.backward()
        torch.testing.assert_close(loss,expected,rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(current.grad,oracle.grad.float(),rtol=1e-6,atol=1e-12)


def test_masked_nonfinite_values_do_not_enter_ratio():
    x=torch.tensor([[0.,float('nan')]],requires_grad=True)
    loss,_,_=weighted_policy_loss(torch.zeros_like(x),x,torch.ones_like(x),torch.tensor([[1.,0.]]),torch.tensor([[1.,0.]]),.2,1.,torch.tensor([[0,-1]]))
    loss.backward()
    torch.testing.assert_close(x.grad,torch.tensor([[-1.,0.]]))
