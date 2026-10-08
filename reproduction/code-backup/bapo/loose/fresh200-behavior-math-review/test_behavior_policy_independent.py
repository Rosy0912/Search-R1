"""Independent CPU math/real-journal review; not native acceptance."""
import sys
from pathlib import Path
import pytest,torch
REPO=Path(__file__).resolve().parents[1]/'bapo-fresh200-behavior-policy-20261006'
sys.path[:0]=[str(REPO),str(REPO/'tests/baselines')]
from search_r1.baselines.behavior_policy import bind_actual_behavior,validate_behavior_tensors
from search_r1.baselines.policy_safety import NonfiniteSamplingPolicy,monitor,dual_clip_surrogate


def fixture():
    from test_fresh200_bapo_sampling_chain import fixture as raw_fixture
    c,m,w,g,q=raw_fixture();data,_=c.generate(m,g,q,{})
    data.batch['old_log_probs']=torch.full_like(data.batch['sampling_log_probs'],-.325)
    return data


def test_binding_preserves_actual_native_and_original_hf_independently():
    data=fixture();native=data.batch['sampling_log_probs'];hf=data.batch['old_log_probs'];hf_saved=hf.clone()
    bind_actual_behavior(data)
    assert validate_behavior_tensors(data)
    assert torch.equal(data.batch['frozen_hf_log_probs'],hf_saved)
    assert torch.equal(data.batch['old_log_probs'],native)
    assert data.batch['old_log_probs'].data_ptr()!=native.data_ptr()
    assert data.batch['frozen_hf_log_probs'].data_ptr()!=hf.data_ptr()
    with pytest.raises(ValueError,match='repeated'):bind_actual_behavior(data)


@pytest.mark.parametrize('failure',['both_transport','old_only','native_nan','greedy','context','frozen_hf_nan'])
def test_actual_journal_errors_are_structural_only_hf_nan_numerical(failure):
    data=fixture();bind_actual_behavior(data)
    if failure=='both_transport':
        data.batch['sampling_log_probs'][0,0]-=.1;data.batch['old_log_probs'][0,0]-=.1
    if failure=='old_only':data.batch['old_log_probs'][0,0]-=.1
    if failure=='native_nan':
        data.batch['sampling_log_probs'][0,0]=float('nan');data.batch['old_log_probs'][0,0]=float('nan')
    if failure=='greedy':data.meta_info['fresh200_evaluation']=True
    if failure=='context':data.meta_info['policy_snapshot_id']='different'
    if failure=='frozen_hf_nan':data.batch['frozen_hf_log_probs'][0,0]=float('nan')
    with pytest.raises(ValueError) as caught:validate_behavior_tensors(data)
    assert isinstance(caught.value,NonfiniteSamplingPolicy)==(failure=='frozen_hf_nan')


def test_binary_behavior_k3_is_not_zero_hf_self_drift():
    b=torch.tensor([[.8,.2]],dtype=torch.float64);h0=torch.tensor([[.6,.4]],dtype=torch.float64)
    mask=torch.ones_like(b);a=-torch.ones_like(b)
    report=monitor(b.log(),h0.log(),a,mask,b,c=3,kl_log_bound=30,target_kl=.2,negative_tail_limit=.2)
    expected=(b*(b.log()-h0.log())).sum().item()
    assert report['sampled_old_policy_kl']==pytest.approx(expected,abs=1e-14)
    assert not torch.equal((h0/b),torch.ones_like(b))
    theta=torch.tensor([.1,-.2],dtype=torch.float64,requires_grad=True)
    lp=torch.log_softmax(theta,0);delta=lp[0]-b[0,0].log()
    actual=dual_clip_surrogate(delta.reshape(1,1),torch.ones(1,1),torch.ones(1,1)).sum()
    ratio=(theta.softmax(0)[0]/b[0,0])
    direct=torch.maximum(-ratio,-ratio.clamp(.8,1.2))
    grad_actual=torch.autograd.grad(actual,theta,retain_graph=True)[0]
    grad_direct=torch.autograd.grad(direct,theta)[0]
    assert torch.allclose(actual,direct,atol=1e-14,rtol=0.) and torch.allclose(grad_actual,grad_direct,atol=1e-14,rtol=0.)


def test_behavior_tail_guard_can_reject_with_zero_hf_drift():
    b=torch.tensor([[.8,.2]],dtype=torch.float64);h0=torch.tensor([[.2,.8]],dtype=torch.float64)
    a=-torch.ones_like(b);mask=torch.ones_like(b);w=torch.tensor([[1.,3.]],dtype=torch.float64)
    report=monitor(b.log(),h0.log(),a,mask,w,c=3,kl_log_bound=30,target_kl=.2,negative_tail_limit=.2)
    assert report['negative_tail_weight_fraction']==.75 and not report['allowed']
    hf=monitor(h0.log(),h0.log(),a,mask,w,c=3,kl_log_bound=30,target_kl=.2,negative_tail_limit=.2)
    assert hf['negative_tail_weight_fraction']==0 and hf['sampled_old_policy_kl']==0
    loss=dual_clip_surrogate(h0.log()-b.log(),a,mask)
    assert loss.tolist()==[[.8,3.]]


def test_behavior_v2_matrix_replay_uses_native_not_hf_and_masks_nans():
    from search_r1.baselines.fresh200_native_evidence import replay_matrices
    payload=dict(old_log_probs=torch.tensor([[-2.,float('nan')],[-2.,-2.]],dtype=torch.float64),
        current_log_probs=torch.tensor([[-.8,float('nan')],[-2.,-2.]],dtype=torch.float64),
        sampling_log_probs=torch.tensor([[-2.,float('nan')],[-2.,-2.]],dtype=torch.float64),
        frozen_hf_log_probs=torch.tensor([[-2.5,float('nan')],[-2.5,-2.5]],dtype=torch.float64),
        advantages=torch.tensor([[-1.,float('nan')],[-1.,-1.]],dtype=torch.float64),
        mask=torch.tensor([[1.,0.],[1.,1.]],dtype=torch.float64),
        denominator_role='actual_native_stochastic_sampling_policy',
        hf_probability_role='unchanged_hf_recomputation_diagnostic')
    result=replay_matrices(payload,2)
    assert result['negative_tail_weight_fraction']==.5
    assert result['sampled_old_policy_kl']==pytest.approx((torch.expm1(torch.tensor(1.2,dtype=torch.float64)).item()-1.2)/2)
    assert result['effective_token_count']==3
    import copy
    for mutation in ('missing_hf','wrong_role','hf_as_old','nonfinite_hf_valid'):
        bad=copy.deepcopy(payload)
        if mutation=='missing_hf':del bad['frozen_hf_log_probs']
        if mutation=='wrong_role':bad['denominator_role']='old_hf'
        if mutation=='hf_as_old':bad['old_log_probs']=bad['frozen_hf_log_probs'].clone()
        if mutation=='nonfinite_hf_valid':bad['frozen_hf_log_probs'][0,0]=float('nan')
        with pytest.raises(ValueError):replay_matrices(bad,2)
