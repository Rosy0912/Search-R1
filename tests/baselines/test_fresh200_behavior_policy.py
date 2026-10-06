"""Behavior measure identity through real raw-token/context fixture, CPU only."""
import pytest
import torch
from test_fresh200_bapo_sampling_chain import fixture
from search_r1.baselines.behavior_policy import bind_actual_behavior, validate_behavior_tensors
from search_r1.baselines.fresh200_contract import verify_sampling_policy_binding
from search_r1.baselines.policy_safety import NonfiniteSamplingPolicy


def data_with_backend_gap(gap=.5):
    controller,manager,_,gen,qa=fixture()
    data,_=controller.generate(manager,gen,qa,{})
    data.batch['old_log_probs']=data.batch['sampling_log_probs']-gap
    data.batch['loss_mask']=data.batch['sampling_mask'].clone()
    from search_r1.llm_agent.sampling_tokens import audit_batch
    data.meta_info['sampling_context_audit']=audit_batch(data)
    return data


def test_large_finite_backend_gap_does_not_replace_actual_measure():
    data=data_with_backend_gap();hf=data.batch['old_log_probs'].clone();native=data.batch['sampling_log_probs'].clone()
    bind_actual_behavior(data)
    verify_sampling_policy_binding(data,{})
    m=data.batch['sampling_mask'].bool()
    assert torch.equal(data.batch['frozen_hf_log_probs'],hf)
    assert torch.equal(data.batch['sampling_log_probs'],native)
    assert torch.equal(data.batch['old_log_probs'],native)
    # At unchanged HF, the true behavior ratio is exp(-.5), not an invented 1.
    ratio=(hf-data.batch['old_log_probs']).exp()
    assert torch.allclose(ratio[m],torch.full_like(ratio[m],torch.exp(torch.tensor(-.5))))
    assert data.batch['old_log_probs'].data_ptr()!=data.batch['sampling_log_probs'].data_ptr()


def test_hf_fallback_and_rebinding_are_rejected():
    data=data_with_backend_gap();bind_actual_behavior(data)
    with pytest.raises(ValueError,match='cannot be repeated'):bind_actual_behavior(data)
    data.batch['old_log_probs']=data.batch['frozen_hf_log_probs'].clone()
    with pytest.raises(ValueError,match='denominator differs'):validate_behavior_tensors(data)


def test_context_mutation_remains_structural_not_numeric_skip():
    data=data_with_backend_gap();bind_actual_behavior(data)
    data.batch['input_ids'][0,-2]+=1
    with pytest.raises(ValueError):validate_behavior_tensors(data)


def test_unchanged_hf_nonfinite_is_typed_numerical_rejection():
    data=data_with_backend_gap();data.batch['old_log_probs'][0,0]=float('nan');bind_actual_behavior(data)
    with pytest.raises(NonfiniteSamplingPolicy):verify_sampling_policy_binding(data,{})
    assert torch.isfinite(data.batch['old_log_probs'][data.batch['sampling_mask'].bool()]).all()


def test_missing_role_cannot_fall_back_to_hf():
    data=data_with_backend_gap()
    with pytest.raises(ValueError,match='role missing'):validate_behavior_tensors(data)


def test_hf_drift_ignores_observation_nans_and_preserves_sequence_weights():
    from search_r1.baselines.behavior_policy import hf_drift_summary
    hf=torch.tensor([[-1.,float('nan')],[-2.,-3.]])
    behavior=torch.tensor([[-1.5,float('nan')],[-2.,-3.]])
    current=torch.tensor([[-1.2,float('nan')],[-2.4,-3.4]])
    mask=torch.tensor([[1,0],[1,1]])
    weights=torch.tensor([[1.,0.],[.5,.5]])
    r=hf_drift_summary(hf,behavior,current,mask,weights)
    assert r['finite'] and r['baseline_vs_behavior_max_abs']==pytest.approx(.5)
    assert r['current_vs_frozen_hf_sequence_mean_abs']==pytest.approx(.3)


def test_hf_drift_nonfinite_valid_or_derived_stays_json_serializable():
    import json
    from search_r1.baselines.behavior_policy import hf_drift_summary
    for hf,current in [(float('nan'),-1.),(1e308,-1e308)]:
        x=torch.tensor([[hf]],dtype=torch.float64);y=torch.tensor([[current]],dtype=torch.float64)
        r=hf_drift_summary(x,y,y,torch.ones(1,1),torch.ones(1,1))
        assert not r['finite'] and r['current_vs_frozen_hf_sequence_mean_abs'] is None
        json.dumps(r,allow_nan=False)
