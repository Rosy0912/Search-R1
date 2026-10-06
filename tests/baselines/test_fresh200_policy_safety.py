"""CPU numerical oracle and real Gloo rejection agreement."""
import json
import math
from types import SimpleNamespace
import pytest
import torch
from search_r1.baselines.policy_safety import (dual_clip_surrogate, finite_inputs, monitor,
    consensus, agree_document, PartialPolicyUpdate)


def observation(delta, adv=None, mask=None, weights=None, **kwargs):
    delta = torch.tensor(delta, dtype=torch.float64)
    adv = torch.full_like(delta, -1.) if adv is None else torch.tensor(adv, dtype=torch.float64)
    mask = torch.ones_like(delta) if mask is None else torch.tensor(mask, dtype=torch.float64)
    weights = mask/mask.sum(-1,keepdim=True).clamp_min(1) if weights is None else torch.tensor(weights,dtype=torch.float64)
    return monitor(torch.zeros_like(delta), delta, adv, mask, weights,
                   c=3., kl_log_bound=30., target_kl=kwargs.pop('target_kl',1e9),
                   negative_tail_limit=kwargs.pop('negative_tail_limit',1.), **kwargs)


@pytest.mark.parametrize('advantage', [-2.,0.,2.])
@pytest.mark.parametrize('delta', [-100., -30., math.log(.7),0., math.log(1.4),math.log(4),100.])
def test_dualclip_value_and_gradient(delta,advantage):
    current=torch.tensor([[delta]],dtype=torch.float64,requires_grad=True)
    old=torch.zeros_like(current);a=torch.full_like(current,advantage);mask=torch.ones_like(current)
    actual=dual_clip_surrogate(current-old,a,mask,.2,3.,20.).sum()
    x=current.detach().clone().requires_grad_();r=x.clamp_max(20).exp()
    expected=torch.maximum(-a*r,-a*r.clamp(.8,1.2))
    expected=torch.where(a<0,torch.minimum(expected,-3*a),expected).sum()
    torch.testing.assert_close(actual,expected,atol=1e-100,rtol=1e-12)
    torch.testing.assert_close(torch.autograd.grad(actual,current)[0],torch.autograd.grad(expected,x)[0],atol=1e-100,rtol=1e-12)


def test_tail_uses_all_negative_weight_not_saturated_only_or_zero_adv():
    report=observation([[math.log(4),0,0],[math.log(4),0,0]],
        adv=[[-1,-1,0],[-1,0,0]],weights=[[.1,.9,100],[1,100,100]])
    assert report['negative_effective_weight_mass']==pytest.approx(2)
    assert report['tail_effective_weight_mass']==pytest.approx(1.1)
    assert report['negative_tail_weight_fraction']==pytest.approx(.55)
    zero=observation([[10]],adv=[[0]])
    assert zero['negative_tail_weight_fraction']==0


def test_k3_is_actual_seqmean_weighted_old_measure():
    report=observation([[.1,.2],[.3,99]],mask=[[1,1],[1,0]])
    expected=((math.expm1(.1)-.1)/2+(math.expm1(.2)-.2)/2+math.expm1(.3)-.3)/2
    assert report['sampled_old_policy_kl']==pytest.approx(expected)
    assert report['kl_estimate_complete']


def test_masked_nonfinite_and_overbound_never_complete_estimate():
    assert observation([[0,float('nan')]],mask=[[1,0]])['allowed']
    invalid=observation([[float('nan')]])
    assert not invalid['allowed'] and invalid['sampled_old_policy_kl'] is None
    over=observation([[31]])
    assert not over['allowed'] and not over['kl_estimate_complete'] and over['overflow_count']==1
    assert observation([[-1e300]])['kl_estimate_complete']
    assert not observation([[0]],mask=[[0]])['allowed']


def test_finite_before_subtraction_and_double_overflow_rejection():
    x=torch.tensor([[1e308]],dtype=torch.float64);one=torch.ones_like(x)
    with pytest.raises(ValueError,match='subtraction'):
        finite_inputs(-x,x,one,one,one)
    with pytest.raises(ValueError,match='input'):
        finite_inputs(x,torch.full_like(x,float('inf')),one,one,one)


def gloo_worker(rank,rendezvous,output):
    torch.distributed.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2)
    results=[]
    for invalid_kind in ('nonfinite','overflow','tail','kl','good'):
        delta=[[0.]]
        if rank==1:
            delta=[{'nonfinite':[float('nan')],'overflow':[31.],'tail':[math.log(4)],'kl':[.5],'good':[0.]}[invalid_kind]]
        report=observation(delta,negative_tail_limit=.1,target_kl=.01)
        results.append(report)
        assert report['allowed']==(invalid_kind=='good')
    one=torch.ones((1,1),dtype=torch.float64)
    for invalid in (False,True):
        new=one.clone()
        if invalid and rank==1:new.fill_(float('inf'))
        caught=False
        try: finite_inputs(one,new,one,one,one)
        except ValueError: caught=True
        assert caught==invalid
    assert consensus(rank==0) is False
    assert agree_document({'enabled':bool(rank)}) is False
    assert agree_document({'enabled':True}) is True
    torch.distributed.destroy_process_group()
    with open(output+str(rank)+'.json','w') as f:json.dump(results,f)


def test_real_gloo_same_rejection_path(tmp_path):
    torch.multiprocessing.spawn(gloo_worker,args=(str(tmp_path/'rendezvous'),str(tmp_path/'rank')),nprocs=2,join=True)
    assert json.loads((tmp_path/'rank0.json').read_text())==json.loads((tmp_path/'rank1.json').read_text())


def test_partial_write_retains_prior_adam_and_zeroes_unapplied_gradient():
    pytest.importorskip("tensordict")
    pytest.importorskip("omegaconf")
    from verl.workers.actor.dp_actor import DataParallelPPOActor
    p=torch.nn.Parameter(torch.tensor(1.));opt=torch.optim.AdamW([p],lr=.1)
    p.square().backward();opt.step();prior=p.detach().clone();opt.zero_grad()
    p.square().backward()
    actor=SimpleNamespace(actor_optimizer=opt,_safety_writes=1,_safety_expected_writes=20,
                         _safety_actual_outer=1,_safety_rollout_identity='frozen-1',_safety_processed=2,_safety_skips=[],_safety_attempt_id=1)
    with pytest.raises(PartialPolicyUpdate) as caught:
        DataParallelPPOActor._reject_safety(actor,'negative_advantage_tail',{'allowed':False})
    assert caught.value.event['status']=='ABORTED' and caught.value.event['optimizer_steps']==1
    assert caught.value.event['update_complete'] is False and p.grad is None
    torch.testing.assert_close(p,prior)
    assert opt.state[p]['step'].item()==1


@pytest.mark.parametrize('outer,expected',[(49,20),(50,10)])
@pytest.mark.parametrize('reject_post',[False,True,'nonfinite_state','pre_mini','pre_rollout','all_minis','half_mini','commit_fault','shadow_step_fault','shadow_prepare_fault'])
def test_actor_schedule_and_partial_adam_transaction_cpu(monkeypatch,outer,expected,reject_post):
    """CPU actor transaction oracle; production admissions are explicitly stubbed.

    This checks actor execution, not CUDA probability/calibration admission. Real
    communication/rejection agreement is independently exercised above with Gloo.
    """
    pytest.importorskip('tensordict');pytest.importorskip('omegaconf')
    from omegaconf import OmegaConf
    from tensordict import TensorDictBase
    from verl import DataProto
    from verl.workers.actor.dp_actor import DataParallelPPOActor
    import search_r1.baselines.policy_safety as safety_module
    import search_r1.baselines.fresh200_contract as contract
    import search_r1.baselines.distillation as reduction
    import search_r1.baselines.objectives as objectives
    import search_r1.llm_agent.sampling_tokens as sampling_tokens
    from search_r1.baselines.behavior_policy import bind_actual_behavior, validate_behavior_tensors
    monkeypatch.setattr(torch.cuda,'current_device',lambda:0)
    monkeypatch.setattr(TensorDictBase,'cuda',lambda self,*a,**kw:self)
    monkeypatch.setattr(torch.distributed,'is_initialized',lambda:True)
    monkeypatch.setattr(torch.distributed,'get_world_size',lambda *a:8)
    monkeypatch.setattr(safety_module,'distributed',lambda *a:False)
    monkeypatch.setattr(reduction,'global_token_count',lambda mask:(mask.sum(),1))
    monkeypatch.setattr(objectives,'row_denominator',lambda rows,device:torch.tensor(float(rows),device=device))
    monkeypatch.setattr(contract,'verify_fixed_policy_safety_artifact',lambda config:(1.,'cpu-test-admission'))
    # The native context/GPU admission boundary is synthetic in this transaction
    # oracle. Exercise the real new denominator binder/transport validator so
    # downstream faults cannot be hidden by an obsolete HF-only fixture.
    monkeypatch.setattr(sampling_tokens,'audit_batch',lambda data:dict(probability_role='stochastic_sampling_policy'))
    def verify_fixture_behavior(data,config):
        validate_behavior_tensors(data)
        return 'cpu-test-sampling'
    monkeypatch.setattr(contract,'verify_sampling_policy_binding',verify_fixture_behavior)
    safety=dict(enabled=True,dual_clip=3,kl_log_bound=30.,target_old_policy_kl=1.,negative_tail_limit=1.)
    config=OmegaConf.create(dict(ppo_mini_batch_size=16,ppo_micro_batch_size=2,ppo_epochs=1,
        state_masking=True,use_kl_loss=True,kl_loss_coef=0.,kl_loss_type='low_var_kl',use_dynamic_bsz=False,baseline_method='bapo',
        loss_reduction='sequence_mean',grad_clip=1.,clip_ratio=.2,entropy_coeff=0.,policy_safety=safety))
    one=torch.ones(320,1,dtype=torch.float64)
    policy_mask=one.clone();policy_mask[-1]=0
    old=-one.clone();old[-1]=float('nan')
    data=DataProto.from_dict(dict(responses=one.long(),input_ids=one.long(),attention_mask=one.long(),
        position_ids=one.long(),loss_mask=policy_mask,old_log_probs=old,
        sampling_log_probs=old-.03,sampling_mask=policy_mask,
        advantages=-one,ref_log_prob=torch.full_like(one,float('inf'))),
        meta_info=dict(temperature=1.,actual_outer=outer,attempt_id=1,rollout_identity='cpu-test-rollout',policy_snapshot_id='cpu-test-snapshot'))
    bind_actual_behavior(data)
    torch.testing.assert_close(data.batch['frozen_hf_log_probs'],old,equal_nan=True)
    torch.testing.assert_close(data.batch['old_log_probs'],data.batch['sampling_log_probs'],equal_nan=True)
    model=torch.nn.Linear(1,1,bias=False).double();model.weight.data.zero_()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    if reject_post=='shadow_step_fault':
        original_adam=torch.optim.AdamW.step;shadow_steps=[0]
        def shadow_failure(optimizer_,*args,**kwargs):
            shadow_steps[0]+=1
            if shadow_steps[0]==2:raise RuntimeError('fixture shadow step allocation/kernel failure')
            return original_adam(optimizer_,*args,**kwargs)
        monkeypatch.setattr(torch.optim.AdamW,'step',shadow_failure)
    if reject_post=='shadow_prepare_fault':
        original_init=torch.optim.AdamW.__init__;shadow_builds=[0]
        def shadow_build_failure(optimizer_,*args,**kwargs):
            shadow_builds[0]+=1
            if shadow_builds[0]==2:raise RuntimeError('fixture shadow preparation failure')
            return original_init(optimizer_,*args,**kwargs)
        monkeypatch.setattr(torch.optim.AdamW,'__init__',shadow_build_failure)
    forward_count=[0]
    def forward(micro_batch,temperature):
        current=micro_batch['frozen_hf_log_probs']+model.weight.reshape(())
        if torch.is_grad_enabled():
            forward_count[0]+=1
            if reject_post=='half_mini' and forward_count[0]==2:current=current*float('nan')
        entropy=torch.where(micro_batch['loss_mask'].bool(),torch.zeros_like(current),float('nan'))
        return entropy,current
    actor=SimpleNamespace(config=config,actor_module=model,actor_optimizer=optimizer,
        _forward_micro_batch=forward,ulysses_sequence_parallel_size=1)
    def step():
        norm=DataParallelPPOActor._optimizer_step(actor)
        if reject_post=='commit_fault':
            from search_r1.baselines.optimizer_transaction import OptimizerCommitFault
            raise OptimizerCommitFault(dict(parameter_copies=1,state_entries=1,local_commit_complete=True))
        if reject_post=='nonfinite_state':optimizer.state[model.weight]['exp_avg'].fill_(float('nan'))
        return norm
    actor._optimizer_step=step
    captured=[]
    actor._safety_evidence_sink=captured.append
    if reject_post in (True,'pre_mini','pre_rollout','all_minis'):
        monitor_count=[0]
        original=DataParallelPPOActor._monitor_frozen_policy
        def reject(self,frozen,safety,stage,monitor_batch=None):
            report=original(self,frozen,safety,stage,monitor_batch)
            if stage=='before_adam_full_minibatch':monitor_count[0]+=1
            if ((reject_post is True and stage=='after_adam') or
                (reject_post=='pre_rollout' and stage=='before_adam') or
                (reject_post=='pre_mini' and stage=='before_adam_full_minibatch' and monitor_count[0]==1) or
                (reject_post=='all_minis' and stage=='before_adam_full_minibatch')):
                report.update(allowed=False,reasons=['cpu-test-brake'])
            return report
        monkeypatch.setattr(DataParallelPPOActor,'_monitor_frozen_policy',reject)
    result=DataParallelPPOActor.update_policy(actor,data)
    event=result['fresh200_update_state']
    if reject_post=='nonfinite_state':
        assert event['status']=='INFRASTRUCTURE_FAULT' and event['optimizer_steps']==1
        assert event['confirmed_previous_optimizer_steps']==1
        assert event['failed_commit_progress']['count_already_recorded'] is True
        assert event['failed_commit_progress']['stage']=='post_commit_state_nonfinite'
        assert not event['update_complete'] and event['postabort'] is True
        assert actor._safety_writes==1 and optimizer.state[model.weight]['step'].item()==1
        return
    if reject_post in ('shadow_step_fault','shadow_prepare_fault'):
        assert event['status']=='INFRASTRUCTURE_FAULT' and event['optimizer_steps']==1
        assert event['confirmed_previous_optimizer_steps']==1 and event['failed_commit_progress']['stage']==('shadow_step' if reject_post=='shadow_step_fault' else 'shadow_prepare')
        assert event['failed_commit_progress']['local_commit_complete'] is False and not event['update_complete']
        assert optimizer.state[model.weight]['step'].item()==1 and model.weight.grad is None
        return
    if reject_post=='commit_fault':
        assert event['status']=='INFRASTRUCTURE_FAULT' and event['optimizer_steps']==1
        assert event['confirmed_previous_optimizer_steps']==0 and event['postabort'] is True
        assert not event['update_complete'] and event['failed_commit_progress']['local_commit_complete']
        assert optimizer.state[model.weight]['step'].item()==1 and model.weight.grad is None
        return
    assert event['expected_optimizer_steps']==expected
    expected_writes=1 if reject_post is True else (0 if reject_post in ('pre_rollout','all_minis') else expected-1 if reject_post in ('pre_mini','half_mini') else expected)
    assert event['optimizer_steps']==expected_writes
    expected_status='ABORTED' if reject_post is True else 'SKIPPED' if expected_writes==0 else 'COMPLETE_WITH_SKIPS' if expected_writes<expected else 'COMPLETE'
    assert event['status']==expected_status
    assert event['update_complete']==(expected_status in ('COMPLETE','COMPLETE_WITH_SKIPS'))
    assert event['processed_opportunities']==event['optimizer_steps']+event['skipped_opportunities']
    assert event['processed_opportunities']+event['unprocessed_opportunities']==expected
    from search_r1.baselines.fresh200_contract import validate_rank_updates
    ranks=[dict(event,rank=i) for i in range(8)]
    assert validate_rank_updates(ranks,outer)==event['update_complete']
    if expected_writes==0:
        assert not optimizer.state and model.weight.grad is None
        return
    assert optimizer.state[model.weight]['step'].item()==event['optimizer_steps']
    assert model.weight.grad is None
    assert all(r['monitoring_scope'] in ('full_frozen_rollout','full_current_minibatch') for r in result['fresh200_policy_monitor'])
    assert any(r['monitoring_scope']=='full_current_minibatch' for r in result['fresh200_policy_monitor'])
    assert len(captured)==len(result['fresh200_policy_monitor'])
    for payload in captured:
        assert payload['old_log_probs'].shape==payload['current_log_probs'].shape==payload['weights'].shape
        assert payload['mask'].shape[0]==payload['report']['monitored_local_rows']
        assert payload['current_log_probs'].device.type=='cpu'
        selected=payload['mask'].bool()
        assert torch.all(payload['old_log_probs'][selected]==-1.03)
        assert payload['report']['denominator_role']=='actual_native_stochastic_sampling_policy'
        assert payload['report']['hf_diagnostic_drift']['baseline_vs_behavior_max_abs']==pytest.approx(.03)


def test_full_minibatch_brake_catches_rollout_average_dilution():
    delta=[[.5]]*16+[[0.]]*304
    assert observation(delta,target_kl=.01)['allowed']
    assert not observation(delta[:16],target_kl=.01)['allowed']
    tails=[[math.log(4)]]*16+[[0.]]*304
    assert observation(tails,negative_tail_limit=.1)['allowed']
    assert not observation(tails[:16],negative_tail_limit=.1)['allowed']


def test_enabled_weighted_sequence_measure_is_micro_partition_invariant():
    pytest.importorskip('tensordict');pytest.importorskip('omegaconf')
    from search_r1.baselines.objectives import weighted_policy_loss
    current=torch.tensor([[-30.,float('nan'),float('nan')],[math.log(4),0.,-.5],[float('nan')]*3],
                         dtype=torch.float64,requires_grad=True)
    old=torch.zeros_like(current)
    mask=torch.tensor([[1.,0.,0.],[1.,1.,1.],[0.,0.,0.]],dtype=torch.float64)
    weights=mask/mask.sum(-1,keepdim=True).clamp_min(1)
    advantages=torch.tensor([[1.,1.,1.],[-1.,-1.,-1.],[0.,0.,0.]],dtype=torch.float64)
    args=dict(cliprange=.2,denominator=torch.tensor(3.),log_ratio_clip=20.,dual_clip=3.)
    full=weighted_policy_loss(old,current,advantages,mask,weights,**args)[0]
    parts=sum(weighted_policy_loss(old[i:i+1],current[i:i+1],advantages[i:i+1],mask[i:i+1],weights[i:i+1],**args)[0] for i in range(3))
    torch.testing.assert_close(full,parts)
    expected=(-math.exp(-30)+(3+1+.8)/3)/3
    assert full.item()==pytest.approx(expected)
    gf=torch.autograd.grad(full,current,retain_graph=True)[0]
    gs=torch.autograd.grad(parts,current)[0]
    torch.testing.assert_close(gf,gs)
    assert torch.isfinite(gf).all() and gf[0,0].item()==pytest.approx(-math.exp(-30)/3,abs=1e-100,rel=1e-12)


@pytest.mark.parametrize('advantage',[-60000.,60000.])
def test_half_inputs_do_not_overflow_dualclip_intermediates(advantage):
    delta=torch.tensor([[30.]],dtype=torch.float16,requires_grad=True)
    a=torch.full_like(delta,advantage);mask=torch.ones_like(delta)
    loss=dual_clip_surrogate(delta,a,mask).sum()
    assert torch.isfinite(loss)
    assert loss.item()==pytest.approx(180000. if advantage<0 else -72000.)
    gradient=torch.autograd.grad(loss,delta)[0]
    assert torch.isfinite(gradient).all() and gradient.item()==0.
