"""Real journal verifier into actual actor admission; CPU, no CUDA release."""
import pytest
import torch
from types import SimpleNamespace


def test_actual_probability_nonfinite_whole_attempt_skip(monkeypatch):
    actor,data,model,optimizer=_fixture(monkeypatch)
    data.batch['old_log_probs'][0,0]=float('nan')
    from verl.workers.actor.dp_actor import DataParallelPPOActor
    result=DataParallelPPOActor.update_policy(actor,data)
    event=result['fresh200_update_state']
    assert event['status']=='SKIPPED' and event['reason']=='nonfinite_sampling_policy'
    assert event['optimizer_steps']==event['processed_opportunities']==0
    assert event['unprocessed_opportunities']==20 and event['postabort'] is False
    assert event['attempt_complete'] is False and event['update_complete'] is False
    assert not optimizer.state and torch.equal(model.weight,torch.zeros_like(model.weight))


@pytest.mark.parametrize('failure',['finite_parity','identity_with_nan','loss_mask_with_nan','sampled_transport_nan','finite_parity_with_nan'])
def test_actual_binding_structural_error_still_fatal(monkeypatch,failure):
    actor,data,_,optimizer=_fixture(monkeypatch)
    if failure=='finite_parity':data.batch['old_log_probs'][0,0]+=.1
    elif failure=='sampled_transport_nan':data.batch['sampling_log_probs'][0,0]=float('nan')
    elif failure=='finite_parity_with_nan':
        data.batch['old_log_probs'][0,0]=float('nan');data.batch['old_log_probs'][1,0]+=.1
    else:
        data.batch['old_log_probs'][0,0]=float('nan')
        if failure=='identity_with_nan':data.meta_info['policy_snapshot_id']='different'
        else:data.batch['loss_mask'][0,0]=0
    from verl.workers.actor.dp_actor import DataParallelPPOActor
    with pytest.raises(RuntimeError,match='Sampling structural admission'):
        DataParallelPPOActor.update_policy(actor,data)
    assert not optimizer.state


def _fixture(monkeypatch):
    from test_fresh200_bapo_sampling_chain import fixture
    from search_r1.llm_agent.sampling_tokens import objects, audit_batch
    from verl import DataProto
    from omegaconf import OmegaConf
    import search_r1.baselines.policy_safety as safety
    import search_r1.baselines.fresh200_contract as contract
    controller,manager,_,_,_=fixture()
    ids=torch.tensor([[0,10]]*320);mask=torch.tensor([[0,1]]*320)
    gen=DataProto.from_dict(dict(input_ids=ids,attention_mask=mask,position_ids=torch.zeros_like(ids)),
        non_tensors={'fresh200_row_id':objects([f'row{i}' for i in range(320)])},
        meta_info=dict(preserve_sampling_tokens=True,policy_snapshot_id='snapshot'))
    data=manager.run_llm_loop(gen,ids)
    data.batch['old_log_probs']=data.batch['sampling_log_probs'].clone()
    data.batch['loss_mask']=data.batch['sampling_mask'].long()
    data.batch['advantages']=-torch.ones_like(data.batch['old_log_probs'])
    data.meta_info.update(actual_outer=1,attempt_id=1,rollout_identity='test-actual-context',temperature=1.)
    assert audit_batch(data)['context_verified']
    config=OmegaConf.create(dict(ppo_mini_batch_size=16,ppo_micro_batch_size=2,use_dynamic_bsz=False,
        loss_reduction='sequence_mean',baseline_method='bapo',state_masking=True,ppo_epochs=1,clip_ratio=.2,
        policy_safety=dict(enabled=True,dual_clip=3,kl_log_bound=30.,target_old_policy_kl=.2,negative_tail_limit=.2)))
    model=torch.nn.Linear(1,1,bias=False);model.weight.data.zero_()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    monkeypatch.setattr(torch.distributed,'is_initialized',lambda:True)
    monkeypatch.setattr(torch.distributed,'get_world_size',lambda *args:8)
    monkeypatch.setattr(safety,'distributed',lambda *args:False)
    monkeypatch.setattr(contract,'verify_fixed_policy_safety_artifact',lambda cfg:(.2,'cpu-artifact'))
    def unexpected_forward(*args,**kwargs):raise AssertionError('Rejected admission reached model forward')
    actor=SimpleNamespace(config=config,actor_module=model,actor_optimizer=optimizer,_forward_micro_batch=unexpected_forward)
    return actor,data,model,optimizer


@pytest.mark.parametrize('snapshot_conflict',[False,True])
def test_eight_rank_actual_verifier_one_numeric_rejection_consensus(monkeypatch,snapshot_conflict):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    import search_r1.baselines.policy_safety as safety
    from search_r1.llm_agent.sampling_trace import sha
    from search_r1.llm_agent.sampling_tokens import audit_batch
    from verl.workers.actor.dp_actor import DataParallelPPOActor
    fixtures=[_fixture(monkeypatch) for _ in range(8)]
    bad=fixtures[3][1];bad.batch['old_log_probs'][0,0]=float('nan')
    if snapshot_conflict:
        bad.meta_info['policy_snapshot_id']='different-valid-snapshot'
        for turns in bad.non_tensor_batch['fresh200_turns']:
            for turn in turns:
                turn['policy_snapshot_id']='different-valid-snapshot'
                native={k:v for k,v in turn.items() if k not in ('native_sha256','kept_token_count','environment_token_ids','turn_sha256')}
                turn['native_sha256']=sha(native)
                turn['turn_sha256']=sha({k:v for k,v in turn.items() if k!='turn_sha256'})
        bad.meta_info['sampling_context_audit']=audit_batch(bad)
    local=threading.local();barrier=threading.Barrier(8,timeout=20);slots={};lock=threading.Lock()
    monkeypatch.setattr(safety,'distributed',lambda *args:True)
    monkeypatch.setattr(safety,'collective_device',lambda *args:torch.device('cpu'))
    def share(value):
        index=local.index;local.index+=1
        with lock:slots.setdefault(index,{})[local.rank]=value.clone()
        barrier.wait();return [slots[index][i] for i in range(8)]
    def reduce(tensor,op=None,group=None):
        values=share(tensor);tensor.copy_(torch.stack(values).min(0).values);barrier.wait()
    def gather(output,tensor,group=None):
        values=share(tensor)
        for dst,src in zip(output,values):dst.copy_(src)
        barrier.wait()
    monkeypatch.setattr(safety.dist,'all_reduce',reduce)
    monkeypatch.setattr(safety.dist,'all_gather',gather)
    def run(rank):
        local.rank=rank;local.index=0
        actor,data,model,optimizer=fixtures[rank]
        if snapshot_conflict:
            with pytest.raises(RuntimeError,match='identity differs across ranks'):
                DataParallelPPOActor.update_policy(actor,data)
            event=None
        else:
            event=DataParallelPPOActor.update_policy(actor,data)['fresh200_update_state']
            assert event['status']=='SKIPPED' and event['optimizer_steps']==event['processed_opportunities']==0
        assert not optimizer.state and torch.equal(model.weight,torch.zeros_like(model.weight))
        return event,local.index
    with ThreadPoolExecutor(max_workers=8) as pool:results=list(pool.map(run,range(8)))
    assert len({count for _,count in results})==1
