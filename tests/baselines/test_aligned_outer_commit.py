"""Real CPU Adam/scheduler and actual driver completion method; no GPU acceptance."""
import copy
import os
from types import SimpleNamespace as NS
from unittest.mock import patch
import pytest
import torch
from omegaconf import OmegaConf
from search_r1.baselines import aligned_outer_commit as commit
from search_r1.baselines.fresh200_driver import Fresh200Progress
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def events(outer=1, attempt=1):
    return [dict(rank=r,baseline_aligned=True,actual_outer=outer,attempt_id=attempt,
        status='COMPLETE',optimizer_steps=20,expected_optimizer_steps=20,
        optimizer_opportunities_total=20,processed_opportunities=20,skipped_opportunities=0,
        unprocessed_opportunities=0,skipped_indices=[],skipped_reasons=[],postabort=False,
        update_complete=True,attempt_complete=True,outer_commit_phase='WRITES_PREPARED',
        accepted_complete_outer=False,scheduler_before=outer-1,scheduler_after=outer-1) for r in range(8)]


def worker(tmp_path, monkeypatch):
    def gather(output, value, group=None):
        for rank in range(8): output[rank]={**copy.deepcopy(value),'rank':rank}
    monkeypatch.setattr(commit.dist,'all_gather_object',gather)
    monkeypatch.setattr(commit.dist,'is_initialized',lambda:False)
    weight=torch.nn.Parameter(torch.tensor([1.]))
    optimizer=torch.optim.Adam([weight],lr=1e-6)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda n:(n+1)/57)
    return NS(rank=0,world_size=8,actor_module_fsdp=NS(process_group=None),
        config=OmegaConf.create({'actor':{'aligned_stage_commit_enabled':True,
            'fresh200_fault_dir':str(tmp_path/'ranks')}}),actor_lr_scheduler=scheduler,
        optimizer=optimizer,weight=weight)


def prepare_writes(w,outer=1,attempt=1):
    for _ in range(20):
        w.optimizer.zero_grad();w.weight.square().sum().backward();w.optimizer.step()
    e=events(outer,attempt);w._fresh200_last_update_event=e[0]
    commit.prepare(w,e)
    return e


def test_twenty_real_writes_wait_for_exact_stage_and_single_scheduler_commit(tmp_path,monkeypatch):
    w=worker(tmp_path,monkeypatch);prepare_writes(w)
    assert int(w.optimizer.state[w.weight]['step'])==20 and w.actor_lr_scheduler.last_epoch==0
    evidence=commit.stage_evidence(1,1,{'bapo/stage_evaluated_count':64,'bapo/stage_validation':.25})
    with pytest.raises(RuntimeError,match='admission'):commit.commit(w,1,2,evidence)
    assert w.actor_lr_scheduler.last_epoch==0
    report=commit.commit(w,1,1,evidence)
    assert report['scheduler_after']==1 and w._aligned_pending_outer is None
    assert report['event']['accepted_complete_outer'] is True
    with pytest.raises(RuntimeError,match='admission'):commit.commit(w,1,1,evidence)
    assert w.actor_lr_scheduler.last_epoch==1
    assert int(w.optimizer.state[w.weight]['step'])==20


@pytest.mark.parametrize('fault',['stage','stage_short','stage_nonfinite','prepared_save','stage_save','rpc'])
def test_actual_driver_failure_retains20_without_accepted_outer_or_lr(tmp_path,monkeypatch,fault):
    w=worker(tmp_path,monkeypatch);e=prepare_writes(w)
    t=RayPPOTrainer.__new__(RayPPOTrainer);t.global_steps=1
    t.fresh200_progress=Fresh200Progress(baseline_aligned=True);t.fresh200_progress.begin_attempt(1)
    t.fresh200_rank_updates=e;t.fresh200_shared_input_bindings={'stage':{'stage':{'rows':64}},'common_input_binding':{'sha256':'a'*64}}
    t.config=OmegaConf.create({'baselines':{'algorithm_source_commit':'a'*40,'alignment':{
        'order_manifest':'/bound/order.json','order_sha256':'b'*64}}})
    monkeypatch.setenv('BAPO_RUN_ID',str(tmp_path))
    accepted=[];faults=[];after=[]
    def stage(outer):
        if fault=='stage':raise RuntimeError('stage failed')
        return {'bapo/stage_evaluated_count':63 if fault=='stage_short' else 64,
                'bapo/stage_validation':float('nan') if fault=='stage_nonfinite' else .5}
    t.baseline_controller=NS(stage_validation=stage,after_actor=lambda *a:after.append(True),actual_outer_updates=0)
    t._fresh200_record_attempt=lambda *a:accepted.append(True) or True
    t._fresh200_fault_backup=lambda ranks:faults.append(copy.deepcopy(ranks))
    def rpc(*args):
        if fault=='rpc':raise RuntimeError('rpc failed')
        raise AssertionError('Failure case must never reach scheduler commit')
    t.actor_rollout_wg=NS(commit_aligned_outer=rpc)
    if fault in ('prepared_save','stage_save'):
        (tmp_path/('aligned-outer-prepared' if fault=='prepared_save' else 'aligned-stage-proof')).write_text('occupied')
    with pytest.raises((RuntimeError,ValueError,FileExistsError)):
        t._finish_aligned_outer(None,{}, {})
    assert len(faults)==1 and not accepted and not after
    assert t.fresh200_progress.completed_outers==0 and t.baseline_controller.actual_outer_updates==0
    assert w.actor_lr_scheduler.last_epoch==0 and int(w.optimizer.state[w.weight]['step'])==20
    assert w._aligned_pending_outer['attempt_id']==1


def test_actual_driver_five_successful_outer_commit_states(tmp_path,monkeypatch):
    w=worker(tmp_path,monkeypatch)
    t=RayPPOTrainer.__new__(RayPPOTrainer)
    t.fresh200_progress=Fresh200Progress(baseline_aligned=True)
    t.fresh200_shared_input_bindings={'stage':{'stage':{'rows':64}},'common_input_binding':{'sha256':'a'*64}}
    t.config=OmegaConf.create({'baselines':{'algorithm_source_commit':'a'*40,'alignment':{
        'order_manifest':'/bound/order.json','order_sha256':'b'*64}}})
    monkeypatch.setenv('BAPO_RUN_ID',str(tmp_path))
    order=[];controller=NS(actual_outer_updates=0)
    def stage(outer):
        order.append(('stage',outer,w.actor_lr_scheduler.last_epoch))
        return {'bapo/stage_evaluated_count':64,'bapo/stage_validation':.5}
    def after(*args):controller.actual_outer_updates+=1
    controller.stage_validation=stage;controller.after_actor=after;t.baseline_controller=controller
    def rpc(outer,attempt,evidence):
        order.append(('commit',outer,w.actor_lr_scheduler.last_epoch))
        r=commit.commit(w,outer,attempt,evidence)
        result=[]
        for rank in range(8):
            report=copy.deepcopy(r);report['rank']=rank;report['event']['rank']=rank;result.append(report)
        return result
    t.actor_rollout_wg=NS(commit_aligned_outer=rpc)
    def record(ranks,metrics):
        order.append(('accepted',t.global_steps,w.actor_lr_scheduler.last_epoch))
        return t.fresh200_progress.record(ranks,t.global_steps)
    t._fresh200_record_attempt=record
    t._fresh200_fault_backup=lambda *args:pytest.fail('Successful stage must not fault')
    for outer in range(1,6):
        t.global_steps=outer;t.fresh200_progress.begin_attempt(outer)
        t.fresh200_rank_updates=prepare_writes(w,outer,outer)
        metrics={};t._finish_aligned_outer(None,metrics,{})
        assert metrics['actor/scheduler_outer_steps']==outer
    assert order==[(name,o,o if name=='accepted' else o-1)for o in range(1,6)for name in ('stage','commit','accepted')]
    assert t.fresh200_progress.completed_outers==controller.actual_outer_updates==5
    assert t.fresh200_progress.actual_adam_writes_per_rank==[100]*8
    assert int(w.optimizer.state[w.weight]['step'])==100 and w.actor_lr_scheduler.last_epoch==5

@pytest.mark.parametrize('fault',['stage','debug'])
def test_actual_fit_defers_acceptance_and_backs_up_postwrite_evaluation_fault(tmp_path,monkeypatch,fault):
    import contextlib
    import io
    import numpy as np
    from torch.utils.data import DataLoader
    from verl import DataProto
    from search_r1.baselines.full_state import StatefulSampler
    t=RayPPOTrainer.__new__(RayPPOTrainer)
    t.config=OmegaConf.create(dict(baseline_aligned=True,max_turns=4,force_final_answer=True,do_search=False,
        data=dict(train_batch_size=1,max_start_length=2048,max_prompt_length=4096,max_response_length=500,max_obs_length=500),
        retriever=dict(url='http://fixture',topk=3),algorithm=dict(no_think_rl=False),
        baselines=dict(algorithm_source_commit='a'*40,alignment=dict(order_manifest='/order',order_sha256='b'*64)),
        actor_rollout_ref=dict(rollout=dict(n_agent=1,n=1),actor=dict(state_masking=False,policy_safety={})),
        trainer=dict(fresh200_enabled=True,aligned_stage_commit_enabled=True,val_before_train=False,critic_warmup=0,
            total_epochs=1,n_gpus_per_node=8,nnodes=1,test_freq=1 if fault=='debug' else 0,save_freq=50,
            acceptance_step=5,default_local_dir=str(tmp_path),fresh200_diagnostic_only=False)))
    t.total_training_steps=201;t.use_critic=False;t.use_reference_policy=False;t.use_rm=False
    t.val_reward_fn=object() if fault=='debug' else None;t.tokenizer=NS(pad_token_id=0,eos_token_id=0)
    t.logger=NS(log=lambda **kwargs:None);t.aligned_order_manifest={}
    t.fresh200_shared_input_bindings={'stage':{'stage':{'rows':64}},'common_input_binding':{}}
    rows=[dict(input_ids=torch.tensor([1]),attention_mask=torch.tensor([1]),position_ids=torch.tensor([0]),index=np.array('q'))]
    t.e1_sampler=StatefulSampler(rows,seed=42)
    def collate(items):
        return {k:torch.stack([r[k]for r in items]) if isinstance(items[0][k],torch.Tensor) else np.array([r[k].item()for r in items])for k in items[0]}
    t.train_dataloader=DataLoader(rows,batch_size=1,sampler=t.e1_sampler,collate_fn=collate,num_workers=0)
    w=worker(tmp_path,monkeypatch);updates=[];stages=[];after=[];backups=[]
    def generate(manager,batch,qa,metrics):
        return DataProto.from_dict(dict(prompts=batch.batch['input_ids'],responses=torch.tensor([[9]]),
            input_ids=torch.tensor([[1,9]]),attention_mask=torch.ones(1,2,dtype=torch.long),
            info_mask=torch.ones(1,2,dtype=torch.long),position_ids=torch.tensor([[0,1]]),old_log_probs=torch.zeros(1,1)),meta_info=dict(batch.meta_info)),[0]
    def advantage(batch,metrics):
        batch.batch['advantages']=torch.ones_like(batch.batch['responses']).float();return batch
    def stage(step):
        stages.append(step);raise RuntimeError('prescribed stage failed')
    controller=NS(method='bapo',generate=generate,compute_advantages=advantage,actual_outer_updates=0,
                  after_actor=lambda *args:after.append(True),stage_validation=stage)
    t.baseline_controller=controller
    t.reward_fn=lambda batch:torch.ones_like(batch.batch['responses']).float()
    t._balance_batch=lambda *args,**kwargs:None;t._validate=lambda:(_ for _ in ()).throw(RuntimeError('debug failed'))
    def update(batch):
        updates.append(True);e=prepare_writes(w)
        return DataProto(meta_info={'metrics':{'actor/optimizer_steps':[20]},'fresh200_rank_updates':e})
    t.actor_rollout_wg=NS(update_actor=update,generate_sequences=lambda batch:generate(None,batch,None,{})[0],
        compute_log_prob=lambda batch:DataProto.from_dict({'old_log_probs':torch.zeros(1,1)}))
    t._fresh200_fault_backup=lambda events:backups.append(copy.deepcopy(events))
    monkeypatch.setenv('BAPO_RUN_ID',str(tmp_path))
    monkeypatch.setattr('search_r1.baselines.baseline_alignment_data.validate_emitted_batch',lambda *a:{})
    with contextlib.redirect_stdout(io.StringIO()),pytest.raises(RuntimeError,match='stage failed' if fault=='stage' else 'debug failed'):
        t.fit()
    assert updates==[True] and not after and len(backups)==1
    assert stages==([1]if fault=='stage' else [])
    assert t.fresh200_progress.completed_outers==controller.actual_outer_updates==0
    assert w.actor_lr_scheduler.last_epoch==0 and int(w.optimizer.state[w.weight]['step'])==20
    assert not list((tmp_path/'diagnostics').glob('*/all-rank-write-ledger.json'))


def test_collective_scheduler_epoch_disagreement_refuses_commit_before_step(tmp_path,monkeypatch):
    w=worker(tmp_path,monkeypatch);prepare_writes(w)
    def disagree(output,value,group=None):
        for rank in range(8):output[rank]={**copy.deepcopy(value),'rank':rank}
        output[7]['scheduler_before']=99
    monkeypatch.setattr(commit.dist,'all_gather_object',disagree)
    evidence=commit.stage_evidence(1,1,{'bapo/stage_evaluated_count':64,'bapo/stage_validation':.5})
    with pytest.raises(RuntimeError,match='state differs'):commit.commit(w,1,1,evidence)
    assert w.actor_lr_scheduler.last_epoch==0 and int(w.optimizer.state[w.weight]['step'])==20


def test_scheduler_fault_retains_pending_native_identity_and_never_accepts(tmp_path,monkeypatch):
    w=worker(tmp_path,monkeypatch);prepare_writes(w)
    monkeypatch.setattr(w.actor_lr_scheduler,'step',lambda:(_ for _ in ()).throw(OSError('scheduler fault')))
    evidence=commit.stage_evidence(1,1,{'bapo/stage_evaluated_count':64,'bapo/stage_validation':.5})
    with pytest.raises(RuntimeError,match='scheduler fault'):commit.commit(w,1,1,evidence)
    assert w._aligned_pending_outer['phase']=='COMMIT_FAULT'
    assert w._fresh200_last_update_event['status']=='INFRASTRUCTURE_FAULT'
    assert not w._fresh200_last_update_event['accepted_complete_outer']
    assert w.actor_lr_scheduler.last_epoch==0 and int(w.optimizer.state[w.weight]['step'])==20


def test_final_rank_ledger_io_fault_keeps_scheduler_side_effect_as_fatal_partial(tmp_path,monkeypatch):
    w=worker(tmp_path,monkeypatch);prepare_writes(w)
    import search_r1.baselines.policy_safety as safety
    def failed(path,event,group):
        event=dict(event,status='INFRASTRUCTURE_FAULT',fatal=True,update_complete=False,attempt_complete=False)
        return event,[event]*8
    monkeypatch.setattr(safety,'persist_rank_update_event',failed)
    evidence=commit.stage_evidence(1,1,{'bapo/stage_evaluated_count':64,'bapo/stage_validation':.5})
    with pytest.raises(RuntimeError,match='persistence failed'):commit.commit(w,1,1,evidence)
    assert w.actor_lr_scheduler.last_epoch==1  # No automatic rollback of an actual side effect.
    assert w._aligned_pending_outer['phase']=='COMMIT_FAULT'
    assert w._fresh200_last_update_event['accepted_complete_outer'] is False
    assert int(w.optimizer.state[w.weight]['step'])==20


@pytest.mark.parametrize('count,score',[(64.,.5),(True,.5),(63,.5),(64,True),(64,float('nan')),(64,float('inf')),(64,-.1),(64,1.1)])
def test_stage_evidence_rejects_nonintegral_coverage_and_invalid_em(count,score):
    with pytest.raises(ValueError):
        commit.stage_evidence(1,1,{'bapo/stage_evaluated_count':count,'bapo/stage_validation':score})


@pytest.mark.parametrize('count,score',[(64.,.5),(64,-.1),(64,1.1)])
def test_rpc_refuses_invalid_stage_fields_even_with_consistent_sha(tmp_path,monkeypatch,count,score):
    w=worker(tmp_path,monkeypatch);prepare_writes(w)
    body=dict(actual_outer=1,attempt_id=1,evaluated_count=count,score=score)
    body['sha256']=commit._digest(body)
    with pytest.raises(RuntimeError,match='admission'):
        commit.commit(w,1,1,body)
    assert w.actor_lr_scheduler.last_epoch==0


def test_scheduled_checkpoint_fault_preserves_factual_completed_outer(tmp_path):
    import json
    t=RayPPOTrainer.__new__(RayPPOTrainer);t.global_steps=1
    t.config=OmegaConf.create({'trainer':{'aligned_stage_commit_enabled':True,'default_local_dir':str(tmp_path)},
        'baselines':{'algorithm_source_commit':'a'*40}})
    t.fresh200_progress=Fresh200Progress(baseline_aligned=True);t.fresh200_progress.begin_attempt(1)
    e=events();t.fresh200_progress.record(e,1)
    for row in e:row.update(outer_commit_phase='STAGE_COMMITTED',accepted_complete_outer=True,scheduler_after=1)
    t.baseline_controller=NS(actual_outer_updates=1)
    native=[];t._save_fresh200_state=lambda path,status,ranks:native.append((status,copy.deepcopy(ranks)))
    try:raise OSError('scheduled checkpoint failed after stage commit')
    except OSError:t._fresh200_fault_backup(e)
    outcome=json.loads((tmp_path/'diagnostics/attempt_1_INFRA_FAULT/all-rank-fatal-ledger.json').read_text())
    assert outcome['fault_phase']=='POST_ACCEPTED_OUTER_CHECKPOINT_INFRASTRUCTURE_FAULT'
    assert outcome['accepted_complete_outer_preserved'] is True
    assert outcome['validated_previous_accounting']['completed_outers']==1
    assert outcome['validated_previous_accounting']['actual_adam_writes_per_rank']==[20]*8
    assert native[0][0]=='PARTIAL' and native[0][1][0]['scheduler_after']==1
