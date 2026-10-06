"""Actual fit control flow with mocked GPU RPCs; no CUDA acceptance."""
import contextlib
import io
import os
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from search_r1.baselines.controller import Controller
from search_r1.baselines.full_state import StatefulSampler
from search_r1.baselines.fresh200_contract import fresh200_lr_factor


@pytest.mark.parametrize('fatal_metadata',[False,True,'commit_fault'])
def test_actual_fit_recycles_sampler_until200_accepted_with_fresh_abort_rollouts(tmp_path,fatal_metadata):
    t=RayPPOTrainer.__new__(RayPPOTrainer)
    t.config=OmegaConf.create(dict(max_turns=4,force_final_answer=False,do_search=False,
        data=dict(max_start_length=2048,max_prompt_length=4096,max_response_length=500,max_obs_length=500),
        retriever=dict(url='http://fixture',topk=3),algorithm=dict(no_think_rl=False),
        baselines=dict(algorithm_source_commit='a'*40),actor_rollout_ref=dict(
            rollout=dict(n_agent=1,n=1),actor=dict(state_masking=False,policy_safety={})),
        trainer=dict(fresh200_enabled=True,val_before_train=False,critic_warmup=0,
            total_epochs=1,n_gpus_per_node=8,nnodes=1,test_freq=0,save_freq=50,acceptance_step=5,
            default_local_dir=str(tmp_path),fresh200_diagnostic_only=False)))
    t.fresh200_shared_input_bindings={}
    t.total_training_steps=201;t.use_critic=False;t.use_reference_policy=False;t.use_rm=False
    t.val_reward_fn=None;t.tokenizer=NS(pad_token_id=0,eos_token_id=0)
    t.logger=NS(log=lambda **kwargs:None)
    rows=[dict(input_ids=torch.tensor([i+1]),attention_mask=torch.tensor([1]),position_ids=torch.tensor([0]),index=np.array('q'+str(i))) for i in range(2)]
    t.e1_sampler=StatefulSampler(rows,seed=42)
    def collate(items):
        return {key:(torch.stack([row[key]for row in items]) if isinstance(items[0][key],torch.Tensor)
                     else np.array([row[key].item() for row in items])) for key in items[0]}
    t.train_dataloader=DataLoader(rows,batch_size=1,sampler=t.e1_sampler,collate_fn=collate,num_workers=0)
    controller=Controller.__new__(Controller);controller.method='bapo';controller.ranker=None;controller.actual_outer_updates=0
    controller.stage=NS(state_dict=lambda:dict(stage=1),load_state_dict=lambda state:None)
    def advantages(batch,metrics):
        batch.batch['advantages']=torch.ones_like(batch.batch['responses']).float()
        metrics['bapo/reward_fixture']=1.;return batch
    controller.compute_advantages=advantages
    stage_calls=[];controller.stage_validation=lambda step:stage_calls.append(step) or {}
    t.baseline_controller=controller
    t.reward_fn=lambda batch:torch.ones_like(batch.batch['responses']).float()
    t._balance_batch=lambda batch,metrics:None
    saves=[];t._save_checkpoint=lambda:saves.append(t.global_steps)
    pauses=[];t._fresh200_pause_after_five=lambda:pauses.append((t.global_steps,t.fresh200_progress.attempts))
    calls=[];native=[];seen=set();worker=NS(scheduler_steps=0)
    def generation(batch):
        # Snapshot changes on every attempt, while the selected corpus wraps normally.
        return DataProto.from_dict(dict(prompts=batch.batch['input_ids'],responses=torch.tensor([[9]]),
            input_ids=torch.cat((batch.batch['input_ids'],torch.tensor([[9]])),dim=1),
            attention_mask=torch.ones(1,2,dtype=torch.long),info_mask=torch.ones(1,2,dtype=torch.long),
            position_ids=torch.tensor([[0,1]]),old_log_probs=torch.zeros(1,1)),
            meta_info=dict(batch.meta_info))
    def update(batch):
        outer=batch.meta_info['actual_outer'];attempt=batch.meta_info['attempt_id'];expected=20 if outer<50 else 10
        first=outer not in seen;seen.add(outer)
        if first and outer in (1,50):status='ABORTED';writes=3 if outer==1 else expected;processed=writes;postabort=True
        elif first and outer==200:status='SKIPPED';writes=0;processed=expected;postabort=False
        else:status='COMPLETE_WITH_SKIPS';writes=expected-1;processed=expected;postabort=False
        complete=status=='COMPLETE_WITH_SKIPS'
        events=[dict(rank=r,actual_outer=outer,attempt_id=attempt,status=status,optimizer_steps=writes,
            expected_optimizer_steps=expected,optimizer_opportunities_total=expected,processed_opportunities=processed,
            skipped_opportunities=processed-writes,unprocessed_opportunities=expected-processed,
            postabort=postabort,update_complete=complete,attempt_complete=processed==expected and not postabort) for r in range(8)]
        if fatal_metadata is True:
            for event in events:event['expected_optimizer_steps']=float(expected)
        elif fatal_metadata=='commit_fault':
            for event in events:
                event.update(status='INFRASTRUCTURE_FAULT',confirmed_previous_optimizer_steps=3,
                    optimizer_steps=4 if event['rank']==0 else 3,
                    failed_commit_progress=dict(local_commit_complete=event['rank']==0,
                        parameter_copies_complete=2 if event['rank']==0 else 1,copy_outcome='UNKNOWN_PARTIAL_ON_FAILURE'))
        calls.append(dict(outer=outer,attempt=attempt,status=status,mini=batch.meta_info['global_mini_batch_size'],
            question=int(batch.batch['prompts'][0,0]),snapshot=batch.meta_info['policy_snapshot_id'],
            lr=fresh200_lr_factor(worker.scheduler_steps),scheduler_before=worker.scheduler_steps))
        if complete:worker.scheduler_steps+=1
        return DataProto(meta_info=dict(metrics={'actor/optimizer_steps':[writes]},fresh200_rank_updates=events))
    def save_native(directory,step,roundtrip,state_status):
        native.append(dict(step=step,tag=state_status,scheduler=worker.scheduler_steps))
        return [dict(rank=r) for r in range(8)]
    worker.generate_sequences=generation;worker.update_actor=update
    worker.save_checkpoint=lambda *args:None;worker.save_training_state=save_native;t.actor_rollout_wg=worker
    def proof_boundary(data,*args):
        return dict(scope='MOCK_GPU_BOUNDARY_ONLY',attempt_id=data.meta_info['attempt_id'],
            sampling_verifier_status='PASS_ACTUAL_BATCH_BINDING_NOT_RELEASE')
    with patch('search_r1.baselines.fresh200_driver.persist_first_sampling_batch',side_effect=proof_boundary),patch.dict(os.environ,{'BAPO_RUN_ID':str(tmp_path)}),patch('verl.trainer.ppo.ray_trainer.compute_data_metrics',return_value={}),patch('verl.trainer.ppo.ray_trainer.compute_timing_metrics',return_value={}),contextlib.redirect_stdout(io.StringIO()):
        if fatal_metadata:
            with pytest.raises(ValueError):t.fit()
        else:t.fit()
    if not fatal_metadata:assert t.fresh200_first_sampling_proof['attempt_id']==2
    if fatal_metadata:
        assert t.fresh200_progress.completed_outers==0 and len(calls)==1 and stage_calls==[] and pauses==[]
        assert native==[dict(step=1,tag='PARTIAL',scheduler=0)]
        backup=torch.load(tmp_path/'fresh200_state/attempt_1_candidate_outer_1_PARTIAL/driver.pt',weights_only=False)
        assert backup['infrastructure_fault'] and backup['previous_completed_outer']==0
        assert backup['rank_write_ledger'][0]['optimizer_steps']==(4 if fatal_metadata=='commit_fault' else 3)
        assert 'VALIDATED_ATTEMPTS_ONLY' in backup['progress_accounting_scope']
        assert (tmp_path/'diagnostics/attempt_1_INFRA_FAULT/driver-fault.raw.log').is_file()
        import json
        fatal=json.loads((tmp_path/'diagnostics/attempt_1_INFRA_FAULT/all-rank-fatal-ledger.json').read_text())
        assert len(fatal['rank_write_ledger'])==8 and fatal['previous_completed_outer']==0
        if fatal_metadata=='commit_fault':
            assert fatal['rank_write_ledger'][1]['failed_commit_progress']['copy_outcome']=='UNKNOWN_PARTIAL_ON_FAILURE'
        return
    assert t.global_steps==200 and t.fresh200_progress.completed_outers==200 and controller.actual_outer_updates==200
    assert t.fresh200_progress.attempts==203 and worker.scheduler_steps==200
    assert len(stage_calls)==200 and stage_calls==list(range(1,201))
    assert saves==[5,50,100,150,200] and pauses==[(5,6)]
    assert calls[0]['outer']==calls[1]['outer']==1 and calls[0]['snapshot']!=calls[1]['snapshot']
    assert calls[0]['lr']==calls[1]['lr']==1/57 and calls[0]['question']!=calls[1]['question']
    assert all(c['mini']==128 for c in calls if c['outer']==49)
    assert all(c['mini']==256 and c['scheduler_before']==49 for c in calls if c['outer']==50)
    assert t.e1_sampler.epoch>=100 # configured total_epochs1 does not truncate completion.
    assert native==[dict(step=1,tag='ABORTED',scheduler=0),dict(step=50,tag='ABORTED',scheduler=49)]
    partial=torch.load(tmp_path/'fresh200_state/attempt_1_candidate_outer_1_ABORTED/driver.pt',weights_only=False)
    assert partial['source_commit']=='a'*40 and partial['attempt_id']==1 and partial['previous_completed_outer']==0
    assert partial['status']=='ABORTED' and partial['next_global_step']==1 and partial['reward_metrics']['bapo/reward_fixture']==1.
    assert partial['progress']['actual_adam_writes_per_rank']==[3]*8
