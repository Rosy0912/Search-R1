"""CPU-only position segmentation fixtures and complete eight-rank evidence wiring."""
import importlib.util
import json
from pathlib import Path
import tempfile

import torch

WORK = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('position_boundary_analysis', WORK/'inspect_tree_position_boundaries.py')
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)

# CPU-safe prepare body and exact branch predicate from the pinned HF 4.46.3
# source. Real execution loads this pair from the installed SHA-bound source.
HF_FIXTURE = '''import torch
def prepare_fa2_from_position_ids(query, key, value, position_ids):
    query = query.view(-1, query.size(-2), query.size(-1))
    key = key.view(-1, key.size(-2), key.size(-1))
    value = value.view(-1, value.size(-2), value.size(-1))
    position_ids = position_ids.flatten()
    indices_q = torch.arange(position_ids.size(0), device=position_ids.device, dtype=torch.int32)
    cu_seq_lens = torch.cat((indices_q[position_ids == 0], torch.tensor(position_ids.size(), device=position_ids.device, dtype=torch.int32)))
    max_length = position_ids.max() + 1
    return (query, key, value, indices_q, (cu_seq_lens, cu_seq_lens), (max_length, max_length))
def _flash_attention_forward(query_states, key_states, value_states, attention_mask, query_length, position_ids):
    if attention_mask is not None:
        raise AssertionError('GPU path must not run')
    elif position_ids is not None and query_length != 1 and not (torch.diff(position_ids, dim=-1) >= 0).all():
        raise AssertionError('GPU path must not run')
    else:
        raise AssertionError('GPU path must not run')
'''

QWEN_FIXTURE = '''class Qwen2FlashAttention2:
    def forward(self,query_states,key_states,value_states,attention_mask,q_len,position_ids):
        return _flash_attention_forward(query_states,key_states,value_states,attention_mask,q_len,position_ids=position_ids)
QWEN2_ATTENTION_CLASSES = {'flash_attention_2':Qwen2FlashAttention2}
'''
WORKER_FIXTURE = '''class ActorRolloutRefWorker:
    def _build_model_optimizer(self,use_remove_padding):
        if use_remove_padding and self.ulysses_sequence_parallel_size > 1:
            apply_monkey_patch(actor_model_config,verbose=True)
        actor_module = AutoModelForCausalLM.from_pretrained(config=actor_model_config,attn_implementation='flash_attention_2')
'''


def inputs(positions, mask=None, loss=None):
    positions=torch.tensor(positions, dtype=torch.int64)
    value=dict(input_ids=torch.arange(positions.numel()).reshape(positions.shape),
        position_ids=positions, attention_mask=torch.ones_like(positions) if mask is None else torch.tensor(mask))
    if loss is not None: value['loss_mask']=torch.tensor(loss)
    return value


def refuses(call):
    try: call()
    except ValueError: return
    raise AssertionError('Invalid evidence was admitted')


def bundle(root, A):
    run=root/'tree-safety300-999999'; package_dir=root/'package'; package_dir.mkdir()
    output=run/'output'; (output/'probability-calibration').mkdir(parents=True)
    (output/'entry-audit').mkdir(); (output/'sampling-consistency').mkdir()
    config=json.dumps({'data':{'train_batch_size':640},'actor_rollout_ref':{'actor':{'ppo_micro_batch_size':16,'ppo_mini_batch_size':384,'ppo_epochs':1,'use_dynamic_bsz':False,
        'ulysses_sequence_parallel_size':1,'policy_safety':{'monitor_rows_per_rank':32}},
        'rollout':{'log_prob_micro_batch_size':32,'n':1,'tensor_model_parallel_size':1,'log_prob_use_dynamic_bsz':False},
        'model':{'use_remove_padding':True}}}).encode(); (run/'resolved.yaml').write_bytes(config)
    (output/'entry-audit/entry-config.yaml').write_bytes(config)
    parent=root/'parent-manifest.json'; parent_identity={'completed_step':110}
    parent.write_text(json.dumps({'identity':parent_identity}))
    package=dict(stage='calibration110', probability_calibration={'environment':{'TREE_PPO_CALIBRATION_ONLY':'1'}},
        code_commit='a'*40,common_data_identity='d'*64,model_identity={'sha256':'e'*64},files={},
        parent_manifest=dict(path=str(parent),sha256=A.file_sha(parent),identity=parent_identity))
    for name in set(A.SOURCE_FILES+P.FORWARD_FILES):
        path=package_dir/'source/repo'/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('Synthetic source fixture\n'); package['files']['source/repo/'+name]=A.file_sha(path)
    manifest=package_dir/'manifest.json';manifest.write_text(json.dumps(package)); package_sha=A.file_sha(manifest)
    identity=dict(job_id='999999',run_id=run.name,code_commit='a'*40,config_sha256=A.file_sha(run/'resolved.yaml'),data_identity='d'*64,model_identity='e'*64)
    ready=dict(status='CALIBRATION_OBSERVED_ZERO_WRITES',parent_completed_outer=110,observed_outer=111,new_completed_outer=0,
        optimizer_committed=0,scheduler_advanced=False,**identity,workers=[])
    final=dict(input_ids=torch.tensor([[1,2,3]]),responses=torch.tensor([[3]]),position_ids=torch.tensor([[0,1,2]]),
        attention_mask=torch.ones((1,3),dtype=torch.int64),old_log_probs=torch.tensor([[-2.25]]),
        loss_mask=torch.ones((1,1),dtype=torch.int64),policy_weights=torch.ones((1,1)))
    for rank in range(8):
        raw=dict(final);raw.pop('loss_mask');raw.pop('policy_weights')
        raw.update(sampled_log_probs=torch.tensor([[-2.]]),distribution={'temperature':1.})
        snapshot=output/'sampling-consistency'/f'rank-{rank:02d}-call-00001.pt';torch.save(raw,snapshot)
        raw_json=snapshot.with_suffix('.json')
        raw_json.write_text(json.dumps(dict(**identity,rank=rank,world_size=8,scheduler_step=110,optimizer_committed=0,
            saved_rows=1,local_rows=1,enforced=False,generation_call=1,snapshot_path=str(snapshot),snapshot_sha256=A.file_sha(snapshot),
            distribution={'temperature':1.},decision={'allowed':False,'reasons':['trial_observation']})))
        monitor=output/'probability-calibration'/f'rank-{rank:02d}-monitor.pt'
        torch.save({key:value.repeat(32,1) for key,value in final.items()},monitor)
        ready['workers'].append(dict(**identity,rank=rank,world_size=8,optimizer_committed=0,scheduler_advanced=False,
            scheduler_step=110,restored_optimizer_steps=[2500],diagnostic_records=[str(raw_json)],
            diagnostic_entries=[dict(path=str(raw_json),sha256=A.file_sha(raw_json),snapshot_path=str(snapshot),snapshot_sha256=A.file_sha(snapshot))],
            monitor_path=str(monitor),monitor_sha256=A.file_sha(monitor)))
    (output/'recovery-loaded.json').write_text(json.dumps({'provenance':{'parent_identity':parent_identity,'parent_cursor':57600,'exact_executor_resume':False}}))
    ready_path=output/'probability-calibration/ready.json';ready_path.write_text(json.dumps(ready));ready_sha=A.file_sha(ready_path)
    actor=output/'trajectories/outer-000111';actor.mkdir(parents=True)
    actor_identity=dict(outer_step=111,job_id='999999',source='a'*40,config=identity['config_sha256'],data='d'*64)
    # A scalar metadata tensor must not be mistaken for a row-partitioned input.
    tensors={key:value.repeat(3840,1) for key,value in final.items()};tensors['scalar_fixture']=torch.tensor(0)
    torch.save(dict(identity=actor_identity,rows=3840,tensors=tensors),actor/'actor-inputs.pt')
    (actor/'manifest.json').write_text(json.dumps(dict(status='ALL_ROLLOUTS_AND_ACTOR_INPUTS_PUBLISHED',identity=actor_identity,
        rows=3840,payloads={'actor-inputs.pt':A.file_sha(actor/'actor-inputs.pt')})))
    return run,package_dir,package_sha,ready_sha


def main():
    torch.set_num_threads(4); checks=[]
    with tempfile.TemporaryDirectory(prefix='position-boundary-cpu-',dir=WORK) as temporary:
        root=Path(temporary); hf=root/'hf_fixture.py'; hf.write_text(HF_FIXTURE)
        rules=P.hf_rules(hf,P.sha(hf))
        out=P.inspect_positions(inputs([[0,1,2]]*4),rules,'normal')
        assert out['rows']['nonzero_start_position_rows']==0
        assert all(v['groups_with_boundary_mismatch']==0 for v in out['local_grouping'].values())
        checks.append('all_zero_row_starts_recover_exact_micro4_and_micro2_boundaries')
        out=P.inspect_positions(inputs([[3,4,5],[0,1,2]]),rules,'cropped-first')
        assert out['rows']['nonzero_start_position_rows']==1
        assert out['local_grouping']['4']['packed_varlen_groups']==1
        assert out['local_grouping']['4']['groups_with_uncovered_prefix']==1
        assert out['local_grouping']['4']['missing_boundaries']==1
        checks.append('nonzero_first_row_missing_initial_cu_boundary_not_called_rope_root_cause')
        out=P.inspect_positions(inputs([[3,4,5],[6,7,8]]),rules,'monotone-nonzero')
        stats=out['local_grouping']['4']
        assert stats['single_sequence_groups']==1 and stats['possible_cross_sample_boundaries']==1
        assert stats['attention_valid_tokens_in_crossing_segments']==6
        checks.append('monotone_flat_positions_select_single_sequence_and_span_true_row_boundary')
        out=P.inspect_positions(inputs([[0,1,0,1],[0,1,2,3]]),rules,'interior-zero')
        assert out['rows']['noncontiguous_valid_position_rows']==1
        assert out['local_grouping']['4']['extra_boundaries']==1
        checks.append('interior_position_zero_adds_extra_hf_boundary')
        out=P.inspect_positions(inputs([[0,1,2],[2,3,4],[0,1,2],[0,1,2]]),rules,'missing-interior')
        assert out['local_grouping']['4']['missing_sample_boundaries']==1
        assert out['local_grouping']['4']['possible_cross_sample_boundaries']==1
        checks.append('packed_varlen_can_keep_start_but_omit_internal_row_boundary')
        out=P.inspect_positions(inputs([[0,1],[2,3],[0,1],[2,3]]),rules,'micro-rule-change')
        assert out['local_grouping']['4']['packed_varlen_groups']==1
        assert out['local_grouping']['2']['single_sequence_groups']==2
        assert out['local_grouping']['4']['missing_sample_boundaries']==out['local_grouping']['2']['missing_sample_boundaries']==2
        checks.append('micro4_vs_micro2_changes_real_hf_branch_predicate_not_just_row_count')
        out=P.inspect_positions(inputs([[88,0,1,2,99],[88,0,1,2,99]],mask=[[0,1,1,1,0]]*2,loss=[[0,0,0,1,0]]*2),rules,'masked-observation')
        assert out['rows']['attention_valid_tokens']==6 and out['rows']['nonzero_start_position_rows']==0
        assert out['local_grouping']['4']['groups_with_boundary_mismatch']==0
        checks.append('padding_excluded_but_loss_mask_zero_observation_retained_for_attention_boundaries')
        out=P.inspect_positions(inputs([[88],[99]],mask=[[0],[0]]),rules,'empty')
        assert out['rows']['empty_rows']==2 and out['local_grouping']['4']['zero_token_groups']==1
        checks.append('all_padding_rows_do_not_enter_cpu_prepare')
        out=P.inspect_positions(inputs([[0]]),rules,'single-token')
        assert out['local_grouping']['4']['single_sequence_groups']==1 and out['local_grouping']['4']['missing_boundaries']==0
        checks.append('query_length_one_follows_actual_hf_single_sequence_condition')
        identity=P.tensor_identity(torch.tensor([[0,1,2]],dtype=torch.int64))
        assert identity['shape']==[1,3] and identity['dtype']=='torch.int64'
        assert identity['shape_and_bytes_sha256']!=P.tensor_identity(torch.tensor([[0,1,3]],dtype=torch.int64))['shape_and_bytes_sha256']
        assert identity['shape_and_bytes_sha256']!=P.tensor_identity(torch.tensor([[0,1,2]],dtype=torch.int32))['shape_and_bytes_sha256']
        checks.append('full_tensor_shape_dtype_and_all_byte_hash_binding')
        refuses(lambda:P.hf_rules(hf,'0'*64));checks.append('changed_hf_source_sha_refused')
        qwen,worker=root/'qwen.py',root/'worker.py'
        qwen.write_text(QWEN_FIXTURE);worker.write_text(WORKER_FIXTURE)
        caller=P.native_qwen_caller(qwen,P.sha(qwen),worker,P.sha(worker))
        assert caller['position_ids_keyword'] is True and caller['actor_monkey_patch_at_sp1'] is False
        assert caller['live_worker_function_identity_observed'] is False
        checks.append('native_qwen_position_keyword_and_constructor_flash2_bound_with_sp1_monkeypatch_false')
        qwen.write_text(QWEN_FIXTURE.replace(',position_ids=position_ids)',' )'))
        refuses(lambda:P.native_qwen_caller(qwen,P.sha(qwen),worker,P.sha(worker)));qwen.write_text(QWEN_FIXTURE)
        checks.append('native_qwen_missing_position_keyword_refuses_source_simulation_as_active_path')
        worker.write_text(WORKER_FIXTURE.replace('> 1','>= 1'))
        refuses(lambda:P.native_qwen_caller(qwen,P.sha(qwen),worker,P.sha(worker)));worker.write_text(WORKER_FIXTURE)
        checks.append('changed_guard_enabling_monkeypatch_at_sp1_refused')
        worker.write_text(WORKER_FIXTURE.replace("attn_implementation='flash_attention_2'","attn_implementation='sdpa'"))
        refuses(lambda:P.native_qwen_caller(qwen,P.sha(qwen),worker,P.sha(worker)));worker.write_text(WORKER_FIXTURE)
        checks.append('different_actor_attention_constructor_refused')
        refuses(lambda:P.context_tools('0'*64));checks.append('changed_context_helper_sha_refused_before_import')
        refuses(lambda:P.inspect_positions(inputs([[0,1]],mask=[[1,2]]),rules,'bad-mask'))
        checks.append('nonbinary_attention_mask_refused')
        A=P.context_tools(P.sha(WORK/'analyze_tree_sampling_calibration.py'))
        run,package,package_sha,ready_sha=bundle(root,A)
        verified=A.verify_bundle(run,package,package_sha,ready_sha,require_node=False)
        out=P.inspect_bundle(A,verified,package,rules,2)
        assert out['actor_input_rows']==3840 and len(out['inspections'])==24
        for kind,count in [('raw_same_turn',8),('final_leaf_monitor_subset',256),('complete_final_actor_inputs_dp_partition',3840)]:
            population=[x for x in out['inspections'] if x['kind']==kind]
            assert len(population)==8 and sum(x['inspection']['rows']['rows'] for x in population)==count
        assert out['actual_optimizer_writes']==out['actual_model_forwards']==0
        assert out['grouping_config']['rank_mini']==48 and out['grouping_config']['rank_actor_micro']==2
        assert 'BEFORE balance' in out['inspections'][-1]['inspection']['grouping_scope']
        assert all(x['inspection']['local_grouping']['4']['missing_boundaries']==0 for x in out['inspections'])
        json.dumps(out,allow_nan=False)
        checks.append('complete_eight_rank_raw_monitor_and3840_dp_input_wiring_and_scalar_metadata_exclusion')
        checks.append('full_actor_post_balance_micro4_hypothetical_and_micro2_prospective_scope_not_original_old_recompute')
        unbound=run/'output/sampling-consistency/rank-00-call-00002.json';unbound.write_text('{}')
        refuses(lambda:P.inspect_bundle(A,verified,package,rules,0));unbound.unlink()
        checks.append('extra_unbound_raw_call_evidence_refused')
        audit=run/'output/tree-rank-audit';audit.mkdir();auditfile=audit/'step-000111.json';auditfile.write_text('{}')
        refuses(lambda:P.inspect_bundle(A,verified,package,rules,0));auditfile.unlink()
        checks.append('optimizer_audit_file_refuses_zero_write_claim')
        checkpoints=run/'output/checkpoints';checkpoints.mkdir();(checkpoints/'unexpected.pt').write_bytes(b'x')
        refuses(lambda:P.inspect_bundle(A,verified,package,rules,0));(checkpoints/'unexpected.pt').unlink()
        checks.append('new_checkpoint_file_refuses_zero_write_claim')
        forward=package/'source/repo'/P.FORWARD_FILES[0];forward.write_text('changed source')
        refuses(lambda:P.inspect_bundle(A,verified,package,rules,0))
        checks.append('frozen_forward_source_mutation_refused')
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(status='POSITION_BOUNDARY_CPU_FIXTURES_PASS',checks=checks,checks_count=len(checks),
        helper_sha256=P.sha(WORK/'inspect_tree_position_boundaries.py'),test_sha256=P.sha(__file__),
        context_helper_sha256=P.sha(WORK/'analyze_tree_sampling_calibration.py'),cuda_initialized=False,
        actual_calibration_assets_loaded=False,actual_model_forwards=0,actual_optimizer_writes=0,remote_calls=False,
        scope='Fabricated CPU fixtures only; actual installed source and real h100-9 tensors still require the separate same-node CPU job'),sort_keys=True,allow_nan=False))


if __name__=='__main__':main()
