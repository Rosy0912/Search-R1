"""Small CPU context, duplication, position, masking and identity counterexamples."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile

import torch

WORK=Path(__file__).resolve().parent
SPEC=importlib.util.spec_from_file_location('context_analysis',WORK/'analyze_tree_sampling_calibration.py')
A=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(A)

def snapshot(tokens,response,positions=None,old=None,sampled=None,loss=None,weights=None,attention=None):
    positions=list(range(len(tokens))) if positions is None else positions
    value=dict(input_ids=torch.tensor([tokens]),responses=torch.tensor([response]),
               position_ids=torch.tensor([positions]),attention_mask=torch.tensor([attention or [1]*len(tokens)]),
               old_log_probs=torch.tensor([old or [-2.25]*len(response)],dtype=torch.float64))
    if sampled is not None:value['sampled_log_probs']=torch.tensor([sampled],dtype=torch.float64)
    if loss is not None:value['loss_mask']=torch.tensor([loss])
    if weights is not None:value['policy_weights']=torch.tensor([weights],dtype=torch.float64)
    return value

def main():
    checks=[]
    raw=snapshot([1,2,3],[3],old=[-2.125],sampled=[-2.])
    final=snapshot([1,2,3],[3],old=[-2.25],loss=[1],weights=[1.])
    index=A.RawIndex();index.add_snapshot(raw,'synthetic-raw')
    result=A.analyze_targets(index,[('synthetic-final',final)])
    assert result['exact_context_matched_token_occurrences']==1
    assert result['final_leaf_old_minus_actual_raw_sampled']['signed_range_over_all_exact_matches']==[-.25,-.25]
    assert index.report()['raw_same_turn_actor_minus_sampled']['signed_min']==-.125
    checks.append('identical_causal_context_and_actual_logprob_differences')
    retoken=snapshot([1,2,33],[33],loss=[1],weights=[1.])
    assert A.analyze_targets(index,[('retokenized',retoken)])['exact_context_matched_token_occurrences']==0
    checks.append('retokenized_current_token_not_position_aligned')
    clipped=snapshot([2,3],[3],positions=[1,2],loss=[1],weights=[1.])
    assert A.analyze_targets(index,[('clipped-prefix',clipped)])['exact_context_matched_token_occurrences']==0
    checks.append('clipped_prior_prefix_not_suffix_aligned')
    raw_obs=snapshot([1,2,90,3],[3],old=[-2.125],sampled=[-2.])
    final_obs=snapshot([1,2,90,3],[90,3],old=[-9.,-2.25],loss=[0,1],weights=[0.,1.])
    observation_index=A.RawIndex();observation_index.add_snapshot(raw_obs,'raw-after-tool-observation')
    observed=A.analyze_targets(observation_index,[('final-with-loss-masked-observation',final_obs)])
    assert observed['effective_supervised_token_occurrences']==observed['exact_context_matched_token_occurrences']==1
    assert observed['matched_examples'][0]['prefix_length']==3
    checks.append('loss_mask_zero_observation_still_in_subsequent_causal_prefix')
    changed_position=snapshot([1,2,3],[3],positions=[1,1,2],loss=[1],weights=[1.])
    assert A.analyze_targets(index,[('changed-prior-position',changed_position)])['exact_context_matched_token_occurrences']==0
    checks.append('prior_position_change_even_with_identical_current_position_refused')
    duplicate=snapshot([1,2,3],[3],old=[-2.375],sampled=[-2.5])
    index.add_snapshot(duplicate,'second-raw-occurrence')
    result=A.analyze_targets(index,[('final-ambiguous-raw-occurrence',final)])
    assert index.report()['duplicate_raw_keys']==1 and result['matched_examples'][0]['raw_occurrence_count']==2
    assert result['final_leaf_old_minus_actual_raw_sampled']['signed_range_over_all_exact_matches']==[-.25,.25]
    assert result['final_leaf_old_minus_actual_raw_sampled']['matched_weight_mean_signed_bounds']==[-.25,.25]
    checks.append('duplicate_sampler_values_keep_range_count_without_sample_average')
    leaves={key:torch.cat([value,value]) for key,value in final.items()}
    repeated=A.analyze_targets(index,[('two-shared-prefix-leaves',leaves)])
    assert repeated['leaf_rows_denominator']==2 and repeated['duplicated_final_context_token_occurrences']==1
    assert repeated['tree_weights']['all_effective_weight_mass']==2
    assert repeated['final_leaf_old_minus_actual_raw_sampled']['weighted_signed_numerator_bounds']==[-.5,.5]
    checks.append('shared_leaf_prefixes_keep_original_tree_weight_and_row_denominator')
    raw_pad=snapshot([0,1,2,3,0],[3,0],positions=[99,0,1,2,88],old=[-2.125,float('nan')],
                     sampled=[-2.,float('nan')],attention=[0,1,1,1,0])
    final_pad=snapshot([1,2,3,0],[3,0],old=[-2.25,float('nan')],loss=[1,1],weights=[1.,float('nan')],attention=[1,1,1,0])
    pad_index=A.RawIndex();pad_index.add_snapshot(raw_pad,'left-and-right-padded')
    padded=A.analyze_targets(pad_index,[('padding-excluded',final_pad)])
    assert padded['exact_context_matched_token_occurrences']==1 and padded['tree_weights']['invalid_effective_weights']==0
    assert pad_index.report()['raw_same_turn_actor_minus_sampled']['invalid_paired_occurrences']==0
    checks.append('attention_padding_and_padded_nan_not_in_context_or_weighted_statistics')
    with tempfile.TemporaryDirectory(prefix='sampling-analysis-cpu-',dir=WORK) as temporary:
        root=Path(temporary);tensor_file=root/'verified.pt';torch.save(raw,tensor_file)
        digest=A.file_sha(tensor_file);assert torch.equal(A.load_verified(tensor_file,digest)['input_ids'],raw['input_ids'])
        tensor_file.write_bytes(b'changed')
        try:A.load_verified(tensor_file,digest)
        except ValueError:checks.append('snapshot_sha_mutation_refused_before_torch_load')
        else:raise AssertionError('SHA mismatch was admitted')
        run=root/'tree-safety300-999999';package_dir=root/'package';package_dir.mkdir()
        (run/'output/probability-calibration').mkdir(parents=True);(run/'output/entry-audit').mkdir()
        config=b'synthetic test configuration';(run/'resolved.yaml').write_bytes(config);(run/'output/entry-audit/entry-config.yaml').write_bytes(config)
        parent=root/'parent-manifest.json';parent_identity={'completed_step':110};parent.write_text(json.dumps({'identity':parent_identity}))
        package=dict(stage='calibration110',probability_calibration={'environment':{'TREE_PPO_CALIBRATION_ONLY':'1'}},
                     code_commit='a'*40,common_data_identity='d'*64,model_identity={'sha256':'e'*64},files={},
                     parent_manifest=dict(path=str(parent),sha256=A.file_sha(parent),identity=parent_identity))
        for name in A.SOURCE_FILES:
            path=package_dir/'source/repo'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('synthetic source fixture\n')
            package['files']['source/repo/'+name]=A.file_sha(path)
        manifest=package_dir/'manifest.json';manifest.write_text(json.dumps(package));package_sha=A.file_sha(manifest)
        ready=dict(status='CALIBRATION_OBSERVED_ZERO_WRITES',parent_completed_outer=110,observed_outer=111,
                   new_completed_outer=0,optimizer_committed=0,scheduler_advanced=False,job_id='999999',
                   code_commit='a'*40,config_sha256=A.file_sha(run/'resolved.yaml'),data_identity='d'*64,model_identity='e'*64,
                   workers=[{'rank':rank} for rank in range(8)])
        ready_path=run/'output/probability-calibration/ready.json';ready_path.write_text(json.dumps(ready));ready_sha=A.file_sha(ready_path)
        (run/'output/recovery-loaded.json').write_text(json.dumps({'provenance':{'parent_identity':parent_identity,'parent_cursor':57600,'exact_executor_resume':False}}))
        A.verify_bundle(run,package_dir,package_sha,ready_sha,require_node=False)
        checks.append('package_ready_config_parent_source_path_sha_binding')
        identity=dict(job_id='999999',run_id=run.name,code_commit='a'*40,config_sha256=ready['config_sha256'],
                      data_identity='d'*64,model_identity='e'*64)
        sampling=run/'output/sampling-consistency';sampling.mkdir()
        workers=[]
        for rank in range(8):
            raw_payload=dict(raw,distribution={'temperature':1.})
            raw_path=sampling/f'rank-{rank:02d}-call-00001.pt';torch.save(raw_payload,raw_path)
            raw_json=sampling/f'rank-{rank:02d}-call-00001.json'
            raw_record=dict(**identity,rank=rank,world_size=8,scheduler_step=110,optimizer_committed=0,
                saved_rows=1,local_rows=1,enforced=False,generation_call=1,snapshot_path=str(raw_path),snapshot_sha256=A.file_sha(raw_path),
                distribution={'temperature':1.},decision={'allowed':False,'reasons':['trial_numeric_observation']})
            raw_json.write_text(json.dumps(raw_record))
            monitor_path=run/'output/probability-calibration'/f'rank-{rank:02d}-monitor.pt'
            torch.save({key:value.repeat(32,1) for key,value in final.items()},monitor_path)
            workers.append(dict(**identity,rank=rank,world_size=8,optimizer_committed=0,scheduler_advanced=False,
                scheduler_step=110,restored_optimizer_steps=[2500],diagnostic_records=[str(raw_json)],
                diagnostic_entries=[dict(path=str(raw_json),sha256=A.file_sha(raw_json),snapshot_path=str(raw_path),snapshot_sha256=A.file_sha(raw_path))],
                monitor_path=str(monitor_path),monitor_sha256=A.file_sha(monitor_path)))
        ready['workers']=workers;ready_path.write_text(json.dumps(ready));ready_sha=A.file_sha(ready_path)
        actor_dir=run/'output/trajectories/outer-000111';actor_dir.mkdir(parents=True)
        actor_path=actor_dir/'actor-inputs.pt'
        actor_identity=dict(outer_step=111,job_id='999999',source='a'*40,config=ready['config_sha256'],data='d'*64)
        torch.save(dict(identity=actor_identity,rows=3840,tensors={key:value.repeat(3840,1) for key,value in final.items()}),actor_path)
        (actor_dir/'manifest.json').write_text(json.dumps(dict(status='ALL_ROLLOUTS_AND_ACTOR_INPUTS_PUBLISHED',
            identity=actor_identity,rows=3840,payloads={'actor-inputs.pt':A.file_sha(actor_path)})))
        verified=A.verify_bundle(run,package_dir,package_sha,ready_sha,require_node=False)
        analysis=A.analyze_bundle(*verified,package_sha,ready_sha,2)
        assert analysis['monitor_subset']['leaf_rows_denominator']==256
        assert analysis['complete_actor_inputs']['leaf_rows_denominator']==3840
        assert analysis['complete_actor_inputs']['exact_context_matched_token_occurrences']==3840
        assert analysis['raw_sampling_evidence']['raw_attention_valid_response_token_occurrences']==8
        assert analysis['complete_actor_inputs']['tree_weights']['all_effective_weight_mass']==3840
        checks.append('complete_eight_rank_bundle_wiring_keeps_monitor_subset_separate_from_full3840_leaves')
        extreme=snapshot([1,2,3],[3],old=[-1.7e308],sampled=[-1.7e308])
        extreme_index=A.RawIndex();extreme_index.add_snapshot(extreme,'extreme-raw')
        extreme_final=snapshot([1,2,3],[3],old=[1.7e308],loss=[1],weights=[1.])
        overflow=A.analyze_targets(extreme_index,[('extreme-final',extreme_final)])
        assert overflow['finite_numeric_evidence']['invalid_sampled_difference_occurrences']==1
        assert overflow['finite_numeric_evidence']['invalid_actor_difference_occurrences']==1
        assert overflow['finite_numeric_evidence']['matched_tokens_with_finite_raw_sampled_and_final_old']==0
        assert overflow['final_leaf_old_minus_actual_raw_sampled']['signed_range_over_all_exact_matches'] is None
        json.dumps(overflow,allow_nan=False)
        checks.append('finite_operand_difference_overflow_explicit_invalid_finite_json')
        huge_weights=snapshot([1,2,3,4],[3,4],old=[-2.25,-2.25],loss=[1,1],weights=[1.7e308,1.7e308])
        overflow=A.analyze_targets(index,[('huge-weight-sum',huge_weights)])
        assert 'total_weight' in overflow['finite_numeric_evidence']['nonfinite_aggregate_fields']
        assert overflow['tree_weights']['all_effective_weight_mass'] is None
        json.dumps(overflow,allow_nan=False)
        checks.append('finite_weight_aggregate_overflow_explicit_null_not_zero_division')
        source_file=package_dir/'source/repo'/A.SOURCE_FILES[0];source_file.write_text('changed source')
        try:A.verify_bundle(run,package_dir,package_sha,ready_sha,require_node=False)
        except ValueError:checks.append('diagnostic_source_mutation_refused')
        else:raise AssertionError('Source mutation was admitted')
    assert not torch.cuda.is_initialized()
    result=dict(status='SAMPLING_CAUSAL_CONTEXT_CPU_TESTS_PASS',checks=checks,checks_count=len(checks),
                helper_sha256=A.file_sha(WORK/'analyze_tree_sampling_calibration.py'),test_sha256=A.file_sha(__file__),
                cuda_initialized=False,actual_model_assets_loaded=False,remote_calls=False,
                scope='Small fabricated CPU tensors and metadata fixtures only; no actual calibration, training or policy acceptance')
    print(json.dumps(result,sort_keys=True,allow_nan=False))

if __name__=='__main__':main()
