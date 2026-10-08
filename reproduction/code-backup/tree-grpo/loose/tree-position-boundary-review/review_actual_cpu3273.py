"""Validate saved actual3273 metadata/report bytes only, never .pt/model/remote."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent;WORK=HERE.parent;BASE=WORK.parent/'outputs/tree-instruct-300'
LOCAL_RUN=BASE/'healthy110-3268';REMOTE=Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
RUN=REMOTE/'runs/tree-safety300-3268';PACKAGE=REMOTE/'instruct300/candidates/safety110-a632ba9-84e3e186'
LOCAL_PACKAGE=WORK/'tree-safety110-a632ba9-v2'
EXPECTED_PACKAGE='84e3e18682afa880f1053a0025eda595fca332d550ec1f6905bc1334049b5a48'
EXPECTED_READY='ba948bdec0b71a3c8ff5c61db1360840134e57fccaf71fb154449f30c300c468'
POSITION='3992c092e7cf4e42469b9e2cf7cab896a44395c70f224101ef37cb4c0fee05c9'
ACTION='b63740191fcfea587756526c9cb2186557c28233e43886b5de23dedb0d191bd1'
CONTEXT='28e665eb822651ff44559b2e9c6110d01542c5e6433db8dd4d07c659af7131ac'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(path):
    path=Path(path)
    assert path.suffix!='.pt','Actual PT access forbidden'
    value=json.loads(path.read_bytes(),parse_constant=lambda text:(_ for _ in ()).throw(ValueError(text)))
    def finite(v):
        if isinstance(v,float):assert math.isfinite(v)
        elif isinstance(v,dict):
            for x in v.values():finite(x)
        elif isinstance(v,list):
            for x in v:finite(x)
    finite(value);return value
capture_path=BASE/'PPO_LAYOUT_CPU_JOB3273_PROGRESS1_RAW.json';capture=read(capture_path)
assert capture['job_id']==3273
control=capture['commands']['scontrol show job 3273'];assert control['rc']==0 and control['stderr']==''
slurm=control['stdout']
for pattern in [r'JobId=3273\b',r'JobName=tree-position-actions-3268-v2\b',r'JobState=COMPLETED\b',r'ExitCode=0:0\b',
    r'NodeList=h100-9\b',r'NumCPUs=4\b',r'TresPerNode=gres:gpu:0\b',r'RunTime=00:00:15\b',r'TimeLimit=00:15:00\b']:
    assert re.search(pattern,slurm),pattern
dispatch=read(BASE/'PPO_LAYOUT_CPU_DISPATCH_V2_RAW.json');intent=read(BASE/'PPO_LAYOUT_CPU_DISPATCH_V2_INTENT.json')
assert dispatch['status']=='SUBMITTED' and dispatch['job_id']==intent['job_id']==3273
assert dispatch['returncode']==0 and dispatch['stdout']=='3273\n' and dispatch['stderr']==''
assert intent['source_sha256']=='6d3532a7bc54992f84be375d0a4cdfe4dfd5172d2dfb31db68e7a3b459ea454d'
assert intent['raw_sha256']==sha(BASE/'PPO_LAYOUT_CPU_DISPATCH_V2_RAW.json')
assert dispatch['previous_terminal_evidence_scope']=='SAVED_VERIFIED_SUCCESSFUL_TERMINAL_LIVE_JOB_RECORD_PURGED'
assert dispatch['previous_live_scontrol']==dict(rc=1,stdout='',stderr='slurm_load_jobs error: Invalid job id specified\n')
assert dispatch['gpu_count']==0 and dispatch['cpu_count']==4 and dispatch['memory_gib']==64 and dispatch['time_minutes']==15
reports={};report_proofs=[]
for name in ['position-analysis-3273.json','retained-actions-3273.json','layout-analysis-3273.err']:
    path=BASE/'cpu3273'/name;raw=path.read_bytes();record=capture['reports'][name]
    assert len(raw)==record['bytes'] and hashlib.sha256(raw).hexdigest()==record['sha256']
    report_proofs.append(dict(path=str(path.resolve()),sha256=record['sha256'],bytes=len(raw)))
    if name.endswith('.err'):assert not raw
    else:
        value=read(path);assert value==record['result'];reports[name]=value
P=reports['position-analysis-3273.json'];A=reports['retained-actions-3273.json']
package=read(LOCAL_PACKAGE/'manifest.json');ready=read(LOCAL_RUN/'output/probability-calibration/ready.json')
assert sha(LOCAL_PACKAGE/'manifest.json')==P['package_manifest_sha256']==dispatch['package_sha256']==EXPECTED_PACKAGE
assert sha(LOCAL_RUN/'output/probability-calibration/ready.json')==P['ready_sha256']==dispatch['ready_sha256']==EXPECTED_READY
assert P['helper_sha256']==POSITION and P['context_helper_sha256']==CONTEXT and A['helper_sha256']==ACTION
assert P['status']=='POSITION_BOUNDARY_EVIDENCE_ANALYZED_NO_TRAINING_ACCEPTANCE'
assert A['status']=='ACTUAL_RETAINED_ACTION_RAW_PREFIX_CLASSIFICATION_ONLY'
identity=dict(job_id='3268',run_id=RUN.name,code_commit=package['code_commit'],config_sha256=ready['config_sha256'],
    data_identity=package['common_data_identity'],model_identity=package['model_identity']['sha256'])
assert P['identity']==identity and P['actual_model_forwards']==P['actual_optimizer_writes']==0
assert P['analysis_cost']['analysis_slurm_job_id']=='3273' and P['analysis_cost']['cuda_initialized'] is False
assert P['analysis_cost']['threads']==4 and P['actor_input_rows']==3840
assert P['package_versions']['transformers']=='4.46.3'
assert P['hf_rule_source']['sha256']=='2753ed0bdb9b37f72a4b4d0663e2f2833ad55a874be37d4599d69eb9a826ede0'
native=P['native_caller_source'];assert native['status']=='NATIVE_HF_SP1_CALLER_SOURCE_BOUND_NOT_RUNTIME_TRACE'
assert native['qwen_sha256']=='4568d588b45dd28152d97c37300a5e227589bc70b9fef5224af29977ecf6abdc'
assert native['worker_sha256']==package['files']['source/repo/verl/workers/fsdp_workers.py']
assert native['position_ids_keyword'] is True and native['actor_monkey_patch_at_sp1'] is False and native['live_worker_function_identity_observed'] is False
assert P['grouping_config']==dict(actor_leaf_rows=3840,config_sha256=identity['config_sha256'],global_actor_micro=16,global_mini=384,
    global_old_micro=32,monitor_rows_per_rank=32,ppo_epochs=1,rank_actor_micro=2,rank_mini=48,rank_mini_divisible_by_both_micros=True,
    rank_old_micro=4,rollout_n=1,sequence_parallel=1,source_train_rows=640,tensor_parallel=1,use_remove_padding=True,world_size=8)

assert len(P['source_files'])==14
for entry in P['source_files']:
    path=Path(entry['path'])
    if path.is_relative_to(PACKAGE):
        relative=str(path.relative_to(PACKAGE));assert package['files'][relative]==entry['sha256']
        assert sha(LOCAL_PACKAGE/relative)==entry['sha256']
    else:
        assert path==RUN/'output/recovery-loaded.json'
        assert sha(LOCAL_RUN/'output/recovery-loaded.json')==entry['sha256']

assert len(P['evidence'])==73
raw_expected={};monitor_expected={}
for worker in ready['workers']:
    rank=worker['rank'];monitor_expected[worker['monitor_path']]=worker['monitor_sha256']
    for entry in worker['diagnostic_entries']:
        raw_expected[entry['path']]=entry
        local_json=LOCAL_RUN/Path(entry['path']).relative_to(RUN)
        assert sha(local_json)==entry['sha256']
raw_evidence=[x for x in P['evidence'] if 'snapshot_path' in x]
assert len(raw_evidence)==len(raw_expected)==64
for entry in raw_evidence:
    expected=raw_expected[entry['path']]
    assert all(entry[k]==expected[k] for k in ['sha256','snapshot_path','snapshot_sha256'])
monitor_evidence=[x for x in P['evidence'] if x['path'] in monitor_expected]
assert len(monitor_evidence)==8
for entry in monitor_evidence:assert entry['sha256']==monitor_expected[entry['path']]
manifest_path=LOCAL_RUN/'output/trajectories/outer-000111/manifest.json';manifest=read(manifest_path)
actor_evidence=P['evidence'][-1];assert actor_evidence['manifest_sha256']==sha(manifest_path)==A['manifest_sha256']
assert actor_evidence['sha256']==manifest['payloads']['actor-inputs.pt']
assert A['trajectories_sha256']==manifest['payloads']['trajectories.jsonl.gz']
assert A['identity']==manifest['identity'] and A['analysis_slurm_job_id']=='3273'

population={};seen=set();tensor_identities=[]
for item in P['inspections']:
    kind,rank=item['kind'],item['rank'];inspection=item['inspection'];label=inspection['label']
    key=(kind,rank,item.get('generation_call'));assert key not in seen;seen.add(key)
    current=population.setdefault(kind,dict(parts=0,ranks=set(),rows=Counter(),local_grouping={'4':Counter(),'2':Counter()}))
    current['parts']+=1;current['ranks'].add(rank)
    if kind=='raw_same_turn':
        expected=raw_expected[str(Path(label).with_suffix('.json'))]
        assert expected['snapshot_path']==label
        raw_meta=read(LOCAL_RUN/Path(expected['path']).relative_to(RUN))
        assert raw_meta['rank']==rank and raw_meta['generation_call']==item['generation_call']
        assert raw_meta['saved_rows']==inspection['rows']['rows']
    elif kind=='final_leaf_monitor_subset':
        assert label in monitor_expected and inspection['rows']['rows']==32
    else:
        assert kind=='complete_final_actor_inputs_dp_partition' and label==actor_evidence['path'] and inspection['rows']['rows']==480
    tensors=inspection['full_input_tensor_identity'];assert set(tensors)=={'input_ids','attention_mask','position_ids'}
    shapes=[x['shape'] for x in tensors.values()];assert shapes[0]==shapes[1]==shapes[2] and shapes[0][0]==inspection['rows']['rows']
    for name,value in tensors.items():
        assert value['dtype']=='torch.int64' and re.fullmatch('[0-9a-f]{64}',value['shape_and_bytes_sha256'])
        tensor_identities.append(dict(kind=kind,rank=rank,generation_call=item.get('generation_call'),tensor=name,**value))
    row=inspection['rows'];assert row['attention_valid_tokens']>0
    for k,v in row.items():
        if isinstance(v,int):current['rows'][k]+=v
    assert all(row[k]==0 for k in ['empty_rows','negative_valid_position_rows','noncontiguous_valid_position_rows','nonzero_start_position_rows'])
    assert 'hypothetical post-balance grouping' in inspection['grouping_scope'] and 'prospective actor grouping' in inspection['grouping_scope']
    for micro in ['4','2']:
        stats=inspection['local_grouping'][micro];assert stats['micro']==int(micro)
        assert stats['groups']==math.ceil(row['rows']/int(micro))
        assert stats['packed_varlen_groups']+stats['single_sequence_groups']+stats['zero_token_groups']==stats['groups']
        assert all(stats[k]==0 for k in ['attention_valid_tokens_in_crossing_segments','extra_boundaries','groups_missing_start_zero',
            'groups_with_boundary_mismatch','groups_with_possible_cross_sample_attention','groups_with_uncovered_prefix',
            'missing_boundaries','missing_sample_boundaries','possible_cross_sample_boundaries','zero_token_groups'])
        for k,v in stats.items():
            if k!='micro' and isinstance(v,int):current['local_grouping'][micro][k]+=v
for value in population.values():value['ranks']=sorted(value['ranks']);assert value['ranks']==list(range(8))
assert len(P['inspections'])==80
assert population['raw_same_turn']['parts']==64 and population['raw_same_turn']['rows']['rows']==11528
assert population['final_leaf_monitor_subset']['rows']['rows']==256 and population['complete_final_actor_inputs_dp_partition']['rows']['rows']==3840
assert population['raw_same_turn']['rows']['attention_valid_tokens']==15868958
assert population['final_leaf_monitor_subset']['rows']['attention_valid_tokens']==466909
assert population['complete_final_actor_inputs_dp_partition']['rows']['attention_valid_tokens']==7041593
counts=A['counts'];assert counts['leaf_rows']==3840 and counts['records']==15346
assert counts['retained_action_token_occurrences']==4237098
assert counts['own_record_exact_causal_prefix_token_occurrences']+counts['own_record_nonmatching_token_occurrences']==4237098
assert counts['generation_vs_final_prefix_equal_records']==15346 and counts['generation_vs_final_prefix_different_records']==0
assert math.isclose(counts['leaf_weight_mass'],3840,abs_tol=1e-9)
assert A['own_record_exact_token_coverage']==counts['own_record_exact_causal_prefix_token_occurrences']/4237098
assert A['own_record_equal_leaf_weight_coverage']==counts['own_record_exact_context_weight_mass']/3840
context=read(BASE/'cpu3271/context-analysis-3271.json')
assert context['identity']==identity and context['helper_source_sha256']==CONTEXT
assert context['complete_actor_inputs']['effective_supervised_token_occurrences']==4237098
additional=context['complete_actor_inputs']['exact_context_matched_token_occurrences']-counts['own_record_exact_causal_prefix_token_occurrences']
assert additional==38
cost=dict(cpu_job_elapsed_seconds=15,reserved_cpu_core_seconds=60,gpu_reserved_seconds=0,position=P['analysis_cost'],
    retained_action={k:A[k] for k in ['wall_seconds','process_cpu_seconds','peak_rss_bytes']},
    instrumented_main_wall_seconds=P['analysis_cost']['wall_seconds']+A['wall_seconds'],
    instrumented_process_cpu_seconds=P['analysis_cost']['process_cpu_seconds']+A['process_cpu_seconds'],
    scope='Instrumented main excludes process import/startup, inline verifier and scheduler; neither sum is exact whole-job CPU consumption.')
proof=dict(status='ACTUAL_CPU3273_POSITION_AND_ACTION_METADATA_REPORTS_VERIFIED_NO_TRAINING_ACCEPTANCE',
    job_id=3273,slurm_state='COMPLETED',slurm_exit_code='0:0',host='h100-9',gpu_count=0,cpu_count=4,
    captured_epoch=capture['epoch'],capture_sha256=sha(capture_path),report_bytes=report_proofs,identity=identity,
    package_manifest_sha256=EXPECTED_PACKAGE,ready_sha256=EXPECTED_READY,helper_sha256=POSITION,action_helper_sha256=ACTION,
    context_helper_sha256=CONTEXT,installed_hf_flash_helper_sha256=P['hf_rule_source']['sha256'],native_qwen_sha256=native['qwen_sha256'],
    source_files_bound=14,raw_monitor_full_asset_metadata_bound=73,full_tensor_identity_records_checked=len(tensor_identities),
    population=population,retained_action_counts=counts,retained_action_token_coverage=A['own_record_exact_token_coverage'],
    retained_action_equal_leaf_weight_coverage=A['own_record_equal_leaf_weight_coverage'],
    global_context_union_additional_matching_tokens_over_own_record=additional,cost=cost,
    actual_model_forwards=0,actual_optimizer_writes=0,probability_consistency_accepted=False,thresholds_calibrated=False,long_training_ready=False,
    reviewer_actual_pt_reads=0,reviewer_remote_calls=0,
    scope='Already-saved actual same-node CPU reports and metadata bytes independently verified. Recorded positions/source-reconstructed boundaries show no anomalies; this does not prove live GPU FlashAttention identities, cause of probability error, sampled-policy correction or training acceptance. Three populations stay separate; full post-balance micro4 hypothetical and micro2 prospective, no GPU update.')
(HERE/'ACTUAL_CPU3273_REPORT_PROOF.json').write_text(json.dumps(proof,sort_keys=True,indent=2,allow_nan=False)+'\n')
print(json.dumps(proof,sort_keys=True,allow_nan=False))
