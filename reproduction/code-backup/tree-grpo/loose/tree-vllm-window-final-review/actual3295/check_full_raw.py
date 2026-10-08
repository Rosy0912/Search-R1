"""Saved27 owned text/JSON records only; no tensor assets/imports/remote/GPU."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math

REVIEW=Path(__file__).resolve().parent
ROOT=REVIEW.parents[2]
RAW=ROOT/'outputs/tree-instruct-300/PPO_WINDOW_GPU3295_ACTUAL_RAW.json'
def digest(b):return hashlib.sha256(b).hexdigest()
def sha(p):return digest(Path(p).read_bytes())
data=RAW.read_bytes()
assert digest(data)=='30deb240d598b4880732d8eae579bfdcc5a9e53fab4363c17cc2319fec244010' and len(data)==380729451
raw=json.loads(data);del data
assert raw['job_id']==3295 and raw['host']=='h100-9' and raw['mutation'] is raw['tensor_assets_read'] is False
rows=raw['source_records'];assert len(rows)==len({x['path'] for x in rows})==27
for row in rows:
    data=row['text'].encode();assert len(data)==row['bytes'] and digest(data)==row['sha256'];del data
    assert Path(row['path']).suffix in {'.json','.jsonl','.log','.out'}
core={Path(x['path']).name:x for x in rows if Path(x['path']).name not in ['ready.json','sampling-params-transition.json']}
result=json.loads(core['result.json']['text']);bundle=json.loads(core['bundle.json']['text']);launch=json.loads(core['launch.json']['text'])
snapshot=ROOT/'outputs/tree-instruct-300/PPO_WINDOW_GPU3295_METADATA_20261006T030125645415Z.json';s=json.loads(snapshot.read_bytes())
assert sha(snapshot)=='d5a4204e4eca434631351c7214e99bec23cc6f7563a97a9e6476698d461af37c'
assert {k:v for k,v in result.items() if k!='rank_records'}==s['result.json']['record']
assert core['result.json']['sha256']==s['result.json']['identity']['sha256']
assert len(result['rank_records'])==8 and sorted(x['rank'] for x in result['rank_records'])==list(range(8))
rank_records={x['rank']:x for x in result['rank_records']};snapshot_ready={x['record']['rank']:x for x in s['ready']}
wanted_anchors={a['id']:(a['rank'],a['input_row'],a['response_offset']) for a in bundle['window_inputs']['anchors']};assert len(wanted_anchors)==9
rank_proofs=[];anchor_proofs=[];emitted_total=0;native_calls_total=0
for rank in range(8):
    row=next(x for x in rows if x['path'].endswith(f'rank-{rank:02d}/ready.json'));v=json.loads(row['text']);rr=rank_records[rank]
    assert rr['path']==row['path'] and rr['sha256']==row['sha256'] and rr['record']==v and v['rank']==rank
    assert snapshot_ready[rank]['identity']=={k:row[k] for k in ['path','bytes','sha256']}
    for key,value in snapshot_ready[rank]['record'].items():
        if key not in ['hf_summary','observer_summary']:assert value==v[key],key
    hf=v['hf'];assert len(hf['logical_records'])==len(hf['serialized_to_unique'])==hf['serialized_count']==435
    assert len(set(hf['serialized_to_unique'].values()))==hf['unique_parameter_count']==434
    assert hf['original_3268_effective_identity_proven'] is False
    for key,rec in hf['logical_records'].items():
        assert rec['loaded_master']==rec['serialized'] and rec['loaded_master']['dtype']=='torch.float32'
        assert rec['expected_effective_bf16']['shape']==rec['loaded_master']['shape'] and rec['expected_effective_bf16']['dtype']=='torch.bfloat16'
        assert rec['unique_parameter_key']==hf['serialized_to_unique'][key]
    for item in v['live_window_weights']['pre']['records']:
        for fragment in item['fragments']:
            expected=hf['logical_records'][fragment['source']]
            assert fragment['source_master_identity']['sha256']==expected['loaded_master']['full_shape_dtype_bytes_sha256']
            assert fragment['source_master_identity']['dtype']==expected['loaded_master']['dtype'] and fragment['source_master_identity']['shape']==expected['loaded_master']['shape']
            assert fragment['expected_cast_identity']['sha256']==expected['expected_effective_bf16']['full_shape_dtype_bytes_sha256']
            assert fragment['expected_cast_identity']['shape']==expected['expected_effective_bf16']['shape'] and fragment['expected_cast_identity']['dtype']=='torch.bfloat16'
    calls=[x['native_forward_receipt'] for x in v['actor_calls'] if 'native_forward_receipt' in x]
    actor_anchors={x['anchor_id']:x for x in v['actor_calls'] if 'anchor_id' in x}
    assert len(calls)==80 and [x['call_index'] for x in calls]==list(range(80))
    for index,call in enumerate(calls):
        assert call['input_start']==index*4 and call['rows']==4 and call['original_actor_calls']==1
        assert call['original_native_input_object_preserved'] is call['rng_unchanged'] is True
        assert call['original_native_input_keys']==['responses','input_ids','attention_mask','position_ids']
        assert set(call['input_identity'])=={'responses','input_ids','attention_mask','position_ids','loss_mask'}
        mask=call['input_identity']['loss_mask'];assert mask['dtype']=='torch.bool' and mask['shape']==[4,500]
        expected=digest(b'torch.bool\0'+str((4,500)).encode()+b'\0'+bytes([1])*2000)
        assert mask['full_shape_dtype_bytes_sha256']==expected
        assert call['diagnostic_loss_mask_scope']=='Raw all-ones mapping AND original response attention; never passed to native'
        assert call['observed_forward_kwargs']==['attention_mask','input_ids','position_ids','use_cache']
        assert call['observed_model_training'] is call['backward_called'] is call['optimizer_instantiated'] is call['scheduler_instantiated'] is False
        assert call['observed_forward_input_shapes']['input_ids']==call['observed_forward_input_shapes']['position_ids']
        assert call['observed_forward_input_shapes']['input_ids'][0]==1
        assert call['observed_forward_input_dtypes']=={'input_ids':'torch.int64','position_ids':'torch.int64'}
    for label,rec in [('warmup',v['warmup']),('first_new_scoring',calls[0])]:
        weights=rec['effective_weights'];assert weights['logical_tensor_count']==len(weights['records'])==435 and weights['trace_completed_successfully'] is True
        assert set(weights['records'])==set(hf['logical_records']) and weights['original_3268_effective_weight_identity_proven'] is weights['acceptance'] is False
        for key,identity in weights['records'].items():
            expected=hf['logical_records'][key]['expected_effective_bf16']
            assert all(identity[k]==expected[k] for k in ['dtype','shape','full_shape_dtype_bytes_sha256']) and identity['device']==f'cuda:{rank}'
        tie=rec['same_window_tie'];assert tie['tied_alias_identity_proven'] is True and tie['historical_3268_identity_proven'] is False
        assert tie['records'] and all(x['same_window_alias'] is True and x['logical_names']==['model.embed_tokens.weight','lm_head.weight'] for x in tie['records'])
    assert v['warmup']['original_actor_calls']==v['warmup_native_forwards']==1 and v['warmup_historical_input_not_new_generation'] is True
    assert v['warmup']['rng_unchanged'] is True and v['warmup']['backward_called'] is v['warmup']['optimizer_instantiated'] is v['warmup']['scheduler_instantiated'] is False
    observer=v['observer'];assert observer['errors']==[] and observer['observer_evidence_complete'] is observer['hooks_restored'] is True
    assert len(observer['requests'])==len(observer['generated'])==observer['counts']['request_registration']==320
    assert sorted(x['input_row'] for x in observer['generated'])==list(range(320))
    count=0
    for generated in observer['generated']:
        request=observer['requests'][generated['request_id']]
        assert request['input_row']==generated['input_row'] and request['prompt_token_ids']==generated['prompt_token_ids']
        length=generated['true_response_length']
        assert 0<length<=500 and length==len(generated['token_ids'])==len(generated['logprobs']) and generated['finished'] is True and generated['completion_index']==0
        assert generated['finish_reason'] in ['stop','length'] and math.isfinite(generated['cumulative_logprob'])
        for token,logprobs in zip(generated['token_ids'],generated['logprobs']):assert str(token) in logprobs and math.isfinite(logprobs[str(token)]['logprob'])
        count+=length
    assert count==v['actual_new_emitted_tokens'];emitted_total+=count;native_calls_total+=len(calls)
    observed={a['id']:a for a in observer['anchors']}
    assert set(observed)==set(actor_anchors)=={key for key,coordinate in wanted_anchors.items() if coordinate[0]==rank}
    for measured in v['anchor_observations']:
        aid=measured['anchor_id'];native=observed[aid];actor=actor_anchors[aid]
        assert measured['status']=='SAME_CAUSAL_CONTEXT_LOGITS_OBSERVED' and native['status']=='observed'
        assert (rank,measured['input_row'],measured['response_offset'])==wanted_anchors[aid]
        assert actor['context_token_ids']==native['context_token_ids']==measured['actor_context_token_ids']==measured['vllm_context_token_ids']
        assert actor['context_positions']==native['context_positions']==measured['actor_context_positions']==measured['vllm_context_positions']
        assert actor['target_token']==native['target_token_id']==measured['actor_target_token_id']==measured['vllm_target_token_id']
        generated=next(x for x in observer['generated'] if x['input_row']==measured['input_row']);offset=measured['response_offset']
        assert offset<len(generated['token_ids']) and generated['token_ids'][offset]==measured['actor_target_token_id']
        assert generated['prompt_token_ids']+generated['token_ids'][:offset]==measured['actor_context_token_ids']
        assert generated['logprobs'][offset][str(generated['token_ids'][offset])]['logprob']==native['sampled_logprob']==measured['actual_selected_sampler_logprob']
        assert actor['native_logprob']==measured['actual_original_actor_native_logprob'] and actor['logits_identity']==measured['tensor_identities']['actor']
        anchor_proofs.append(dict(rank=rank,anchor_id=aid,source_coordinate=wanted_anchors[aid],context_length=len(measured['actor_context_token_ids']),status='Actual generated prefix, native input context and actor same-forward context/target/scalars/hash all match'))
    transition=next(x for x in rows if x['path'].endswith(f'rank-{rank:02d}/sampling-params-transition.json'))
    assert json.loads(transition['text'])==v['sampling_params_transition']
    assert v['sampling_params_transition']['allowed'] is True and v['sampling_params_transition']['changed_fields']==['output_kind']
    assert v['sampling_params_before']==v['sampling_params_transition']['before'] and v['sampling_params_after']==v['sampling_params_transition']['after']
    assert v['actual_active_method_sources'] and all(chain for chain in v['actual_active_method_sources'].values())
    assert v['tensor_evidence']['path'].endswith(f'rank-{rank:02d}/window.pt') and v['tensor_evidence']['bytes']>0
    rank_proofs.append(dict(rank=rank,ready_path=row['path'],ready_bytes=row['bytes'],ready_sha256=row['sha256'],HF_serialized435_verified=True,full_native80_receipts_verified=True,warmup_and_first_new_native435_weight_digests_match_HF_cast=True,raw_allones_mask80_hashes_verified_and_native_four_keys_preserved=True,generated_rows=320,emitted_tokens=count,PT_identity_only_not_read=v['tensor_evidence']))
    del v
assert emitted_total==808994 and native_calls_total==640 and len(anchor_proofs)==9
assert result['actual_generation_invocations']==8 and result['actual_actor_scoring_native_forwards']==640 and result['actual_optimizer_writes']==0
assert launch==s['launch.json']['record'] and core['launch.json']['sha256']==result['launch_sha256']
watch=[json.loads(line) for line in core['probe-scheduler-observation-3295.jsonl']['text'].splitlines() if line.strip()]
assert watch[-1]['terminal_state_observed'] is True and watch[-1]['scheduler_state']=='COMPLETED' and watch[-1]['stdout']==s['scheduler']['stdout']
guards=[json.loads(line) for line in core['resources.jsonl']['text'].splitlines() if line.strip()];assert len(guards)==82
for value in guards:
    assert value['job_id']=='3295' and not value.get('event') and len(value['gpus'])==8
    assert all(0<=g['foreign_memory_fraction']<=.15 and 0<=g['foreign_sm_upper']<=15 for g in value['gpus'])
process=[json.loads(line) for line in core['process-resources.jsonl']['text'].splitlines() if line.strip()];maxima={}
for value in process:
    assert value['adapter']['job_id']=='3295' and value['adapter']['original_background_processes_preserved'] is True
    for item in value['processes']:maxima[item['pid']]=max(maxima.get(item['pid'],0),(item['user_cpu_ticks']+item['system_cpu_ticks'])/item['hz'])
initial=REVIEW/'MEASUREMENT_RECEIPT.json';assert sha(initial)=='3fed2325d24b6a1efb91f1c77515aa4f84a304bb2dd15c1efd7ae75cf16e53d9'
firststderr=ROOT/'work/tree-vllm-window-native-input-repair3293/ACTUAL3295_CAPTURE_ATTEMPT1_STDERR.txt'
laststderr=ROOT/'work/tree-vllm-window-native-input-repair3293/ACTUAL3295_CAPTURE_ATTEMPT2_STDERR.txt'
assert firststderr.exists() and firststderr.stat().st_size>0 and laststderr.read_bytes()==b''
receipt=dict(status='ACTUAL3295_FULL27_RAW_BYTE_BINDING_AND_COMPLETE_NATIVE_WINDOW_METADATA_INDEPENDENTLY_VERIFIED_NO_PROBABILITY_ACCEPTANCE',reviewer_task='/root/tree_window_final_review',reviewed_at_utc=datetime.now(timezone.utc).isoformat(),job_id='3295',
    original_condensed_measurement_receipt_sha256=sha(initial),full_raw_path=str(RAW),full_raw_sha256=sha(RAW),full_raw_bytes=380729451,raw27_record_sizes_and_SHA_verified=True,
    original_result95925907_bytes_bound=True,all8_original_ready_bytes_bound=True,full_result_rank_records_equal_saved_ready=True,actual_measurement_execution_verified=True,
    full640_native_receipts_verified=True,full435_HF_serialized_to434_unique_records_verified=True,allrank_warmup_and_firstnew435_native_effective_weight_hashes_match_HF_BF16=True,
    allrank_native_four_key_original_object_receipts_and_diagnostic_allones_masks_verified=True,actual_generation_invocations=8,actual_original_prompt_rows_global=2560,actual_emitted_tokens=emitted_total,actual_scoring_native_forwards=640,actual_historical_warmup_forwards=8,actual_optimizer_writes=0,actual_backward_calls=0,actual_scheduler_instantiated=False,
    all2560_generated_request_prompt_rows_and_true_lengths_complete=True,raw_scalar_logprobs_finite=True,matched_same_context_anchor_count=9,anchors=anchor_proofs,rank_proofs=rank_proofs,
    actual_PT_tensor_evidence_identities_bound_only=True,actual_PT_or_model_assets_read=False,full_scalar_distribution_recomputed_from_PT=False,global_p99=None,
    raw_measurement_scope='Metadata confirms source-computed full raw distributions and direct nine-anchor scalars; no PT vectors/raw actor logprobs independently re-read or recomputed. No probability threshold calibrated or accepted.',
    first_reader_refusal_preserved_sha256=sha(firststderr),second_reader_empty_stderr_sha256=sha(laststderr),reader_scope='Audit reader only: first64MiB/file cap rejected95925907-byte result; secondinventory-bounded128MiB/file512MiB/total succeeded. Not a runtime resource/train/probability gate change.',
    reviewer_initial_checker_error='Initial inline checker incorrectly required raw envelope job_id string; actual envelope int3295 identified and structural identity check corrected. No receipt or measurement acceptance written by failing checker.',
    costs=dict(scheduler_allocated_GPU_hours=8*419/3600,scheduler_allocated_CPU_hours=192*419/3600,supervisor_wall_seconds=result['wall_seconds'],supervisor_GPU_hours=result['gpu_hours'],supervisor_allocated_CPU_hours=result['allocated_cpu_hours'],sampled_process_CPU_maxima_sum_seconds=sum(maxima.values()),sampled_process_sample_count=len(process),sampled_process_PID_count=len(maxima),complete_consumed_CPU_hours=None,sampled_consumption_scope='PID maxima from incomplete process snapshots; not allocation or complete exit accounting'),
    raw_resources82_guard_records_verified=True,terminal_watcher_completed_verified=True,cleanup_residual=result['cleanup_residual'],original8_background_preserved=result['original_background_pid_start_ticks_preserved'],
    reviewer_remote_access=False,reviewer_GPU_execution=False,CPU_suites_repeated=False,probability_accepted=False,thresholds_calibrated=False,training_accepted=False,GPU_capacity_proven=False,actual_loaded_binary_identity_proven=False,historical3268_mu_identity_proven=False,intent_spent_no_retry=True,
    source_records=[{key:value[key] for key in ['path','bytes','sha256']} for value in rows])
target=REVIEW/'FULL_RAW_MEASUREMENT_BINDING_RECEIPT.json';assert not target.exists()
target.write_text(json.dumps(receipt,sort_keys=True,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(receipt_sha256=sha(target),sampled_process_samples=len(process),PID_count=len(maxima),sampled_CPU_seconds=sum(maxima.values())),sort_keys=True))
