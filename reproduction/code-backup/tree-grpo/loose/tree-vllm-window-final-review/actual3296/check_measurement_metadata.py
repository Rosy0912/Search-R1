"""Saved28 owned text/JSON records only; no tensor assets/imports/remote/GPU."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math

REVIEW=Path(__file__).resolve().parent
ROOT=REVIEW.parents[2]
RAW=ROOT/'outputs/tree-instruct-300/PPO_WINDOW_GPU3296_ACTUAL_RAW.json'
def digest(b):return hashlib.sha256(b).hexdigest()
def sha(p):return digest(Path(p).read_bytes())
data=RAW.read_bytes()
assert digest(data)=='cb5589540b43c767de48bf4b7a483277ce658793a05918cb9a00e83743c4ffec' and len(data)==383665002
raw=json.loads(data);del data
assert raw['job_id']==3296 and raw['host']=='h100-9' and raw['mutation'] is raw['tensor_assets_read'] is False
rows=raw['source_records'];assert len(rows)==len({x['path'] for x in rows})==28
for row in rows:
    data=row['text'].encode();assert len(data)==row['bytes'] and digest(data)==row['sha256'];del data
    assert Path(row['path']).suffix in {'.json','.jsonl','.log','.out'}
core={Path(x['path']).name:x for x in rows if Path(x['path']).name not in ['ready.json','sampling-params-transition.json']}
result=json.loads(core['result.json']['text']);bundle=json.loads(core['bundle.json']['text']);launch=json.loads(core['launch.json']['text'])
snapshot=ROOT/'outputs/tree-instruct-300/PPO_WINDOW_GPU3296_METADATA_20261006T035548067459Z.json';s=json.loads(snapshot.read_bytes())
assert sha(snapshot)=='b79a17c25572d95f4d1306ff166d7cab4f96b07e0f77edca8a0c4923437b1fea'
assert {k:v for k,v in result.items() if k!='rank_records'}==s['result.json']['record']
assert core['result.json']['sha256']==s['result.json']['identity']['sha256']
assert len(result['rank_records'])==8 and sorted(x['rank'] for x in result['rank_records'])==list(range(8))
rank_records={x['rank']:x for x in result['rank_records']};snapshot_ready={x['record']['rank']:x for x in s['ready']}
wanted_anchors={a['id']:(a['rank'],a['input_row'],a['response_offset']) for a in bundle['window_inputs']['anchors']};assert len(wanted_anchors)==9
rank_proofs=[];anchor_proofs=[];emitted_total=0;native_calls_total=0
oldpath=ROOT/'outputs/tree-instruct-300/PPO_WINDOW_GPU3295_ACTUAL_RAW.json'
ob=oldpath.read_bytes();assert len(ob)==380729451 and digest(ob)=='30deb240d598b4880732d8eae579bfdcc5a9e53fab4363c17cc2319fec244010'
oldraw=json.loads(ob);del ob
boundary_proofs=[];sampled_token_total=0
boundary_names=['embedding']
for layer in range(36):boundary_names += ['layers.%02d.input_layernorm'%layer,'layers.%02d.post_attention_layernorm'%layer]
boundary_names+=['final_norm'];assert len(boundary_names)==74

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
    old=json.loads(next(x['text'] for x in oldraw['source_records'] if x['path'].endswith(f'rank-{rank:02d}/ready.json')))
    generated_by_row={x['input_row']:x for x in observer['generated']};old_by_row={x['input_row']:x for x in old['observer']['generated']}
    assert set(generated_by_row)==set(old_by_row)==set(range(320))
    for rowid in range(320):
        cur,past=generated_by_row[rowid],old_by_row[rowid]
        for field in ['prompt_token_ids','token_ids','true_response_length','logprobs','cumulative_logprob','finished','completion_index','finish_reason']:
            assert cur[field]==past[field],(rank,rowid,field)
        sampled_token_total+=cur['true_response_length']
    assert v['raw_probability_observations']==old['raw_probability_observations']
    assert v['anchor_observations']==old['anchor_observations']
    assert v['generation_invocations']==1 and v['original_prompt_rows']==320
    assert v['actor_scoring_native_forwards']==80 and v['actor_native_forwards']==81 and v['warmup_native_forwards']==1
    assert v['optimizer_committed']==0 and v['backward_called'] is v['optimizer_instantiated'] is v['scheduler_instantiated'] is v['scheduler_advanced'] is False
    assert v['original_master_state_unchanged'] is v['original_modes_restored'] is True
    assert v['probability_accepted'] is v['training_accepted'] is v['GPU_capacity_proven'] is v['actual_loaded_binary_identity_proven'] is False
    assert v['rng_scope']['outer_before']==v['rng_scope']['outer_after']
    assert v['rng_scope']['sampler_rng_unchanged_claim'] is False and v['rng_scope']['new_generation_not_historical_replay'] is True
    assert v['wrappers']['instance_hooks_restored'] is True and v['wrappers']['distribution_changed'] is False
    assert v['wrappers']['counts']['sync']==v['wrappers']['counts']['output']==1 and v['wrappers']['registered_requests']==320
    pre,post=v['live_window_weights']['pre'],v['live_window_weights']['post']
    assert pre['serialized_source_keys']==435 and pre['serialized_target_keys']==len(pre['records'])==255 and pre['all_parameter_values_match'] is True
    assert post['target_bytes_unchanged_since_live_postsync_snapshot'] is post['source_cast_all_values_match'] is post['same_window_tie'] is True
    assert len(post['metadata']['parameters'])==255 and len(post['metadata']['buffers'])==36
    assert v['live_window_weights']['buffers_compared'] is True
    local_ids={key for key,coord in wanted_anchors.items() if coord[0]==rank}
    points={x['anchor_id']:x for x in v['anchor_observations']};paired={x['anchor_id']:x for x in v['actual_normalized_boundary_observations']}
    assert set(points)==set(paired)==local_ids
    bd={label:v[field] for label,field in [('actor','actor_boundary_observer'),('vllm','native_boundary_observer')]}
    for backend,current in bd.items():
        assert current['backend']==backend and current['rank']==rank
        assert current['bound_model_fields']==current['actual_model_fields']=={'hidden_size':2048,'num_hidden_layers':36}
        assert current['observer_evidence_complete'] is current['hooks_restored'] is True and current['errors']==[]
        assert current['native_exception_seen'] is current['observer_reconstructed_residual'] is current['observer_replaced_native_outputs'] is current['actual_loaded_binary_identity_proven'] is False
        assert current['new_model_forwards']==0 and current['probability_accepted'] is current['training_accepted'] is False
        assert current['captured_vector_count']==current['maximum_vectors_per_rank']==74*len(local_ids) and current['maximum_vectors_global_per_backend']==666
        assert set(current['hook_calls'])==set(boundary_names) and set(current['hook_calls'].values())=={80 if backend=='actor' else v['wrappers']['counts']['execute']}
        assert set(current['records'])==local_ids
        prov=current['provenance'];assert prov['hf_config_identity']==bundle['window_inputs']['hf_config_file']
        assert prov['additional_model_forwards']==0 and prov['original_model_outputs_unmodified'] is True and prov['actual_loaded_binary_identity_proven'] is False
        assert set(prov['actual_bound_method_sources'])==set(boundary_names)
        for label,methods in prov['actual_bound_method_sources'].items():
            expected_sources={'forward':bundle['boundary_method_sources']['actor_embedding' if label=='embedding' else 'actor_norm']} if backend=='actor' else ({'forward':bundle['boundary_method_sources']['vllm_embedding']} if label=='embedding' else {'forward':bundle['boundary_method_sources']['vllm_custom_op'],'dispatched':bundle['boundary_method_sources']['vllm_norm']})
            assert set(methods)==set(expected_sources)
            for method,chain in methods.items():
                assert len(chain)==1 and {k:chain[0][k] for k in ['path','sha256']}==expected_sources[method]
        for chain in prov['hook_registration'].values():
            assert len(chain)==1 and {k:chain[0][k] for k in ['path','sha256']}==bundle['boundary_method_sources']['torch_module']
    for aid in local_ids:
        current=points[aid];comparison=paired[aid];nr=bd['vllm']['records'][aid];ar=bd['actor']['records'][aid]
        assert current['status']=='SAME_CAUSAL_CONTEXT_LOGITS_OBSERVED'
        context={'token_ids':current['vllm_context_token_ids'],'positions':current['vllm_context_positions'],'target_token_id':current['vllm_target_token_id']}
        context_sha=digest(json.dumps(context,sort_keys=True,separators=(',',':')).encode())
        for item in [nr,ar]:
            assert item['status']=='ACTUAL_NATIVE_NORMALIZATION_BOUNDARIES_OBSERVED' and item['captured_boundary_count']==74
            assert item['target_token_id']==current['vllm_target_token_id'] and item['context_length']==len(context['token_ids']) and item['context_identity_sha256']==context_sha
            assert item['request_id']==observed[aid]['request_id'] and item['response_offset']==current['response_offset'] and item['input_row']==current['input_row']
            assert item['embedding_and_normalization_outputs_only'] is True and item['residual_reconstructed'] is False
            assert set(item['boundary_tensor_identities'])==set(boundary_names)
        assert comparison['status']=='SAME_CAUSAL_CONTEXT_ACTUAL_NORMALIZED_BOUNDARIES_OBSERVED' and comparison['boundary_count']==74
        assert comparison['probability_accepted'] is comparison['training_accepted'] is comparison['unique_kernel_cause_proven'] is comparison['historical_extreme_reproduced'] is False
        boundary_rows= comparison['boundaries'];assert [x['boundary'] for x in boundary_rows]==boundary_names
        for item in boundary_rows:
            label=item['boundary'];assert item['count']==2048
            assert item['native_identity']==nr['boundary_tensor_identities'][label] and item['actor_identity']==ar['boundary_tensor_identities'][label]
            for key in ['native_identity','actor_identity']:
                assert item[key]['shape']==[2048] and item[key]['dtype']=='torch.bfloat16' and len(item[key]['full_shape_dtype_bytes_sha256'])==64
            assert all(math.isfinite(item[key]) and item[key]>=0 for key in ['max_abs','mean_abs','rms'])
            assert item['mean_abs']<=item['rms']+1e-15<=item['max_abs']+1e-15
            assert type(item['exact_equal']) is bool and item['exact_equal']==(item['native_identity']==item['actor_identity'])==(item['max_abs']==0)
        first=next((x['boundary'] for x in boundary_rows if not x['exact_equal']),None)
        assert first==comparison['first_nonidentical_observed_boundary']
        boundary_proofs.append(dict(anchor_id=aid,rank=rank,input_row=current['input_row'],response_offset=current['response_offset'],target_token_id=current['vllm_target_token_id'],context_length=len(context['token_ids']),context_identity_sha256=context_sha,actual_boundaries_verified=74,native_actor_identity_pair_count=74,first_nonidentical_observed_boundary=first,first_difference_max_abs=next(x['max_abs'] for x in boundary_rows if x['boundary']==first) if first else 0,embedding_exact_equal=boundary_rows[0]['exact_equal'],unique_kernel_cause_proven=False))
    rank_proofs[-1].update(full_generated_IDs_and_logprob_records_equal_3295=True,raw_stats_and9anchor_data_equal3295=True,original_four_key80_RNG_preserved=True,boundary_native_and_actor74peranchor_complete=True,native_boundary_hook_calls_each=v['wrappers']['counts']['execute'],actor_boundary_hook_calls_each=80,outer_rng_restored=True,new_generation_sampler_rng_not_claimed_unchanged=True)
    del old,v
assert emitted_total==808994 and native_calls_total==640 and len(anchor_proofs)==9
assert result['actual_generation_invocations']==8 and result['actual_actor_scoring_native_forwards']==640 and result['actual_optimizer_writes']==0
assert launch==s['launch.json']['record'] and core['launch.json']['sha256']==result['launch_sha256']
watch=[json.loads(line) for line in core['probe-scheduler-observation-3296.jsonl']['text'].splitlines() if line.strip()]
assert watch[-1]['terminal_state_observed'] is True and watch[-1]['scheduler_state']=='COMPLETED' and watch[-1]['stdout']==s['scheduler']['stdout']
guards=[json.loads(line) for line in core['resources.jsonl']['text'].splitlines() if line.strip()];assert len(guards)==82
for value in guards:
    assert value['job_id']=='3296' and not value.get('event') and len(value['gpus'])==8
    assert all(0<=g['foreign_memory_fraction']<=.15 and 0<=g['foreign_sm_upper']<=15 for g in value['gpus'])
process=[json.loads(line) for line in core['process-resources.jsonl']['text'].splitlines() if line.strip()];maxima={};exited_samples=[]
for value in process:
    assert value['adapter']['job_id']=='3296' and value['adapter']['original_background_processes_preserved'] is True
    for item in value['processes']:
        if set(item)=={'pid','exited'}:
            assert type(item['pid']) is int and item['exited'] is True
            exited_samples.append(dict(epoch=value['epoch'],record=item));continue
        assert all(type(item[k]) is int and item[k]>=0 for k in ['pid','user_cpu_ticks','system_cpu_ticks']) and item['hz']>0
        maxima[item['pid']]=max(maxima.get(item['pid'],0),(item['user_cpu_ticks']+item['system_cpu_ticks'])/item['hz'])

# Complete actual metadata scope, no asset/vector reads or native recompute.
assert sampled_token_total==808994 and len(boundary_proofs)==9
assert len(exited_samples)==3 and {x['record']['pid'] for x in exited_samples}=={1118899,1118869,1118917}
assert sha(ROOT/'work/tree_shared_owned_runtime_fixed.py')==json.loads((ROOT/'work/single-tp1-window-0f2aafd1-466df7c2/package-manifest.json').read_bytes())['external_files']['/dataset/shaoshitong/search-r1-rl/tree-grpo/tools/tree_shared_owned_runtime_fixed.py']
assert raw['journal']==json.loads((ROOT/'work/tree-vllm-window-boundaries3295/DISPATCH_REMOTE_RESULT.json').read_bytes())
assert raw['scheduler']==s['scheduler'] and 'JobState=COMPLETED' in raw['scheduler']['stdout'] and 'ExitCode=0:0' in raw['scheduler']['stdout'] and 'RunTime=00:06:59' in raw['scheduler']['stdout']
assert result['diagnostic_body_complete'] is result['final_guard_healthy'] is result['original_background_pid_start_ticks_preserved'] is True
assert result['cleanup_residual']=={} and result['probability_consistency_accepted'] is result['actual_training_accepted'] is result['gpu_training_capacity_proven'] is False
assert result['runner_commit']=='0f2aafd1ef38221de0a6bb93d0a78c6b0bfedc84' and result['package_manifest_sha256']=='89cba7b35bca1163f1542461dc767e6790f54a57aa0c44aa267d7c2462e2fb65'
assert result['release_sha256']=='0bde62135e38569ea90a84c4c865ffb832f67109dbc5f49986d40b0bba166d9f'
assert result['actual_generation_invocations']==8 and result['actual_actor_scoring_native_forwards']==640 and result['actual_new_emitted_tokens']==808994 and result['actual_optimizer_writes']==0
assert launch['bundle_payload_sha256']=='d355e8764e7988cb4a775324bbccf437e2e714be44eb8a19bd9ca5b59f624d23'
assert launch['actual_loaded_binary_identity_proven'] is False
assert len(launch['gpu_uuids'])==len(set(launch['gpu_uuids']))==len(launch['leases'])==8 and {x['uuid'] for x in launch['leases']}==set(launch['gpu_uuids'])
protected={'1680782':1502352181,'1680783':1502352181,'1680784':1502352181,'1680785':1502352181,'1696443':1502398762,'1696444':1502398762,'1696445':1502398762,'1696446':1502398762}
assert launch['background_pid_start_ticks']==protected
initial=json.loads(core['initial-allocation.json']['text']);assert initial['background_pid_start_ticks']==protected and initial['job_id']=='3296' and initial['allocated_cpus']==192 and initial['allocated_gpus']==8 and initial['requested_cpus']==32
assert set(int(x.split(',')[1].strip()) for x in result['final_gpu']['processes'])==set(int(x) for x in protected)
assert all({x['uuid'] for x in value['gpus']}==set(launch['gpu_uuids']) for value in guards)
assert all(set(p['pid'] for p in g['foreign'])<={int(x) for x in protected} for value in guards for g in value['gpus'])
assert result['allocated_cpus']==192 and result['allocated_gpus']==8 and result['requested_cpus']==32
implied_GPU_wall=result['gpu_hours']*3600/8;implied_CPU_wall=result['allocated_cpu_hours']*3600/192
assert 0<result['wall_seconds']<=419 and all(math.isfinite(x) for x in [implied_GPU_wall,implied_CPU_wall])
assert round(implied_GPU_wall,3)==round(implied_CPU_wall,3)==round(result['wall_seconds'],3)==round(result['end_epoch']-result['start_epoch'],3)

maxforeignmem=max(g['foreign_memory_fraction'] for v in guards for g in v['gpus']);maxforeignsm=max(g['foreign_sm_upper'] for v in guards for g in v['gpus'])
wholecardpeak=max(g['total_used_memory_bytes'] for v in guards for g in v['gpus'])
statistics=[sr['record']['raw_probability_observations'] for sr in s['ready']]
rawmax=max(x['max_abs'] for x in statistics);rawmean=sum(x['tree_weighted_abs_numerator'] for x in statistics)/sum(x['physical_effective_token_occurrences'] for x in statistics)
above=sum(x['above_trial_001_count'] for x in statistics)
assert rawmax==1.3392277956008911 and rawmean==0.007177164269124814 and above==84390
capture=ROOT/'work/capture_single_window3296_actual_metadata.py';capturestderr=ROOT/'work/tree-vllm-window-boundaries3295/ACTUAL3296_CAPTURE_STDERR.txt';assert capturestderr.read_bytes()==b''
receipt=dict(status='ACTUAL3296_FULL28_NATIVE_AND_74_BOUNDARY_MEASUREMENT_INDEPENDENTLY_ACCEPTED_NO_PROBABILITY_OR_TRAINING_ACCEPTANCE',reviewed_at_utc=datetime.now(timezone.utc).isoformat(),reviewer_task='/root/tree_window_final_review',job_id='3296',raw_file=dict(path=str(RAW),bytes=383665002,sha256=sha(RAW)),terminal_metadata=dict(path=str(snapshot),bytes=snapshot.stat().st_size,sha256=sha(snapshot)),bound_prior_dispatch_receipt_sha256=sha(REVIEW/'DISPATCH_RECEIPT.json'),bound_package_source_receipt_sha256=sha(ROOT/'work/tree-vllm-window-final-review/boundaries3295/PACKAGE_INDEPENDENT_RECEIPT.json'),bound_final_control_source_proof_sha256=sha(ROOT/'work/tree-vllm-window-final-review/boundaries3295/FINAL_CONTROL_SOURCE_PROOF.json'),actual_raw28_unique_record_bytes_and_hashes_verified=True,actual8_ready_records_equal_result_and_condensed_verified=True,scheduler_terminal='COMPLETED0:0',scheduler_runtime_seconds=419,actual_measurement_execution_verified=True,actual_generation_invocations=8,actual_complete_original_prompt_rows=2560,actual_emitted_tokens=808994,actual_actor_scoring_native_forwards=640,actual_historical_warmup_native_forwards=8,actual_actor_native_forwards_including_warmup=648,actual_optimizer_writes=0,actual_backward_calls=0,actual_scheduler_instantiated=False,formal_training_outer_updates_contributed=0,all640_original_native4key_object_input_RNG_forward_receipts_verified=True,all8_warmup_and_firstnew435_HF_nativecast_alias_metadata_verified=True,HF_serialized_to_unique435_to434_verified=True,allrank_live435to255_cast_prepost_buffers_verified=True,allmaster_gradients_modes_unchanged_metadata=True,all2560_generation_prompt_token_ids_and_all808994_actual_sampled_token_logprob_metadata_equal3295=True,allrank_raw_probability_statistics_and9_full_context_anchor_observations_equal3295=True,prior3295_fullraw_sha256='30deb240d598b4880732d8eae579bfdcc5a9e53fab4363c17cc2319fec244010',prior3295_fullraw_receipt_sha256='c8f18426ba67c4c1e8c898486a57207459b52c6fb33247aa3d955fb0dfb2e00a',same_causal_context_anchor_count=9,actual_boundary_vectors_per_backend=666,actual_boundary_pairs_global=666,actual_boundaries_per_anchor=74,boundary_actual_method_and_HFconfig_sourcepins_verified=True,allboundary_hooks_restored_and_no_native_result_replacement_metadata_verified=True,allrank_native_boundary_hookcounts_equal_original_execute_and_actor80=True,boundaries=sorted(boundary_proofs,key=lambda x:(x['rank'],x['response_offset'])),rank_proofs=rank_proofs,anchors=anchor_proofs,raw_max_abs=rawmax,raw_mean_abs=rawmean,above_trial_diagnostic001_count=above,global_p99=None,raw_statistics_scope='Native-source-computed unweighted raw emitted statistics; no PT scalar vector or quantile recomputation. Existing tree_weighted_* field labels are the raw all-ones diagnostic mapping, not Tree leaf/loss weights or production thresholds.',all82_guard_records_verified=True,maximum_foreign_memory_fraction=maxforeignmem,maximum_foreign_SM_upper=maxforeignsm,wholecard_5second_sampled_peak_bytes=wholecardpeak,wholecard_peak_is_training_capacity_proof=False,original8_background_PID_startticks_preserved=protected,cleanup_residual={},terminal_GPU_processes_only_original8=True,costs=dict(Slurm_allocated_GPU_hours=8*419/3600,Slurm_allocated_CPU_hours=192*419/3600,Slurm_requested_CPU_count=32,Slurm_allocated_CPU_count=192,supervisor_wall_seconds=result['wall_seconds'],supervisor_GPU_hours=result['gpu_hours'],supervisor_allocated_CPU_hours=result['allocated_cpu_hours'],separate_time_calls_GPU_implied_wall_delta_seconds=implied_GPU_wall-result['wall_seconds'],separate_time_calls_CPU_implied_wall_delta_seconds=implied_CPU_wall-result['wall_seconds'],cost_field_consistency_resolution_seconds=.001,accounting_resolution_is_not_probability_or_guard_tolerance=True,sampled_process_snapshot_count=len(process),sampled_process_PID_count=len(maxima),sampled_PID_lifetime_CPU_maxima_sum_seconds=sum(maxima.values()),actual_exited_or_unreadable_process_samples=exited_samples,missing_tick_entries_not_zero_imputed=True,complete_consumed_CPU_hours=None,consumed_CPU_scope='Incomplete per-process sampled maxima, separate from full allocated CPU hours; not complete exit accounting'),audit_reader_source=dict(path=str(capture),sha256=sha(capture),scope='Saved source128MiB/file512MiB-total, only28 owned JSON/log/out records and scheduler/intent metadata; gzip management capture, no PT/model/index/data assets; no runtime resource gate modified'),capture_stderr_empty_sha256=sha(capturestderr),checker_source_sha256=sha(Path(__file__)),checker_scope_assumption_errors=[dict(path=str(REVIEW/f'CHECKER_SCOPE_ERROR_{i}.json'),sha256=sha(REVIEW/f'CHECKER_SCOPE_ERROR_{i}.json'),scope='Reviewer-only local metadata checker correction; original sources/actual exit1 preserved; no runtime or acceptance fabricated') for i in [1,2,3,4]],source_records=[{k:x[k] for k in ['path','bytes','sha256']} for x in rows],actual_PT_tensor_file_identities_only_bound=True,reviewer_PT_or_model_or_data_asset_read=False,reviewer_remote_access=False,reviewer_GPU_execution=False,CPU_suites_repeated=False,production_source_modified=False,full_boundary_vectors_or_actor_raw_distributions_independently_recomputed=False,probability_accepted=False,production_thresholds_calibrated=False,training_accepted=False,GPU_capacity_proven=False,actual_loaded_binary_identity_proven=False,historical3268_mu_identity_proven=False,historical3295_extreme_reproduced_proven=False,unique_kernel_cause_proven=False,first_boundary_scope='Earliest differing captured embedding/norm output for9 fixed selected predictor contexts only; not a unique attention/norm/kernel or collapse cause',new_intent_spent_no_retry=True)
target=REVIEW/'MEASUREMENT_RECEIPT.json';assert not target.exists();target.write_text(json.dumps(receipt,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(receipt_sha256=sha(target),actual_records=len(rows),first_differences=[(x['anchor_id'],x['first_nonidentical_observed_boundary'],x['first_difference_max_abs']) for x in receipt['boundaries']],sampled_process_snapshots=len(process),sampled_process_PID_count=len(maxima),sampled_process_CPU_seconds=sum(maxima.values()),wholecard_peak=wholecardpeak,probability_accepted=False),sort_keys=True))
