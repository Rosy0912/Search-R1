"""Read-only, CPU, same-node causal-context evidence for Tree sampling calibration.

Raw sampler/FSDP pairs and final-leaf old-policy values are distinct evidence.
Exact matching uses every attention-valid prior token and its recorded position.
No averages choose among duplicate raw observations for a final supervised token.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import socket
import struct
import sys
import time

import torch

PREFIX_VERSION = b'TreeSamplingContext/v1\x00'
TASK_ROOT = Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
SOURCE_FILES = ('search_r1/baselines/sampling_consistency.py',
                'search_r1/baselines/policy_safety_runtime.py',
                'verl/workers/fsdp_workers.py', 'verl/trainer/ppo/ray_trainer.py',
                'verl/workers/rollout/vllm_rollout/vllm_rollout.py',
                'verl/trainer/config/ppo_tree_safety300.yaml')

def require(condition, message):
    if not condition:
        raise ValueError(message)

def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()

def load_verified(path, expected):
    """SHA and inode/size/time identity are checked on the same open descriptor."""
    path = Path(path)
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(8*1024**2), b''):
            digest.update(chunk)
        require(digest.hexdigest() == expected, 'Snapshot SHA changed: '+str(path))
        stream.seek(0)
        value = torch.load(stream, map_location='cpu', weights_only=False)
        after, current = os.fstat(stream.fileno()), path.stat()
        for key in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns'):
            require(getattr(before,key) == getattr(after,key) == getattr(current,key),
                    'Snapshot changed while loading: '+str(path))
    return value

def json_bound(path, expected):
    path = Path(path)
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        payload = stream.read()
        require(hashlib.sha256(payload).hexdigest() == expected, 'JSON SHA changed: '+str(path))
        after, current = os.fstat(stream.fileno()), path.stat()
        for key in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns'):
            require(getattr(before, key) == getattr(after, key) == getattr(current, key),
                    'JSON changed while loading: '+str(path))
    return json.loads(payload)

class Range:
    __slots__ = ('count','sampled_finite','actor_finite','paired_finite','sampled_min',
                 'sampled_max','actor_min','actor_max','pair_min','pair_max','first')
    def __init__(self, first):
        self.count = self.sampled_finite = self.actor_finite = self.paired_finite = 0
        self.sampled_min = self.actor_min = self.pair_min = math.inf
        self.sampled_max = self.actor_max = self.pair_max = -math.inf
        self.first = first
    def add(self, sampled, actor):
        self.count += 1
        if math.isfinite(sampled):
            self.sampled_finite += 1
            self.sampled_min, self.sampled_max = min(self.sampled_min,sampled), max(self.sampled_max,sampled)
        if math.isfinite(actor):
            self.actor_finite += 1
            self.actor_min, self.actor_max = min(self.actor_min,actor), max(self.actor_max,actor)
        if math.isfinite(sampled) and math.isfinite(actor) and math.isfinite(actor-sampled):
            self.paired_finite += 1
            delta = actor-sampled
            self.pair_min, self.pair_max = min(self.pair_min,delta), max(self.pair_max,delta)

class RawIndex:
    def __init__(self):
        self.values = {}
        self.rows = self.valid_tokens = self.paired_finite = self.invalid_pairs = 0
        self.signed_sum = self.abs_sum = self.max_abs = 0.
        self.delta_min, self.delta_max = math.inf, -math.inf
        self.zero_based_position_rows = 0
        self.prompt_attention_anomaly_rows = 0
    def add_snapshot(self, tensors, label):
        layout = normalize(tensors, raw=True)
        sampled = layout['sampled_log_probs'].tolist()
        actor = layout['old_log_probs'].tolist()
        for row in range(layout['rows']):
            self.rows += 1
            valid_positions = layout['position_ids'][row][layout['attention_mask'][row].bool()].tolist()
            self.zero_based_position_rows += int(valid_positions == list(range(len(valid_positions))))
            prompt_attention = layout['attention_mask'][row,:layout['response_start']].tolist()
            self.prompt_attention_anomaly_rows += int(prompt_attention != sorted(prompt_attention))
            for key, offset, prefix_length, token, position in causal_keys(layout, row, supervised=False):
                sample_value, actor_value = float(sampled[row][offset]), float(actor[row][offset])
                if key not in self.values:
                    self.values[key] = Range(dict(path=label,row=row,response_offset=offset,
                                                  prefix_length=prefix_length,token_id=token,position_id=position))
                self.values[key].add(sample_value,actor_value)
                self.valid_tokens += 1
                delta = actor_value-sample_value
                if math.isfinite(sample_value) and math.isfinite(actor_value) and math.isfinite(delta):
                    self.paired_finite += 1
                    self.signed_sum += delta; self.abs_sum += abs(delta)
                    self.max_abs = max(self.max_abs,abs(delta))
                    self.delta_min, self.delta_max = min(self.delta_min,delta), max(self.delta_max,delta)
                else:
                    self.invalid_pairs += 1
    def report(self):
        duplicates = [value for value in self.values.values() if value.count > 1]
        return dict(raw_rows=self.rows,raw_attention_valid_response_token_occurrences=self.valid_tokens,
                    unique_exact_context_token_keys=len(self.values),duplicate_raw_keys=len(duplicates),
                    duplicate_raw_occurrences=sum(value.count-1 for value in duplicates),
                    raw_keys_with_distinct_sampled_logprob_range=sum(value.sampled_finite > 1 and
                        value.sampled_min != value.sampled_max for value in self.values.values()),
                    raw_rows_recorded_positions_equal_zero_based_valid_sequence=self.zero_based_position_rows,
                    raw_rows_recorded_positions_differ_from_zero_based_valid_sequence=self.rows-self.zero_based_position_rows,
                    raw_prompt_rows_with_internal_or_right_attention_padding=self.prompt_attention_anomaly_rows,
                    raw_same_turn_actor_minus_sampled=dict(finite_paired_occurrences=self.paired_finite,
                        invalid_paired_occurrences=self.invalid_pairs,
                        signed_min=None if not self.paired_finite else self.delta_min,
                        signed_max=None if not self.paired_finite else self.delta_max,
                        max_abs=None if not self.paired_finite else self.max_abs,
                        occurrence_mean_signed=finite_div(self.signed_sum,self.paired_finite),
                        occurrence_mean_abs=finite_div(self.abs_sum,self.paired_finite)),
                    count_scope='Physical raw generation rows/tokens include repeated calls and GPU-divisibility duplicates; not unique trajectories or tool calls')

def finite_div(numerator, denominator):
    return numerator/denominator if denominator and math.isfinite(numerator) and math.isfinite(denominator) else None

def finite_pair(lo, hi):
    return [lo, hi] if math.isfinite(lo) and math.isfinite(hi) else None

def normalize(tensors, raw=False):
    required = {'input_ids','position_ids','attention_mask','responses','old_log_probs'}
    required |= {'sampled_log_probs'} if raw else {'loss_mask','policy_weights'}
    require(required.issubset(tensors), 'Missing retained context tensors: '+str(required-set(tensors)))
    for key in required:
        require(isinstance(tensors[key],torch.Tensor) and tensors[key].device.type == 'cpu',
                'CPU tensor required: '+key)
    ids, positions, attention = (tensors[key] for key in ('input_ids','position_ids','attention_mask'))
    require(ids.ndim == 2 and positions.shape == attention.shape == ids.shape,
            'This Qwen evidence requires explicit 2D position IDs and identical full-context shapes')
    require(not ids.is_floating_point() and not positions.is_floating_point(), 'Token/position IDs must be integral')
    require(bool(((attention == 0)|(attention == 1)).all()), 'Nonbinary attention evidence')
    responses = tensors['responses']; rows, length = ids.shape
    require(responses.ndim == 2 and responses.shape[0] == rows, 'Response rows changed')
    width = responses.shape[1]; start = length-width
    require(width > 0 and start >= 0 and torch.equal(ids[:,start:],responses),
            'Responses must correspond exactly to the retained input suffix')
    result = dict(tensors,rows=rows,width=width,response_start=start)
    for key in ('old_log_probs','sampled_log_probs') if raw else ('old_log_probs','loss_mask','policy_weights'):
        tensor = tensors[key]
        if not raw and key in ('loss_mask','policy_weights') and tensor.shape == ids.shape:
            tensor = tensor[:,start:]
        require(tensor.shape == responses.shape, 'Response-aligned shape changed: '+key)
        result[key] = tensor
    if not raw:
        require(bool(((result['loss_mask'] == 0)|(result['loss_mask'] == 1)).all()), 'Nonbinary supervision mask')
    return result

def causal_keys(layout, row, supervised):
    """Prefix excludes current token; key also carries current token and position.

    Attention-valid observations enter the prefix even when loss_mask is zero.
    Padding never enters the prefix. Current response supervision is masked only
    for lookup, not for building subsequent tokens' causal contexts.
    """
    ids=layout['input_ids'][row].tolist(); positions=layout['position_ids'][row].tolist()
    attention=layout['attention_mask'][row].tolist()
    loss=layout['loss_mask'][row].tolist() if supervised else None
    prefix=hashlib.sha256(PREFIX_VERSION); prefix_length=0; start=layout['response_start']
    for offset,(token,position,valid) in enumerate(zip(ids,positions,attention)):
        if not valid:
            continue
        response_offset=offset-start
        if response_offset >= 0 and (not supervised or loss[response_offset]):
            pair=struct.pack('<qq',int(token),int(position))
            yield prefix.digest()+pair,response_offset,prefix_length,int(token),int(position)
        prefix.update(struct.pack('<qq',int(token),int(position)))
        prefix_length += 1

def analyze_targets(index, snapshots, max_examples=8):
    rows=nonempty_rows=supervised=matched=duplicate_matches=invalid_old=invalid_weights=0
    sampled_numeric=actor_numeric=invalid_sampled_delta=invalid_actor_delta=0
    total_weight=matched_weight=sampled_weight=actor_weight=0.
    lower_sampled=upper_sampled=lower_actor=upper_actor=abs_lower=abs_upper=0.
    global_sampled_min,global_sampled_max=math.inf,-math.inf
    uniform_error=0.; unmatched=[]; matched_examples=[]; target_key_counts={}
    for label,tensors in snapshots:
        layout=normalize(tensors,raw=False); old=layout['old_log_probs'].tolist(); weights=layout['policy_weights'].tolist()
        effective=layout['attention_mask'][:,-layout['width']:].bool() & layout['loss_mask'].bool()
        for row in range(layout['rows']):
            rows += 1; count=int(effective[row].sum()); nonempty_rows += int(count > 0)
            for key,offset,prefix_length,token,position in causal_keys(layout,row,supervised=True):
                supervised += 1; target_key_counts[key]=target_key_counts.get(key,0)+1
                weight=float(weights[row][offset]); old_value=float(old[row][offset])
                good_weight=math.isfinite(weight) and weight >= 0
                if good_weight:
                    total_weight += weight
                    uniform_error=max(uniform_error,abs(weight-1/count))
                else:
                    invalid_weights += 1
                if not math.isfinite(old_value):invalid_old += 1
                raw=index.values.get(key)
                if raw is None:
                    if len(unmatched) < max_examples:
                        unmatched.append(dict(path=label,row=row,response_offset=offset,
                            prefix_sha256=key[:32].hex(),prefix_length=prefix_length,
                            token_id=token,position_id=position,
                            reason='No identical complete attention-valid token+position prefix/current-token key in retained raw generation evidence'))
                    continue
                matched += 1; duplicate_matches += int(raw.count > 1)
                if good_weight:matched_weight += weight
                if not math.isfinite(old_value):continue
                if raw.sampled_finite:
                    lo,hi=old_value-raw.sampled_max,old_value-raw.sampled_min
                    good_delta = math.isfinite(lo) and math.isfinite(hi)
                    if good_delta:
                        sampled_numeric += 1
                        global_sampled_min,global_sampled_max=min(global_sampled_min,lo),max(global_sampled_max,hi)
                    else:
                        invalid_sampled_delta += 1
                    if good_weight and good_delta:
                        sampled_weight += weight;lower_sampled += weight*lo;upper_sampled += weight*hi
                        abs_lower += weight*(0. if lo <= 0 <= hi else min(abs(lo),abs(hi)))
                        abs_upper += weight*max(abs(lo),abs(hi))
                    if len(matched_examples) < max_examples:
                        matched_examples.append(dict(path=label,row=row,response_offset=offset,
                            prefix_sha256=key[:32].hex(),prefix_length=prefix_length,token_id=token,position_id=position,
                            final_leaf_recomputed_old_logprob=old_value,raw_occurrence_count=raw.count,
                            raw_sampled_finite_count=raw.sampled_finite,
                            raw_sampled_logprob_range=[raw.sampled_min,raw.sampled_max],
                            final_old_minus_raw_sampled_range=finite_pair(lo,hi),
                            difference_finite=good_delta,first_raw_observation=raw.first))
                if raw.actor_finite:
                    alo,ahi=old_value-raw.actor_max,old_value-raw.actor_min
                    good_delta = math.isfinite(alo) and math.isfinite(ahi)
                    actor_numeric += int(good_delta)
                    invalid_actor_delta += int(not good_delta)
                    if good_weight and good_delta:
                        actor_weight += weight
                        lower_actor += weight*alo
                        upper_actor += weight*ahi
    weight_ok=invalid_weights == 0
    aggregates=dict(total_weight=total_weight,matched_weight=matched_weight,
        sampled_weight=sampled_weight,actor_weight=actor_weight,lower_sampled=lower_sampled,
        upper_sampled=upper_sampled,lower_actor=lower_actor,upper_actor=upper_actor,
        abs_lower=abs_lower,abs_upper=abs_upper)
    overflow_fields=[name for name,value in aggregates.items() if not math.isfinite(value)]
    def number(value):
        return value if math.isfinite(value) else None
    def bounds(lo,hi,denominator):
        return [finite_div(lo,denominator),finite_div(hi,denominator)]
    return dict(leaf_rows_denominator=rows,nonempty_supervised_leaf_rows=nonempty_rows,
        effective_supervised_token_occurrences=supervised,unique_final_context_token_keys=len(target_key_counts),
        duplicated_final_context_token_occurrences=sum(count-1 for count in target_key_counts.values()),
        exact_context_matched_token_occurrences=matched,exact_context_unmatched_token_occurrences=supervised-matched,
        token_exact_context_coverage=finite_div(matched,supervised),matched_tokens_with_multiple_raw_occurrences=duplicate_matches,
        tree_weights=dict(all_effective_weight_mass=number(total_weight) if weight_ok else None,
            matched_effective_weight_mass=number(matched_weight) if weight_ok else None,
            weighted_exact_context_coverage=finite_div(matched_weight,total_weight) if weight_ok else None,
            invalid_effective_weights=invalid_weights,max_abs_difference_from_original_uniform_leaf_weights=uniform_error),
        finite_numeric_evidence=dict(invalid_final_leaf_old_logprob_occurrences=invalid_old,
            matched_tokens_with_finite_raw_sampled_and_final_old=sampled_numeric,
            matched_tokens_with_finite_raw_actor_and_final_old=actor_numeric,
            invalid_sampled_difference_occurrences=invalid_sampled_delta,
            invalid_actor_difference_occurrences=invalid_actor_delta,
            nonfinite_aggregate_fields=overflow_fields),
        final_leaf_old_minus_actual_raw_sampled=dict(
            signed_range_over_all_exact_matches=None if not sampled_numeric else [global_sampled_min,global_sampled_max],
            weighted_signed_numerator_bounds=finite_pair(lower_sampled,upper_sampled) if weight_ok else None,
            weighted_absolute_numerator_envelope=finite_pair(abs_lower,abs_upper) if weight_ok else None,
            finite_matching_weight_denominator=number(sampled_weight) if weight_ok else None,
            matched_weight_mean_signed_bounds=bounds(lower_sampled,upper_sampled,sampled_weight) if weight_ok else None,
            original_full_leaf_row_mean_signed_bounds=bounds(lower_sampled,upper_sampled,rows) if weight_ok else None),
        final_leaf_old_minus_same_turn_raw_actor=dict(
            weighted_signed_numerator_bounds=finite_pair(lower_actor,upper_actor) if weight_ok else None,
            finite_matching_weight_denominator=number(actor_weight) if weight_ok else None,
            matched_weight_mean_signed_bounds=bounds(lower_actor,upper_actor,actor_weight) if weight_ok else None),
        unmatched_examples=unmatched,matched_examples=matched_examples,
        duplicate_scope='Every final supervised leaf token retains its actual Tree weight. Shared prefixes repeated among leaves count repeatedly in this training view. Multiple raw samples are represented by exact min/max/count, never averaged into a presumed sampled old policy.',
        limits='Unmatched contexts cannot support a sampled-policy comparison. Range envelopes do not select which duplicate raw occurrence generated a final token; partial exact-context coverage cannot establish all PPO old logprobs equal the sampling policy.')

def verify_bundle(run, package_dir, expected_package_sha, expected_ready_sha, require_node=True):
    run,package_dir=Path(run).resolve(),Path(package_dir).resolve()
    if require_node:
        require(socket.gethostname().split('.')[0] == 'h100-9', 'Analysis must run on h100-9 with all assets local')
        require(run.is_relative_to(TASK_ROOT/'runs') and package_dir.is_relative_to(TASK_ROOT), 'Assets outside this same-node task')
    package=json_bound(package_dir/'manifest.json',expected_package_sha)
    require(package['stage']=='calibration110' and package['probability_calibration']['environment']=={'TREE_PPO_CALIBRATION_ONLY':'1'},
            'Only dedicated zero-write calibration evidence is supported')
    source=[]
    for name in SOURCE_FILES:
        relative='source/repo/'+name;expected=package['files'][relative]
        require(file_sha(package_dir/relative)==expected, 'Frozen diagnostic source changed: '+name)
        source.append(dict(path=str(package_dir/relative),sha256=expected))
    ready_path=run/'output/probability-calibration/ready.json'
    ready=json_bound(ready_path,expected_ready_sha)
    require(ready['status']=='CALIBRATION_OBSERVED_ZERO_WRITES' and ready['parent_completed_outer']==110 and
            ready['observed_outer']==111 and ready['new_completed_outer']==0 and ready['optimizer_committed']==0 and
            ready['scheduler_advanced'] is False, 'Not a complete zero-write observation')
    identity=dict(job_id=run.name.rsplit('-',1)[1],run_id=run.name,code_commit=package['code_commit'],
                  config_sha256=ready['config_sha256'],data_identity=package['common_data_identity'],
                  model_identity=package['model_identity']['sha256'])
    require(all(ready.get(key)==value for key,value in identity.items() if key!='run_id'), 'Ready source/config/assets binding changed')
    require(file_sha(run/'resolved.yaml')==identity['config_sha256'] and
            file_sha(run/'output/entry-audit/entry-config.yaml')==identity['config_sha256'], 'Actual config identity changed')
    parent=package['parent_manifest']
    require(parent['identity']['completed_step']==110 and file_sha(parent['path'])==parent['sha256'], 'Healthy110 parent manifest changed')
    restored_path=run/'output/recovery-loaded.json'
    restored_sha=file_sha(restored_path);restored=json_bound(restored_path,restored_sha)
    require(restored['provenance']['parent_identity']==parent['identity'] and
            restored['provenance']['parent_cursor']==57600 and restored['provenance']['exact_executor_resume'] is False,
            'Actual restoration parent/cursor binding changed')
    require(len(ready['workers'])==8, 'Missing eight actual worker observations')
    source.append(dict(path=str(restored_path),sha256=restored_sha,kind='recovery_loaded_metadata'))
    return run,package,ready,identity,source

def analyze_bundle(run, package, ready, identity, source, expected_package_sha, expected_ready_sha, max_examples):
    index=RawIndex(); evidence=[]; monitor=[]; common_calls=None;raw_json_paths=set();call_modes={}
    def inside(path,directory):
        path=Path(path).resolve();require(path.is_relative_to(directory.resolve()),'Evidence escaped current same-node run');return path
    for rank,worker in enumerate(sorted(ready['workers'],key=lambda row:row['rank'])):
        require(worker['rank']==rank and worker['world_size']==8 and worker['optimizer_committed']==0 and
                worker['scheduler_advanced'] is False and worker['scheduler_step']==110 and
                worker['restored_optimizer_steps']==[2500] and
                all(worker.get(key)==value for key,value in identity.items()), 'Worker source/parent identity changed')
        entries=worker['diagnostic_entries'];require([entry['path'] for entry in entries]==worker['diagnostic_records'], 'Diagnostic entry paths changed')
        calls=[]
        for entry in entries:
            path=inside(entry['path'],run/'output/sampling-consistency');raw=json_bound(path,entry['sha256'])
            require(path.name==f'rank-{rank:02d}-call-{raw["generation_call"]:05d}.json', 'Raw file/call/rank path binding changed')
            require(path not in raw_json_paths, 'Duplicate raw evidence path');raw_json_paths.add(path)
            require(raw['rank']==rank and raw['world_size']==8 and raw['scheduler_step']==110 and
                    raw['optimizer_committed']==0 and all(raw.get(key)==value for key,value in identity.items()), 'Raw diagnostic identity changed')
            require(raw['saved_rows']==raw['local_rows'] and raw['saved_rows'] > 0 and raw['enforced'] is False,
                    'Calibration raw snapshot is incomplete or from another stage')
            call=raw['generation_call']; mode=call_modes.setdefault(call,dict(generation_call=call,ranks=[],
                captured_distributions=[],global_diagnostic_decision=raw['decision']))
            require(raw['decision']==mode['global_diagnostic_decision'], 'Raw global diagnostic decisions differ across workers')
            mode['ranks'].append(rank)
            if raw['distribution'] not in mode['captured_distributions']:
                mode['captured_distributions'].append(raw['distribution'])
            snapshot=inside(entry['snapshot_path'],run/'output/sampling-consistency')
            require(snapshot.name==path.with_suffix('.pt').name, 'Raw snapshot basename differs from its diagnostic')
            require(raw['snapshot_path']==str(snapshot) and raw['snapshot_sha256']==entry['snapshot_sha256'], 'Raw snapshot binding changed')
            value=load_verified(snapshot,entry['snapshot_sha256'])
            require(value['input_ids'].shape[0]==raw['saved_rows'], 'Raw snapshot sample count differs')
            require(value['distribution']==raw['distribution'], 'Raw snapshot sampling distribution differs from bound metadata')
            index.add_snapshot(value,str(snapshot));calls.append(raw['generation_call'])
            evidence.append(dict(kind='raw_same_turn',path=str(path),sha256=entry['sha256'],
                                 snapshot_path=str(snapshot),snapshot_sha256=entry['snapshot_sha256'],rows=raw['saved_rows']))
            del value
        if common_calls is None:common_calls=calls
        require(calls==common_calls==list(range(1,len(calls)+1)), 'Raw call coverage differs across workers')
        path=inside(worker['monitor_path'],run/'output/probability-calibration')
        require(path.name==f'rank-{rank:02d}-monitor.pt', 'Monitor path/rank binding changed')
        value=load_verified(path,worker['monitor_sha256'])
        require(value['input_ids'].shape[0]==32, 'Actual monitor row count differs from the frozen32/rank protocol')
        monitor.append((str(path),value))
        evidence.append(dict(kind='final_leaf_monitor_subset',path=str(path),sha256=worker['monitor_sha256']))
    require(raw_json_paths=={path.resolve() for path in (run/'output/sampling-consistency').glob('rank-*-call-*.json')},
            'Raw diagnostic file set differs from the complete pinned worker entries')
    monitor_report=analyze_targets(index,monitor,max_examples);del monitor
    actor_evidence=[]
    def actor_inputs():
        for path in sorted((run/'output/trajectories').glob('outer-*/actor-inputs.pt')):
            manifest_path=path.parent/'manifest.json';manifest_sha=file_sha(manifest_path)
            manifest=json_bound(manifest_path,manifest_sha)
            require(manifest['status']=='ALL_ROLLOUTS_AND_ACTOR_INPUTS_PUBLISHED', 'Incomplete actor-input receipt')
            expected=dict(outer_step=111,job_id=identity['job_id'],source=identity['code_commit'],
                          config=identity['config_sha256'],data=identity['data_identity'])
            require(manifest['identity']==expected, 'Actor-input receipt identity changed')
            value=load_verified(path,manifest['payloads'][path.name])
            require(value['identity']==expected and value['rows']==manifest['rows']==value['tensors']['input_ids'].shape[0],
                    'Actor-input tensor identity/rows changed')
            actor_evidence.append(dict(kind='complete_final_actor_inputs',path=str(path),sha256=manifest['payloads'][path.name],
                                       manifest_path=str(manifest_path),manifest_sha256=manifest_sha,rows=value['rows']))
            yield str(path),value['tensors']
    actor_report=analyze_targets(index,actor_inputs(),max_examples)
    require(actor_evidence and actor_report['leaf_rows_denominator']==640*6, 'Complete Tree640x6 actor inputs missing')
    audit=run/'output/tree-rank-audit'
    require(not list(audit.glob('step-*.json')) and not list((audit/'ratio-tail').glob('*')),
            'Zero-write evidence contains actual actor optimizer audit files')
    require(not (run/'output/checkpoints').exists() or not any((run/'output/checkpoints').rglob('*')),
            'Zero-write evidence unexpectedly saved a new training checkpoint')
    return dict(status='EXACT_CONTEXT_SAMPLING_EVIDENCE_ANALYZED_NO_TRAINING_ACCEPTANCE',identity=identity,
        parent_manifest_sha256=package['parent_manifest']['sha256'],package_manifest_sha256=expected_package_sha,
        ready_sha256=expected_ready_sha,helper_source_sha256=file_sha(__file__),diagnostic_source_files=source,
        prefix_hash_definition='SHA256 seeded with TreeSamplingContext/v1\\0; append each prior attention-valid (int64 token,int64 recorded position) in order; lookup key also includes current token and position. Left/right padding excluded; loss-masked observations remain in the causal prefix.',
        raw_sampling_evidence=index.report(),monitor_subset=monitor_report,complete_actor_inputs=actor_report,
        captured_raw_generation_call_modes=[call_modes[call] for call in sorted(call_modes)],
        evidence=evidence+actor_evidence,actual_optimizer_writes=0,actual_forward_passes=0,
        interpretation_limits=[
            'Raw same-turn sampler/FSDP numerical differences and final postprocessed Tree leaf contexts are reported separately.',
            'Monitor rows are a subset of complete actor inputs; their counts/weights must not be added to the complete-input totals.',
            'Exact-context labels refer to recorded attention-valid tokens and positions. vLLM API receives stripped prompt_token_ids, without explicit attention_mask or position_ids; the actual stripped list and internal engine KV/position derivation are not directly exported by these snapshots. Internal/right prompt padding or nonzero position offsets require separate examination.',
            'Context nonmatches may reflect postprocessing, retokenization, rolling truncation or evidence coverage; the unmatched count alone cannot assign a cause.',
            'Min/max/count retain duplicate raw observations; no average is substituted for the particular sampled policy of a final token.',
            'This CPU analysis is evidence, not calibrated thresholds, optimizer acceptance, five-outer acceptance or final300/51713 completion.'])

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--package',type=Path,required=True)
    parser.add_argument('--expected-package-manifest-sha256',required=True)
    parser.add_argument('--expected-ready-sha256',required=True)
    parser.add_argument('--max-examples',type=int,default=8)
    args=parser.parse_args();require(0 <= args.max_examples <= 32,'Example count out of range')
    require(not os.environ.get('CUDA_VISIBLE_DEVICES',''), 'Hide GPU devices for this independent0GPU CPU analysis')
    started=time.perf_counter();cpu=time.process_time();torch.set_num_threads(4)
    run,package,ready,identity,source=verify_bundle(args.run,args.package,args.expected_package_manifest_sha256,args.expected_ready_sha256)
    result=analyze_bundle(run,package,ready,identity,source,args.expected_package_manifest_sha256,args.expected_ready_sha256,args.max_examples)
    require(not torch.cuda.is_initialized(),'CPU context analysis unexpectedly initialized CUDA')
    result['analysis_cost']=dict(wall_seconds=time.perf_counter()-started,process_cpu_seconds=time.process_time()-cpu,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
        threads=4,cuda_initialized=False,analysis_slurm_job_id=os.environ.get('SLURM_JOB_ID'))
    print(json.dumps(result,sort_keys=True,allow_nan=False))

if __name__=='__main__':
    main()
