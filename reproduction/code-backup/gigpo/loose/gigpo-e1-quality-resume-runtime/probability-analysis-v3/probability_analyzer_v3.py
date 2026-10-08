"""Read-only probability evidence analysis for an explicitly bound actual calibration job; never approval.

JSON mode checks original bytes/identity/structural coverage. Optional cpu mode
reads bound PT originals and recomputes probability errors, with CUDA disabled.
No tolerance is selected and no GPU/model inference occurs. Optional output is
one new bounded aggregate JSON in own-job target logs, never copied raw tensors.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import select
import socket
import stat
import platform
from pathlib import Path, PurePosixPath
import re
import sys
from failure_contract_v2 import validate_failure_context

SOURCE = '52befe968c856162d1bc79cd0171bb38309f4f9e'
ROOT = Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004')
MAX_EXAMPLES = 20
SCOPE = 'GIGPO_POLICY_SAFETY_HEALTHY25_CALIBRATION'
RESTORE = '3a073aba77147db7d581cd514ee5c4ae34965e5ff9c163716bd25b47aeb478e4'
DATA = '0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
STATE_MODULE_SHA = 'b75c96537b006a70d101a11c5f2210634f55dd7796db1b06264425eb2f738015'
REQUIRED = ('actual-rollout-2560.pt', 'training-prefix-turns.json',
            'actual-probability-alignment.json', 'full-old-versus-sampled-prefix.json',
            'actual-vllm-backend-identity.json')
SHA_PATTERN = re.compile(r'[0-9a-f]{64}')
WORKER_FIELDS = ('rank', 'pid', 'start_ticks', 'scheduler_step', 'learning_rates',
                 'model_shards_sha256', 'optimizer_tensors_sha256', 'rng_tensors_sha256',
                 'local_shards_sha256', 'buffers_sha256', 'complete_state_sha256')


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def validate_backend_binding(binding):
    require(binding['helpers'] == str(ROOT / 'tools/policy-safety-calibration-52befe9-v9')
            and binding['backend_change_scope'] ==
                'CALIBRATION_FLASH_ATTN_ENV_PROPAGATION_AND_READONLY_ACTUAL_BACKEND_GATE_ONLY',
            'Analyzer v3 requires the exact reviewed external v9 backend scope')
    expected = dict(calibration_driver_v1_py='31f191711844c0c875149502e9c2ddfc9569f40ef88163baf860aa46dbd762b9',
        calibration_worker_v1_py='f2867517e79ecd87293c8ccf56c5600d61e597192fd49da233818d882f81df4d',
        calibration_run_v1_py='b0b344d4f8cb13d215538110570ea4ae124ca3ebea88de6e86f59477adc0e7c5')
    records = {row['path']: row for row in binding['files']}
    for key, digest in expected.items():
        name = key.removesuffix('_py') + '.py'
        path = str(Path(binding['helpers']) / name)
        require(path in records and records[path]['sha256'] == digest,
                'Actual backend wrapper/driver/worker bytes outside the reviewed v9 implementation')


def validate_backend_original(value, binding_sha):
    require(value['status'] == 'PASS_ACTUAL_METADATA_ONLY_NOT_PROBABILITY_PARITY'
            and value['expected_backend'] == 'FLASH_ATTN' and value['rank_count'] == 8
            and value['layers_per_rank'] == 36 and value['probability_accepted'] is False,
            'Backend original is missing, refused, or overstates probability acceptance')
    rows = value['actual_observations']
    require(len(rows) == 8 and all(type(row['rank']) is int for row in rows)
            and {row['rank'] for row in rows} == set(range(8))
            and all(type(row['pid']) is int and row['pid'] > 0 for row in rows)
            and len({row['pid'] for row in rows}) == 8, 'Backend original eight-worker identity differs')
    for row in rows:
        require(row['scope'] == 'ACTUAL_RUNNER_AND_ALL_LAYER_BACKEND_METADATA_ONLY'
                and row['source_commit'] == SOURCE and row['binding_sha256'] == binding_sha,
                'Backend original source/binding identity differs')
        require(row['requested_backend'] == 'FLASH_ATTN'
                and row['runner_backend_class'] == 'vllm.attention.backends.flash_attn.FlashAttentionBackend'
                and row['runner_backend_name'] == 'flash-attn', 'Actual runner backend fallback or unknown')
        require(row['model_dtype'] == 'torch.bfloat16' and row['cache_dtype'] == 'auto'
                and row['block_size'] == 16 and row['model_sliding_window'] is None,
                'Backend original actual dtype/cache/block/sliding configuration differs')
        layers = row['attention_layers']
        require(len(layers) == 36 and len({layer['name'] for layer in layers}) == 36,
                'Backend original missing/duplicate model layers')
        require(all(layer['implementation_class'] == 'vllm.attention.backends.flash_attn.FlashAttentionImpl'
                    and layer['head_size'] == 128 and layer['num_heads'] == 16
                    and layer['num_kv_heads'] == 2 and layer['sliding_window'] == [-1, -1]
                    for layer in layers), 'Actual layer backend fallback or configuration mismatch')
        require(row['observation_forward_calls'] == row['observation_tensor_value_reads'] == 0
                and row['actual_vllm_runner_positions_observed'] is False
                and row['probability_accepted'] is False, 'Backend observation scope overstated')
    return dict(scope='ORIGINAL_BACKEND_METADATA_IDENTITY_ONLY_NOT_NUMERICAL_PARITY',
                actual_observations=rows, probability_accepted=False)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path, expected):
    require(Path(path).is_absolute() and Path(path).is_file() and not Path(path).is_symlink()
            and Path(path).resolve() == Path(path) and SHA_PATTERN.fullmatch(expected) and sha(path) == expected,
            'Original JSON SHA mismatch: ' + str(path))
    return json.loads(Path(path).read_text())


class OriginalStore:
    def __init__(self, run):
        self.run = PurePosixPath(run)
        require(self.run.is_absolute() and '..' not in self.run.parts,
                'Actual run must be absolute without traversal')

    def path(self, original):
        path = PurePosixPath(original)
        require(path.is_absolute() and '..' not in path.parts
                and path.is_relative_to(self.run), 'Foreign original path: ' + original)
        local = Path(original)
        require(not local.is_symlink() and local.is_file() and local.resolve()==local,
                'Missing/symlink original or parent: ' + str(local))
        return local

    def verify(self, record):
        require(set(record) >= {'path', 'bytes', 'sha256'}, 'Incomplete original record')
        local = self.path(record['path'])
        require(type(record['bytes']) is int and record['bytes'] >= 0
                and SHA_PATTERN.fullmatch(record['sha256']), 'Invalid original identity')
        require(local.stat().st_size == record['bytes'] and sha(local) == record['sha256'],
                'Original bytes/SHA mismatch: ' + record['path'])
        return local


def structural_analysis(alignment, turns, cross):
    actions = alignment['actions']
    require(alignment['scope'] == cross['scope'] == SCOPE, 'Wrong probability scope')
    require(len(actions) == 2560 and {row['source'] for row in actions} == set(range(2560)),
            'Actual2560 row coverage missing or duplicate')
    mapping = {}
    for row in actions:
        require(type(row['source']) is int, 'Trajectory row ID is not an integer')
        require(row['actions'], 'Trajectory without actual action records')
        steps = [a['step_id'] for a in row['actions']]
        require(all(type(step) is int for step in steps),'Action step ID is not an integer')
        require(steps == list(range(len(steps))), 'Missing zero-token action timeline or duplicate step')
        for action in row['actions']:
            key = (row['source'], action['step_id'])
            mapping[key] = action
    observed = []
    for serial, active in enumerate(turns):
        require(active and len({row['source'] for row in active}) == len(active), 'Duplicate/empty active turn')
        for row in active:
            key = (row['source'], row['step_id'])
            require(key in mapping, 'Turn not present in2560 action rows')
            for field in ('generation_prefix', 'scoring_prefix', 'retained_action_tokens'):
                require(row[field] == mapping[key][field], 'Original turn/action record differs: ' + field)
            observed.append(key)
    require(len(observed) == len(set(observed)) and set(observed) == set(mapping),
            'Turn inventory does not cover each original action exactly once')
    checks = {(row['source'], row['step_id']): row for row in cross['checks']}
    require(len(checks) == len(cross['checks']) and set(checks) == set(mapping),
            'Missing/duplicate action probability checks')
    mismatches = {name: [] for name in ('prefix', 'tokens', 'position', 'mask', 'engine_prefix')}
    for key, action in mapping.items():
        count = len(action['retained_action_tokens'])
        check = checks[key]
        require(check['effective_tokens'] == count, 'Cross-check action length differs')
        require(len(action['sampled_tokens']) >= count, 'Retokenized action longer than sampled vector')
        if count and action['generation_prefix'] != action['scoring_prefix']:
            mismatches['prefix'].append(key)
        if count and action['sampled_tokens'][:count] != action['retained_action_tokens']:
            mismatches['tokens'].append(key)
        natural = list(range(len(action['generation_prefix']), len(action['generation_prefix']) + count))
        same_position = (action['sampling_action_position_ids'] == action['scoring_action_position_ids'] == natural)
        same_mask = action['sampling_action_mask'] == action['scoring_action_mask'] == [1] * count
        same_engine = action['engine_prompt_token_ids'] == action['generation_prefix']
        # check.same_position also asserts actual prompt positions; raw cpu mode
        # independently verifies that prefix component rather than guessing it.
        if not same_position or not check['same_position_ids']:
            mismatches['position'].append(key)
        if not same_mask:
            mismatches['mask'].append(key)
        if not same_engine:
            mismatches['engine_prefix'].append(key)
        require(check['same_effective_mask'] == same_mask
                and check['engine_prefix_matches_generation'] == same_engine,
                'Derived structural flags contradict original action fields')
        if not same_position:
            require(check['same_position_ids'] is False, 'Position mismatch hidden by summary')
        for field in ('full_trajectory_vs_sample_prefix_old_max_abs', 'full_vs_prefix_old_joint_sum_delta',
                      'full_vs_vllm_joint_sum_delta', 'prefix_old_vs_vllm_joint_sum_delta'):
            require(type(check[field]) in (int, float) and math.isfinite(check[field]),
                    'Nonfinite recorded probability error')
    recorded = alignment['alignment']
    require(recorded['trajectory_rows'] == 2560 and recorded['recorded_actions'] == len(mapping),
            'Alignment coverage summary differs')
    for name, field in (('prefix', 'prefix_mismatch_count'), ('tokens', 'action_token_mismatch_count'),
                        ('position', 'position_ids_mismatch_count'), ('mask', 'effective_mask_mismatch_count'),
                        ('engine_prefix', 'engine_prefix_mismatch_count')):
        require(recorded[field] == len(mismatches[name]), 'Recomputed structural count differs: ' + field)
    return mapping, checks, dict(trajectory_rows=2560, recorded_actions=len(mapping),
        effective_tokens=sum(len(action['retained_action_tokens']) for action in mapping.values()),
        mismatch_counts={k: len(v) for k, v in mismatches.items()},
        mismatch_examples={k: v[:MAX_EXAMPLES] for k, v in mismatches.items()})


def validate_ready_json(ready, binding, binding_sha, manifest_sha, manifest, job, run):
    require(ready['scope'] == SCOPE and ready['status'] == 'WAITING_BEFORE_FIRST_ADAM'
            and ready['actual_adam_writes'] == 0 and ready['original_adam'] == 500
            and ready['scheduler'] == 25 and ready['data_cursor'] == 13312
            and ready['sampled_trajectories'] == 2560 and ready['retained_training_updates_contributed'] == 0,
            'Not actual zero-Adam2560 live pause')
    identity = ready['identity']
    require(identity['code_commit'] == SOURCE and identity['job_id'] == job and identity['run_id'] == PurePosixPath(run).name
            and identity['native_binding_sha256'] == binding_sha
            and identity['material_manifest_sha256'] == manifest_sha,
            'Ready source/job/binding/material identity differs')
    require(ready['post_sampling_pause'] == manifest['post_sampling_pause'], 'Pause summary outside manifest')
    require(type(ready['post_sampling_pause']['driver_pid']) is int and ready['post_sampling_pause']['driver_pid']>0
            and str(ready['post_sampling_pause']['driver_start_ticks']).isdecimal(), 'Missing actual driver PID/start')
    workers = identity['post_sampling_worker_identities']
    require(len(workers) == 8 and {row['rank'] for row in workers} == set(range(8))
            and len({row['pid'] for row in workers}) == 8, 'Missing actual eight live worker identities')
    require(workers == ready['post_sampling_pause']['workers'], 'Ready worker summary differs')
    require(identity['post_sampling_worker_pause_sha256'] == ready['post_sampling_pause']['worker_pause_sha256']
            and identity['post_sampling_driver_complete_state_sha256'] == ready['post_sampling_pause']['driver_complete_state_sha256']
            and identity['post_sampling_pause_raw_sha256'] == ready['post_sampling_pause']['raw']['sha256'],
            'Complete pause digests not closed')
    for row in workers:
        require(all(key in row for key in WORKER_FIELDS), 'Incomplete worker state identity')
        require(row['scheduler_step'] == 25 and type(row['pid']) is int and row['pid'] > 0 and str(row['start_ticks']).isdecimal(),
                'Wrong worker scheduler/PID/start identity')
        require(all(SHA_PATTERN.fullmatch(value) for key, value in row.items() if key.endswith('_sha256')),
                'Invalid worker digest')
    return dict(job=job, run_id=identity['run_id'], actual_adam_writes=0,
                raw_state_sha256=identity['post_sampling_pause_raw_sha256'], worker_count=8,
                driver_pid=ready['post_sampling_pause']['driver_pid'], live_os_identity_recheck_required=True)


def error_summary(values):
    ordered = sorted(values)
    if not ordered:
        return dict(count=0, minimum=None, maximum=None, quantiles={})
    return dict(count=len(values), minimum=min(values), maximum=max(values),
                quantiles={str(q): ordered[round(q * (len(ordered) - 1))] for q in (0., .5, .9, .99, 1.)})


def cpu_analysis(store, inventory, turns, mapping, checks, ready, state_module_path, pad, receipts, binding, failure_config=None):
    # This path is only for reading our SHA-bound actual originals in a target CPU
    # process. Loading PT on GPU or running any inference is not implemented.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.dont_write_bytecode=True
    import torch
    require(not torch.cuda.is_initialized(), 'Unexpected CUDA before raw evidence CPU load')
    state_path = Path(state_module_path)
    # This is calibration_state_v1.py, not calibration_state metadata digest.
    require(sha(state_path) == STATE_MODULE_SHA, 'Frozen complete-state digest module changed')
    spec = importlib.util.spec_from_file_location('frozen_calibration_state_readonly', state_path)
    state_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(state_module)
    def load(relative):
        return torch.load(store.path(str(store.run / relative)), map_location='cpu', weights_only=False)
    batch = load('actual-rollout-2560.pt')
    require(batch['scope'] == SCOPE and batch['data_cursor']['cursor'] == 13312, 'Actual batch cursor/scope differs')
    tensors, non_tensors = batch['tensors'], batch['non_tensors']
    require(len(tensors['responses']) == 2560, 'Actual tensor rows not2560')
    source_rows = [int(value) for value in non_tensors['audit_trajectory_row']]
    require(len(source_rows) == 2560 and set(source_rows) == set(range(2560)), 'Real tensor row permutation incomplete')
    by_row = {source: index for index, source in enumerate(source_rows)}
    action_count_by_source = {}
    for source, step in mapping:
        action_count_by_source[source] = action_count_by_source.get(source, 0) + 1
    width = tensors['responses'].shape[1]
    valid = (tensors['attention_mask'][:, -width:] * tensors['info_mask'][:, -width:]).float()
    require(torch.equal(valid.bool(), tensors['loss_mask'].bool()), 'Original effective attention/info/loss mask differs')
    expected_weights = valid / valid.sum(-1, keepdim=True).clamp_min(1)
    require(torch.equal(tensors['policy_weights'], expected_weights), 'Original sequence_mean weights differ')
    require(bool((tensors['step_ids'][valid.bool()] >= 0).all()), 'Effective action lacks GiGPO step identity')
    require(all(bool(torch.isfinite(tensors[field][valid.bool()]).all()) for field in
                ('old_log_probs', 'ref_log_prob', 'advantages', 'policy_weights')), 'Nonfinite old/ref/advantage/weight')
    require(len(set(map(str, non_tensors['trajectory_uid']))) == 2560, 'Trajectory identity duplicates')
    for source in range(0, 2560, 5):
        group = [by_row[row] for row in range(source, source + 5)]
        require(len({str(non_tensors['index'][row]) for row in group}) == 1
                and len({str(non_tensors['uid'][row]) for row in group}) == 1, 'Explicit five-response question group differs')
    token_errors = {name: [] for name in ('prefix_old_repeated', 'prefix_old_sampler', 'full_old_prefix_old', 'full_old_sampler')}
    joint_errors = {name: [] for name in token_errors}
    action_errors, partial_actions, row_mappings = [], [], {}
    comparison_records = {}
    for serial, active in enumerate(turns):
        raws = [load(f'probabilities/train/call-{serial}-rank-{rank}.pt') for rank in range(8)]
        local_rows = [len(raw['tensors']['responses']) for raw in raws]
        require(len(set(local_rows)) == 1 and sum(local_rows) >= len(active)
                and sum(local_rows) - len(active) < 8, 'Unexpected DP padding/row mapping')
        for rank, raw in enumerate(raws):
            require(raw['rank'] == rank and raw['serial'] == serial and raw['phase'] == 'train', 'Raw turn/rank identity differs')
            parameters = raw['actual_sampling_parameters']
            require(parameters['temperature'] == parameters['top_p'] == parameters['n'] == 1
                    and parameters['top_k'] == -1 and parameters['repetition_penalty'] == 1
                    and parameters['presence_penalty'] == parameters['frequency_penalty'] == 0
                    and parameters['max_tokens'] == 500 and parameters['logprobs'] >= 1,
                    'Actual sampling distribution differs')
            require(state_module.state_digest(raw['repeated_probe_state_before']) ==
                    state_module.state_digest(raw['repeated_probe_state_after']), 'HF repeated observation changed complete state')
            sampled_width=raw['sampled_ids'].shape[1]
            response_width=raw['tensors']['responses'].shape[1]
            expected_raw_mask=(raw['tensors']['attention_mask'][:,-response_width:][:,:sampled_width].bool()
                               & raw['sampled_ids'].ne(pad))
            require(torch.equal(expected_raw_mask,raw['valid_comparison_mask']), 'Raw comparison mask differs')
            original= json.loads(store.path(str(store.run/f'probabilities/train/call-{serial}-rank-{rank}.json')).read_text())
            old_all=raw['tensors']['old_log_probs'][:,:sampled_width]
            repeated_all=raw['actor_repeated_log_probs'][:,:sampled_width]
            sampler_all=raw['sampled_log_probs']
            bad=(~torch.isfinite(old_all)|~torch.isfinite(repeated_all)|~torch.isfinite(sampler_all))&expected_raw_mask
            safe=expected_raw_mask&~bad
            diagnostics=dict(valid_tokens=int(expected_raw_mask.sum()),nonfinite_effective_count=int(bad.sum()),
                actor_old_max_abs_delta=float((old_all.double()-repeated_all.double())[safe].abs().max()) if bool(safe.any()) else 0.,
                actor_vllm_max_abs_delta=float((old_all.double()-sampler_all.double())[safe].abs().max()) if bool(safe.any()) else 0.)
            require(all(original[key]==value for key,value in diagnostics.items()),'Raw generated/padded/discarded token receipt differs')
            comparison_records[(serial, rank)] = dict(valid_tokens=0, nonfinite_effective_count=0,
                                                       actor_old_max_abs_delta=0., actor_vllm_max_abs_delta=0.)
        for index, action in enumerate(active):
            rank, local = divmod(index, local_rows[0])
            raw = raws[rank]; key = (action['source'], action['step_id']); original = mapping[key]
            count = len(action['retained_action_tokens']); row = by_row[action['source']]
            sampled = raw['sampled_ids'][local]
            engine_prefix = raw['actual_engine_prompt_token_ids'][local]
            raw_input = raw['tensors']; prompt_width = raw_input['prompts'].shape[1]
            prompt_mask = raw_input['attention_mask'][local, :prompt_width].bool()
            require(raw_input['prompts'][local][prompt_mask].tolist() == engine_prefix == original['engine_prompt_token_ids'],
                    'Engine input differs from actual effective prompt tokens')
            require(sampled.tolist() == original['sampled_tokens'][:len(sampled)], 'JSON sampled vector changed')
            require(all(value == pad for value in original['sampled_tokens'][len(sampled):]), 'Cross-rank padded sampled tail differs')
            raw_positions = raw_input['position_ids'][local]
            natural_prefix = list(range(len(engine_prefix)))
            prompt_positions_equal = raw_positions[:prompt_width][prompt_mask].tolist() == natural_prefix
            selected = tensors['step_ids'][row].eq(action['step_id']) & tensors['loss_mask'][row].bool()
            indices = selected.nonzero().flatten().tolist()
            require(tensors['responses'][row][selected].tolist() == action['retained_action_tokens'], 'Effective action tokens changed')
            require(len(indices) == count and (not indices or indices == list(range(indices[0], indices[0] + count))),
                    'One action has noncontiguous effective slots')
            full_processed_count = non_tensors['action_records'][row][action['step_id']]['action_tokens']
            require(len(non_tensors['action_records'][row]) == len(non_tensors['anchors'][row]) == action_count_by_source[action['source']],
                    'GiGPO action/anchor timeline changed')
            action_advantage = tensors['advantages'][row][selected]
            require(not count or bool(torch.equal(action_advantage, action_advantage[:1].expand_as(action_advantage))),
                    'GiGPO advantage broadcast differs within an action')
            if full_processed_count != count:
                partial_actions.append(dict(source=key[0], step_id=key[1], processed_action_tokens=full_processed_count,
                                            retained_action_tokens=count))
            old = raw_input['old_log_probs'][local, :count].double()
            repeated = raw['actor_repeated_log_probs'][local, :count].double()
            sampler = raw['sampled_log_probs'][local, :count].double()
            full = tensors['old_log_probs'][row][selected].double()
            require(len(old) == len(repeated) == len(sampler) == len(full) == count, 'Raw probability coverage incomplete')
            require(all(bool(torch.isfinite(value).all()) for value in (old, repeated, sampler, full)), 'Nonfinite effective raw probability')
            response_positions = tensors['position_ids'][row, -tensors['responses'].shape[1]:][selected].tolist()
            sample_positions = raw_positions[prompt_width:prompt_width + count].tolist()
            sample_mask = raw_input['attention_mask'][local, prompt_width:prompt_width + count].tolist()
            same_position = prompt_positions_equal and sample_positions == response_positions == list(range(len(engine_prefix), len(engine_prefix) + count))
            require(same_position == checks[key]['same_position_ids'], 'CPU position reconstruction disagrees with original summary')
            require(sample_mask == original['sampling_action_mask'], 'Raw effective attention mask differs')
            differences = dict(prefix_old_repeated=old-repeated, prefix_old_sampler=old-sampler,
                               full_old_prefix_old=full-old, full_old_sampler=full-sampler)
            for name, difference in differences.items():
                token_errors[name].append(difference.abs().clone())
                if count:
                    joint_errors[name].append(float(difference.sum()))
            check = checks[key]
            actual = dict(full_trajectory_vs_sample_prefix_old_max_abs=float((full-old).abs().max()) if count else 0.,
                          full_vs_prefix_old_joint_sum_delta=float((full-old).sum()),
                          full_vs_vllm_joint_sum_delta=float((full-sampler).sum()),
                          prefix_old_vs_vllm_joint_sum_delta=float((old-sampler).sum()))
            require(all(actual[field] == check[field] for field in actual), 'CPU recomputed joint/token summary differs')
            projected = comparison_records[(serial, rank)]
            projected['valid_tokens'] += count
            if count:
                projected['actor_old_max_abs_delta'] = max(projected['actor_old_max_abs_delta'],float((old-repeated).abs().max()))
                projected['actor_vllm_max_abs_delta'] = max(projected['actor_vllm_max_abs_delta'],float((old-sampler).abs().max()))
            action_errors.append(dict(source=key[0],step_id=key[1],effective_tokens=count,**actual))
            row_mappings.setdefault(row, []).extend(indices)
    for row in range(2560):
        indices = row_mappings[row]
        require(len(indices) == len(set(indices)) and sorted(indices) == tensors['loss_mask'][row].bool().nonzero().flatten().tolist(),
                'Effective policy slots not mapped exactly once')
    for record in receipts:
        require(all(record[key] == value for key,value in comparison_records[(record['serial'],record['rank'])].items()),
                'CPU recomputed retained worker receipt differs')
    pause_verified = False
    config=failure_config
    if ready is not None:
        pause = load('post-sampling-pause-state.pt')
        actual_workers = state_module.validate_pause_workers([entry['identity'] for entry in pause['workers']])
        identity = ready['identity'];summary = ready['post_sampling_pause']
        require(actual_workers == identity['post_sampling_worker_identities'], 'Raw actual eight-worker pause differs')
        for entry in pause['workers']:
            worker = dict(entry['identity']);digest = worker.pop('complete_state_sha256')
            require(state_module.state_digest(dict(identity=worker,metadata=entry['metadata'])) == digest,
                    'Raw worker complete state digest differs')
        driver = pause['driver']
        require(state_module.state_digest(actual_workers) == identity['post_sampling_worker_pause_sha256']
                and state_module.state_digest(driver) == identity['post_sampling_driver_complete_state_sha256'],
                'Raw complete state digests differ')
        require(driver['pid'] == summary['driver_pid'] and driver['start_ticks'] == summary['driver_start_ticks']
                and driver['global_steps'] == 26 and driver['actual_outer_updates'] == 0
                and driver['sampler']['cursor'] == 13312, 'Raw driver progression differs')
        algorithm=driver['algorithm_state']
        require(algorithm['method']=='gigpo'
                and algorithm['groups']=='stateless_per_rollout_with_step_ids_and_advantage_broadcast'
                and algorithm['parameters']==driver['config']['baselines']['gigpo'], 'GiGPO actual state/config grouping differs')
        parameters=algorithm['parameters']
        require(parameters['gamma']==.95 and parameters['omega']==1.,'GiGPO algorithm parameters differ')
        config=driver['config']
        require(state_module.state_digest(driver['fixed_batch']['tensors']) == state_module.state_digest(tensors)
                and state_module.state_digest(driver['fixed_batch']['non_tensors']) == state_module.state_digest(non_tensors),
                'Actual fixed2560 rollout differs from live driver pause')
        pause_verified = True
    require(config is not None,'CPU algorithm checks need actual bound resolved config or actual driver pause')
    algorithm_checks=validate_gigpo_batch(torch,tensors,non_tensors,valid,config,binding)
    require(not torch.cuda.is_initialized(), 'Unexpected CUDA during CPU raw evidence analysis')
    return dict(scope='RAW_ORIGINAL_CPU_RECOMPUTED_NO_TOLERANCE_NO_APPROVAL',partial_action_count=len(partial_actions),
                partial_action_examples=partial_actions[:MAX_EXAMPLES],complete_action_verified=not partial_actions,
                partially_retained_action_count=sum(0 < row['retained_action_tokens'] < row['processed_action_tokens'] for row in partial_actions),
                fully_discarded_action_count=sum(row['retained_action_tokens'] == 0 < row['processed_action_tokens'] for row in partial_actions),
                original_sequence_mean_weights_equal=True,explicit_five_response_groups_equal=True,
                state_group_and_episode_step_advantages_source_cpu_exactly_equal=True,
                advantage_broadcast_exact_no_allclose_tolerance=True,
                row_normalization='effective_action_token_mean_once_per_trajectory',
                optimizer_global_minibatch_rows=256,dp_ranks=8,per_rank_row_denominator=32,
                reference_full_old_finite_checked=True,reference_actual_sampling_prefix_probability_available=False,
                structural_action_semantics_decision_required=bool(partial_actions) or any(
                    row['sampled_tokens'][:len(row['retained_action_tokens'])]!=row['retained_action_tokens']
                    for row in mapping.values()),
                observation_anchor_semantic_identity_reconstructed=False,
                algorithm_configuration_checks=algorithm_checks,
                token_abs_errors={k:tensor_error_summary(v) for k,v in token_errors.items()},
                joint_signed_errors={k:error_summary(v) for k,v in joint_errors.items()},
                joint_abs_errors={k:error_summary([abs(x) for x in v]) for k,v in joint_errors.items()},
                live_pause_raw_digest_verified=pause_verified,action_examples=action_errors[:MAX_EXAMPLES])


def validate_actual_execution(job,run):
    require(isinstance(job,str) and job.isdecimal() and int(job)>0,'Explicit numeric actual job is required')
    path=PurePosixPath(run)
    require(path.is_absolute() and '..' not in path.parts and path.parent==PurePosixPath(str(ROOT/'runs'))
            and re.fullmatch(r'gigpo-e1-policy-safety-calibration25-job'+re.escape(job)+r'-[0-9]+',path.name),
            'Explicit actual run/job identity differs')
    require(platform.system()=='Linux' and socket.gethostname().split('.')[0]=='h100-8',
            'Actual evidence analysis runs only on h100-8; no Mac/mirror raw PT transfer')


def validate_binding_manifest(binding,manifest):
    require(manifest['scope']==SCOPE and manifest['code_commit']==SOURCE
            and manifest['restore_manifest_sha256']==RESTORE,'Wrong manifest source/restore/scope')
    require(binding['code_commit']==SOURCE and binding['restore_step']==25
            and binding['restore_manifest_sha256']==RESTORE and binding['common_manifest_sha256']==DATA
            and binding['calibration_phase']=='probability_probe_then_wait'
            and binding['target_old_policy_kl'] is None and binding['negative_tail_limit'] is None,
            'Wrong frozen binding or invented production thresholds')
    require(binding['status']=='REVIEWED_FOR_NATIVE_ONLY'
            and binding['readiness']=='READY_FOR_PROBABILITY_PROBE_THEN_WAIT_ONLY'
            and binding['node']=='h100-8' and binding['source_native_acceptance']['status']=='PASS'
            and binding['source_native_acceptance']['code_commit']==SOURCE,
            'Not the independently frozen source/native calibration binding')
    require(all(binding[key] is None for key in ('actor_recompute_abs_tolerance','actor_vllm_abs_tolerance',
            'actor_recompute_joint_abs_tolerance','actor_vllm_joint_abs_tolerance')),
            'Initial binding already selects probability tolerances')
    require(binding['prefix_mismatch_limit']==binding['action_token_mismatch_limit']==0,
            'Structural probability identities cannot use a tolerance')
    records=binding['files'];require(len(records)==len({r['path'] for r in records}),'Duplicate full-source binding paths')


def verify_probability_source_files(binding):
    paths=[Path(binding['helpers'])/name for name in ('calibration_driver_v1.py','calibration_worker_v1.py',
           'calibration_contract_v1.py','calibration_state_v1.py')]
    paths += [Path(binding['repo'])/relative for relative in ('search_r1/baselines/advantages.py',
        'search_r1/baselines/objectives.py','search_r1/baselines/runtime_audit.py','search_r1/baselines/policy_safety.py',
        'search_r1/baselines/rollout.py','verl/workers/actor/dp_actor.py','verl/workers/fsdp_workers.py')]
    by={r['path']:r for r in binding['files']}
    for path in paths:
        record=by.get(str(path))
        require(record is not None and path.is_file() and not path.is_symlink() and path.resolve()==path
                and path.stat().st_size==record['bytes'] and sha(path)==record['sha256'],
                'Frozen actual probability/source module differs: '+str(path))
    return [dict(path=str(path),sha256=by[str(path)]['sha256']) for path in paths]


def parse_proc_stat(text):
    fields=text.rsplit(')',1)[1].split()
    require(len(fields)>19 and fields[0] in ('R','S','D','T','t','I'),'Not a live readable leader state')
    return dict(state=fields[0],start_ticks=fields[19])


def cgroup_owned(text,job):
    paths=[]
    for line in text.splitlines():
        pieces=line.split(':',2)
        require(len(pieces)==3,'Unreadable cgroup identity')
        if pieces[1]=='' or 'freezer' in pieces[1].split(','):paths.append(pieces[2])
    marker='/slurm_h100-8/uid_0/job_'+job+'/step_0'
    return bool(paths) and all(path==marker or path.startswith(marker+'/') for path in paths)


def observe_one_live(identity,job):
    pid=identity['pid'];require(type(pid) is int and pid>0 and str(identity['start_ticks']).isdecimal(),'Invalid live PID/start')
    require(hasattr(os,'pidfd_open'),'Kernel pidfd identity required, no weaker fallback')
    fd=os.pidfd_open(pid,0)
    try:
        first=parse_proc_stat(Path(f'/proc/{pid}/stat').read_text())
        cg1=Path(f'/proc/{pid}/cgroup').read_text()
        info=Path(f'/proc/self/fdinfo/{fd}').read_text()
        fdpid=re.search(r'^Pid:\s*(\d+)\s*$',info,re.M)
        require(fdpid is not None and int(fdpid.group(1))==pid and not select.select([fd],[],[],0)[0],
                'Paused PID kernel identity already exited or changed')
        second=parse_proc_stat(Path(f'/proc/{pid}/stat').read_text())
        cg2=Path(f'/proc/{pid}/cgroup').read_text()
        require(first['start_ticks']==second['start_ticks']==str(identity['start_ticks'])
                and cg1==cg2 and cgroup_owned(cg2,job) and not select.select([fd],[],[],0)[0],
                'Paused process start/cgroup/liveness changed')
        return dict(identity,state_before=first['state'],state_after=second['state'],
            cgroup_sha256=hashlib.sha256(cg2.encode()).hexdigest(),pidfd_same_live_kernel_identity=True)
    finally:os.close(fd)


def observe_live_pause(identities,job):
    require(len(identities)==9 and len({r['pid'] for r in identities})==9,'Missing/duplicate eight workers plus driver')
    return [observe_one_live(row,job) for row in identities]


def tensor_error_summary(chunks):
    import torch
    if not chunks:return error_summary([])
    values=torch.cat(chunks);require(bool(torch.isfinite(values).all()),'Nonfinite probability error statistic')
    ordered=values.sort().values
    return dict(count=len(values),minimum=float(ordered[0]),maximum=float(ordered[-1]),
        quantiles={str(q):float(ordered[round(q*(len(ordered)-1))]) for q in (0.,.5,.9,.99,1.)})


def validate_gigpo_batch(torch,tensors,non_tensors,valid,config,binding):
    parameters=config['baselines']['gigpo'];actor=config['actor_rollout_ref']['actor']
    require(config['baselines']['method']=='gigpo' and parameters['gamma']==.95 and parameters['omega']==1.
            and config['baselines']['loss_reduction']=='sequence_mean','Actual GiGPO objective/config differs')
    require(actor['ppo_mini_batch_size']==256 and actor['ppo_micro_batch_size']==16 and actor['ppo_epochs']==1
            and actor['use_kl_loss'] is True and actor['kl_loss_coef']==.001 and actor['kl_loss_type']=='low_var_kl'
            and actor['entropy_coeff']==0 and actor['clip_ratio']==.2 and actor['grad_clip']==1.,
            'Actual calibration optimizer/global row objective differs')
    adv_path=Path(binding['repo'])/'search_r1/baselines/advantages.py'
    candidates=[r for r in binding['files'] if r['path']==str(adv_path)]
    require(len(candidates)==1 and adv_path.is_file() and not adv_path.is_symlink() and adv_path.resolve()==adv_path
            and adv_path.stat().st_size==candidates[0]['bytes'] and sha(adv_path)==candidates[0]['sha256'],
            'GiGPO frozen estimator source differs')
    spec=importlib.util.spec_from_file_location('frozen_gigpo_readonly_estimator',adv_path)
    estimator=importlib.util.module_from_spec(spec);spec.loader.exec_module(estimator)
    expected=estimator.gigpo(tensors['token_level_rewards'],valid,tensors['step_ids'],
        non_tensors['anchors'],non_tensors['uid'],parameters['gamma'],parameters['omega'],parameters['normalize'])
    require(torch.equal(expected,tensors['advantages']) and torch.equal(expected,tensors['returns']),
            'Full CPU state groups/episode-step advantage recomputation differs')
    from collections import Counter
    state_keys=[(str(uid),anchor) for uid,anchors in zip(non_tensors['uid'],non_tensors['anchors']) for anchor in anchors]
    groups=Counter(state_keys)
    return dict(gamma=parameters['gamma'],omega=parameters['omega'],normalize=parameters['normalize'],
        state_group_keys_count=len(state_keys),unique_state_groups=len(groups),
        shared_state_group_entries=sum(n for n in groups.values() if n>1),
        unique_question_uids=len(set(map(str,non_tensors['uid']))),global_sequence_mean_rows=256,
        dp_ranks=8,per_rank_row_denominator=32,actor_global_micro_rows=16,actor_per_rank_micro_rows=2,
        joint_action_scope='all effective tokens selected by original trajectory-row + step_id exactly once',
        old_reference_formal_input_identity_interface_required=True)


def write_aggregate_once(path,job,text):
    output=Path(path)
    require(output.is_absolute() and output.parent==ROOT/'logs' and output.resolve()==output
            and re.fullmatch(r'probability-analysis-job'+re.escape(job)+r'-v2-[A-Za-z0-9-]+\.json',output.name),
            'Aggregate output must be a new own-job JSON in original target logs')
    fd=os.open(str(output),os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o644)
    with os.fdopen(fd,'w') as stream:stream.write(text);stream.flush();os.fsync(stream.fileno())


def analyze(args):
    validate_actual_execution(args.job,args.run)
    run=PurePosixPath(args.run)
    require(Path(args.binding).parent==ROOT/'logs', 'Binding must be the original target logs artifact')
    failure=args.failure_context is not None
    require(args.manifest==str(run/'probability-material-manifest.json'),'Wrong original material location')
    require(failure or args.ready==str(run/'probability-identity-ready.json'),'Wrong original ready location')
    manifest=read_json(args.manifest,args.manifest_sha)
    binding=read_json(args.binding,args.binding_sha)
    ready=None if failure else read_json(args.ready,args.ready_sha)
    validate_binding_manifest(binding,manifest)
    validate_backend_binding(binding)
    source_files=verify_probability_source_files(binding)
    store=OriginalStore(args.run)
    records=manifest['files'];inventory={r['path']:r for r in records}
    require(len(inventory)==len(records), 'Duplicate material original path')
    for record in records:store.verify(record)
    for relative in REQUIRED:require(str(store.run/relative) in inventory,'Missing probability original: '+relative)
    def json_original(relative):return json.loads(store.path(str(store.run/relative)).read_text())
    backend = validate_backend_original(json_original('actual-vllm-backend-identity.json'), args.binding_sha)
    backend_pids = {row['rank']: row['pid'] for row in backend['actual_observations']}
    turns=json_original('training-prefix-turns.json');alignment=json_original('actual-probability-alignment.json')
    cross=json_original('full-old-versus-sampled-prefix.json')
    mapping,checks,structural=structural_analysis(alignment,turns,cross)
    expected={str(store.run/relative) for relative in REQUIRED}
    for serial in range(len(turns)):
        for rank in range(8):
            for suffix in ('pt','json'):expected.add(str(store.run/f'probabilities/train/call-{serial}-rank-{rank}.{suffix}'))
    pause=None;failure_evidence=None;failure_config=None;summary=None
    if failure:
        require(all(getattr(args,key) is not None for key in ('failure_context_sha','failure_review','failure_review_sha')),
                'Explicit failed-job context and independent zero-Adam review originals are required')
        failure_evidence,failure_config=validate_failure_context(args,binding,manifest,read_json,sha,require,ROOT,SOURCE,RESTORE)
    else:
        pause=validate_ready_json(ready,binding,args.binding_sha,args.manifest_sha,manifest,args.job,args.run)
        require(ready['material_manifest']==str(run/'probability-material-manifest.json')
                and ready['approval_path']==str(run/'probability-identity-approval.json'), 'Ready original manifest/approval locations differ')
        expected.update(str(store.run/relative) for relative in ('post-sampling-pause-state.pt','post-sampling-pause-state.json'))
        summary=json_original('post-sampling-pause-state.json')
        require({row['rank']: row['pid'] for row in summary['workers']} == backend_pids,
                'Actual backend worker identities differ from the live post-sampling pause')
        require(summary==ready['post_sampling_pause']
                and summary['raw']==inventory[str(run/'post-sampling-pause-state.pt')], 'Pause original summary/raw identity differs')
    require(set(inventory)==expected,'Missing or extra original manifest inventory')
    receipts=alignment['probability_receipts']
    require(len(receipts)==8*len(turns) and {(r['serial'],r['rank']) for r in receipts}==
            {(serial,rank) for serial in range(len(turns)) for rank in range(8)},'Worker turn receipt coverage incomplete')
    for record in receipts:
        relative=f"probabilities/train/call-{record['serial']}-rank-{record['rank']}"
        raw_record=json_original(relative+'.json')
        require(raw_record['raw']==inventory[str(run/(relative+'.pt'))]
                and record['raw']==raw_record['raw'],'Raw probability receipt identity mismatch')
        require(raw_record['source_commit']==SOURCE and raw_record['rank']==record['rank']
                and raw_record['serial']==record['serial'] and raw_record['rows']>0
                and raw_record['comparison_scope']=='SAME_PREFIX_SAME_TOKENS','Worker receipt source/mapping differs')
        require(raw_record['pid'] == backend_pids[raw_record['rank']],
                'Actual per-turn sampler receipt worker differs from the observed backend worker')
        require(raw_record['repeated_probability_probe_rng_unchanged']
                and raw_record['repeated_probability_probe_complete_state_unchanged']
                and raw_record['repeated_probe_identity_before']==raw_record['repeated_probe_identity_after'],
                'Repeated HF observation changed real state')
        require(record['raw_generated_token_diagnostics']=={key:raw_record[key] for key in
                ('valid_tokens','nonfinite_effective_count','actor_old_max_abs_delta','actor_vllm_max_abs_delta')},
                'Raw discarded/padded-token diagnostics outside original worker receipt')
    identities=[];live_before=None
    if not failure:
        identities=[dict(pid=w['pid'],start_ticks=w['start_ticks'],role='worker',rank=w['rank'])
                    for w in ready['post_sampling_pause']['workers']]
        identities.append(dict(pid=summary['driver_pid'],start_ticks=summary['driver_start_ticks'],role='driver'))
        live_before=observe_live_pause(identities,args.job)
    result=dict(status='BOUND_ORIGINAL_JSON_COVERAGE_AND_LIVE_OS_ONLY_NOT_PROBABILITY_ACCEPTANCE',job_id=args.job,
                run=args.run,source=SOURCE,manifest_sha256=args.manifest_sha,binding_sha256=args.binding_sha,
                ready_sha256=args.ready_sha,file_count=len(records),structural=structural,live_pause_json=pause,
                failed_terminal_zero_adam_diagnostic=failure_evidence,normal_ready_analysis=not failure,
                frozen_probability_source_files=source_files,
                actual_backend_original=backend,
                cpu_raw_analysis=None,numerical_tolerances_selected=False,probability_approval_generated=False,
                production_thresholds_selected=False,target_old_policy_kl=None,negative_tail_limit=None,
                actual_calibration_acceptance=False,gpu_used=False,live_os_before=live_before,
                live_complete_state_recheck_in_worker_not_performed_by_analyzer=True,
                raw_pt_read_location='h100-8_CPU_ONLY_NO_MIRROR_NO_MODEL_INFERENCE',bounded_examples_limit=MAX_EXAMPLES)
    if args.mode=='cpu':
        require(args.state_module is not None,'CPU raw mode needs explicit frozen calibration_state_v1.py path')
        require(args.state_module==str(Path(binding['helpers'])/'calibration_state_v1.py'), 'State module outside actual bound helpers')
        result['cpu_raw_analysis']=cpu_analysis(store,inventory,turns,mapping,checks,ready,args.state_module,binding['pad_token_id'],receipts,binding,failure_config)
        result['status']='BOUND_ACTUAL_CPU_RECOMPUTED_NO_TOLERANCE_NO_APPROVAL'
    # A read-only analysis does not assume files or process identities stayed unchanged.
    for record in records:store.verify(record)
    require(sha(args.binding)==args.binding_sha and sha(args.manifest)==args.manifest_sha
            and (failure or sha(args.ready)==args.ready_sha),
            'Actual frozen inputs changed during analysis')
    verify_probability_source_files(binding)
    result['live_os_after']=observe_live_pause(identities,args.job) if not failure else None
    if failure:
        validate_failure_context(args,binding,manifest,read_json,sha,require,ROOT,SOURCE,RESTORE)
        result['status']='ACTUAL_FAILED_ZERO_ADAM_CPU_DIAGNOSTICS_NO_LIVE_PAUSE_NO_APPROVAL'
    result['actual_probability_identity_accepted']=False
    return result


def main():
    sys.dont_write_bytecode=True
    os.environ['CUDA_VISIBLE_DEVICES']=''
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('job','manifest','manifest-sha','binding','binding-sha','run'):
        parser.add_argument('--'+name,required=True)
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--ready');modes.add_argument('--failure-context')
    parser.add_argument('--ready-sha');parser.add_argument('--failure-context-sha')
    parser.add_argument('--failure-review');parser.add_argument('--failure-review-sha')
    parser.add_argument('--mode',choices=('json','cpu'),default='json');parser.add_argument('--state-module')
    parser.add_argument('--output',help='Optional new aggregate JSON in original target logs; never raw PT')
    args=parser.parse_args()
    try:
        if args.ready is not None:require(args.ready_sha is not None,'Normal ready requires its actual original SHA')
        result=analyze(args)
    except Exception as error:
        result=dict(status='MATERIAL_REJECTED_OR_INCOMPLETE_NO_APPROVAL',job_id=args.job,error_type=type(error).__name__,
                    error=str(error),probability_approval_generated=False,gpu_used=False,
                    numerical_tolerances_selected=False,production_thresholds_selected=False)
        print(json.dumps(result,indent=2,allow_nan=False));return 2
    output=json.dumps(result,indent=2,allow_nan=False)+'\n'
    if args.output is not None:write_aggregate_once(args.output,args.job,output)
    print(output,end='');return 0


if __name__=='__main__':sys.exit(main())
