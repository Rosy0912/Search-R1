#!/usr/bin/env python3
"""Read saved job3332 evidence on H100; never forward, sample, approve or write.

Prepared for a separate nonauthor review before execution. Run with Python -B,
PYTHONDONTWRITEBYTECODE=1 and CUDA_VISIBLE_DEVICES=''. All inputs are fixed.
Only bounded scalar/metadata JSON is emitted. Full vocabulary vectors stay on
H100 and are neither copied into the report nor claimed present when absent.
"""
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import types

sys.dont_write_bytecode = True

RUN = Path('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct200-safety-20261006/runs/gigpo-e1-fresh200-live-job3332-1791429907029168081')
ROOT = RUN.parents[1]
RUNTIME = ROOT / 'tools/fresh200-continuation-v14-evidenceclone1'
BINDING = ROOT / 'logs/fresh200-evidence-repair3325-once-v1/BINDING.json'
BINDING_SHA = '2c161a96fd4f1313e1c5de935a98319bf85b9247b4c7632c73afac282ed500ec'
BINDING_BYTES = 20571536
SOURCE = '2ddbb0a1619ce7cde28b4c5be110ffe5bd11c963'
RAW = RUN / 'probabilities/train/call-1-rank-2.pt'
RAW_SHA = '6f2098179df870fda32fba5efae455941a90dff6b92df1ca1e23d659f5dafb95'
RAW_BYTES = 129709754
RECEIPT = RAW.with_suffix('.json')
PREFIX = RUN / 'training-prefix-turns.json'
PID = 1805679
START_TICKS = '2237476976'
RANK = 2
SERIAL = 1
TARGET_SOURCE = 661
TARGET_LOCAL = 17
TARGET_TOKEN = 1
TARGET_ID = 26865
TARGET_REQUEST = '498'
MATERIAL = RUN / 'probability-material-manifest.json'
MATERIAL_BYTES = 50432
MATERIAL_SHA = '7e43b667dcf0c8a4361f82b250ab3e8eaf534a20275e31eca689051c303359ab'
PREFIX_BYTES = 237823396
PREFIX_SHA = 'a8936be3c20293ac50a8fdfae8d2ec5aceeeeccf0476da7cc03a3dede4be4a0f'
HF_NORMALIZER_SOURCE = ROOT / 'repo-v11-2ddbb0a/verl/utils/torch_functional.py'
HF_NORMALIZER_SHA = '21eedf934c643b6c0a6809df89915c17b80796254920dcd4cbe2ec5850a7f575'
HF_NORMALIZER_BYTES = 18647
MAX_STDOUT_BYTES = 32768
MODULE_HASHES = {
    'calibration_state_v1': ('e37bcc7c26613ebffdde532150c1d6c735e31e2f30a9568a9a4daef9b19caf8f', 4455),
    'raw_hf_observation_v1': ('ac725dd92ad35563b9920f23ee3be8c154ff60f2f25a8e0bbc62bd6d457eb485', 11911),
    'raw_runner_observation_v1': ('6a023cec9c451159a298d66e04d31f239debcc94bc4b6b8ce2c4c32bdd0ca2dd', 17239),
    'raw_alignment_v1': ('a43943129d95175db198fe43e8c93414dde9f70da4ebb2699fe11029607cea0f', 13406),
    'decisive_normalizer_v1': ('1359eeab58b288a48d0f0535b104271340cdbd5ac916fad393439f9e32807fa3', 24996),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fingerprint(stat):
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def stream_receipt(path, expected_bytes=None, expected_sha=None):
    """Hash actual opened bytes in bounded blocks; refuse a changing inode."""
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        before = os.fstat(handle.fileno())
        count = 0
        while True:
            block = handle.read(4 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
            count += len(block)
        after = os.fstat(handle.fileno())
    require(fingerprint(before) == fingerprint(after) == fingerprint(path.stat()),
            'Input changed during stream hashing: ' + path.name)
    value = dict(path=str(path), bytes=count, sha256=digest.hexdigest())
    require(count == before.st_size, 'Incomplete input read: ' + path.name)
    if expected_bytes is not None:
        require(count == expected_bytes, 'Unexpected input bytes: ' + path.name)
    if expected_sha is not None:
        require(value['sha256'] == expected_sha, 'Unexpected input SHA256: ' + path.name)
    return value


def read_json_snapshot(path):
    # An atomically replaced prefix file may later gain more turns. Bind the
    # opened snapshot. The caller separately binds the fixed actual gate SHA.
    with path.open('rb') as handle:
        before = os.fstat(handle.fileno())
        content = handle.read()
        after = os.fstat(handle.fileno())
    require(fingerprint(before) == fingerprint(after) and len(content) == before.st_size,
            'JSON changed within opened snapshot: ' + path.name)
    return json.loads(content), dict(path=str(path), bytes=len(content),
                                    sha256=hashlib.sha256(content).hexdigest())


def read_verified_raw_bytes():
    """Deserialize these exact immutable bytes, rather than reopening a path."""
    with RAW.open('rb') as handle:
        before = os.fstat(handle.fileno())
        require(before.st_size == RAW_BYTES, 'Unexpected raw size before bounded read')
        content = handle.read(RAW_BYTES + 1)
        after = os.fstat(handle.fileno())
    require(fingerprint(before) == fingerprint(after) == fingerprint(RAW.stat()),
            'Raw file changed while capturing immutable bytes')
    require(type(content) is bytes and len(content) == RAW_BYTES
            and hashlib.sha256(content).hexdigest() == RAW_SHA, 'Unexpected immutable raw bytes or SHA256')
    return content


def ids_sha(ids):
    return hashlib.sha256(json.dumps(ids, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def verified_bounded_vector(statistics, selected_index, target):
    """Borrow an actual verified CPU vector; never emit its tensor values."""
    import torch
    matches = [item for item in statistics['bounded_full_vocab_vectors']
               if int(item['selected_flat_index']) == selected_index and int(item['target_id']) == target]
    require(len(matches) <= 1, 'Duplicate bounded selected vector metadata')
    if not matches:
        return None
    vector = matches[0].get('logits_fp32')
    require(isinstance(vector, torch.Tensor) and vector.device.type == 'cpu'
            and vector.dtype == torch.float32 and vector.ndim == 1
            and vector.shape[0] == statistics['vocabulary_size']
            and 0 <= target < vector.shape[0] and bool(torch.isfinite(vector).all()),
            'Bounded selected vector lacks the actual finite FP32 vocabulary tensor')
    indices = [i for i, position in enumerate(statistics['selected_flat_indices'])
               if int(position) == selected_index and int(statistics['target_ids'][i]) == target]
    require(len(indices) == 1
            and float(vector[target]) == float(statistics['pre_normalization_selected_logits_fp32'][indices[0]]),
            'Bounded actual vector selected logit differs from selected scalar')
    return vector


def normalizer_vector_summary(hf_vector, vllm_vector, target):
    """CPU arithmetic on saved vectors, if present; no model or normalization call."""
    import torch
    quantiles = torch.tensor([0., .5, .9, .99, 1.], dtype=torch.float64)

    def summary(vector):
        if vector is None:
            return dict(available=False, shape=None, count=None, values_emitted=False)
        # Temporary derived CPU arithmetic stays on H100. Original saved
        # vectors are never modified and no values are transferred to Mac.
        values = vector.double()
        return dict(available=True, shape=list(vector.shape), dtype=str(vector.dtype),
            count=vector.numel(), finite_count=int(torch.isfinite(vector).sum()),
            selected_target_logit=float(vector[target]), minimum=float(values.min()), maximum=float(values.max()),
            mean=float(values.mean()), logits_quantile_probabilities=[0., .5, .9, .99, 1.],
            logits_quantiles=[float(value) for value in torch.quantile(values, quantiles)], values_emitted=False)

    result = dict(scope='BOUNDED_ORIGINAL_SAVED_NORMALIZER_VECTORS_H100_CPU_STATISTICS_ONLY',
                  hf=summary(hf_vector), vllm=summary(vllm_vector), vector_values_emitted=False)
    if hf_vector is None or vllm_vector is None:
        result['paired_vocabulary_delta'] = dict(available=False,
            reason='At least one actual anomalous selected vector was not in the original bounded capture; no vector is reconstructed.')
        return result
    require(hf_vector.shape == vllm_vector.shape, 'Actual HF/vLLM saved vocabulary vector shapes differ')
    delta = hf_vector.double() - vllm_vector.double()
    absolute = delta.abs()
    average = delta.mean()
    result['paired_vocabulary_delta'] = dict(available=True, shape=list(delta.shape), count=delta.numel(),
        finite_count=int(torch.isfinite(delta).sum()), unequal_count=int((delta != 0).sum()),
        signed_minimum=float(delta.min()), signed_maximum=float(delta.max()), signed_mean=float(average),
        signed_population_std=float(delta.std(unbiased=False)), max_abs=float(absolute.max()),
        max_abs_vocabulary_id=int(absolute.argmax()), selected_target_delta=float(delta[target]),
        absolute_quantile_probabilities=[0., .5, .9, .99, 1.],
        absolute_quantiles=[float(value) for value in torch.quantile(absolute, quantiles)],
        max_abs_after_removing_constant_mean_shift=float((delta - average).abs().max()), values_emitted=False)
    return result


def load_checked_modules(binding):
    """Compile only exact frozen source bytes; never load an unbound .pyc."""
    prepared = {}
    for name, (sha, count) in MODULE_HASHES.items():
        path = RUNTIME / (name + '.py')
        entries = [entry for entry in binding['runtime_files'] if entry['path'] == str(path)]
        require(len(entries) == 1 and entries[0]['sha256'] == sha and entries[0]['bytes'] == count,
                'Helper is not exactly bound: ' + name)
        actual = stream_receipt(path, count, sha)
        content = path.read_bytes()
        require(len(content) == count and hashlib.sha256(content).hexdigest() == sha,
                'Helper bytes changed after hashing: ' + name)
        require(name not in sys.modules, 'Refuse a preloaded helper: ' + name)
        prepared[name] = (path, content, actual)
    modules, receipts = {}, []
    for name, (path, content, actual) in prepared.items():
        module = types.ModuleType(name)
        module.__file__ = str(path)
        module.__package__ = ''
        sys.modules[name] = module
        exec(compile(content, str(path), 'exec'), module.__dict__)
        modules[name] = module
        receipts.append(actual)
    return modules, receipts


def hf_token(raw, local, token_index):
    offset = 0
    found = None
    for serial, call in enumerate(raw['perturn_hf_flat_calls']):
        count = len(call['flat_micro']['responses'])
        if offset <= local < offset + count:
            require(found is None, 'Duplicate HF row ownership')
            found = (serial, call, local - offset)
        offset += count
    require(found is not None, 'Missing HF row ownership')
    serial, call, row = found
    norm = call['actual_normalizer_calls'][0]
    branch = norm['actual_normalizer_branch']
    require(branch in ('flash_attn_cross_entropy', 'torch_log_softmax'), 'Unknown actual HF normalizer branch')
    expected_kernel = 'logprobs_from_logits_flash_attn' if branch == 'flash_attn_cross_entropy' else 'logprobs_from_logits_naive'
    for item, name in ((norm['normalizer'], 'logprobs_from_logits'), (norm['actual_kernel'], expected_kernel)):
        require(item['source_path'] == str(HF_NORMALIZER_SOURCE) and item['source_sha256'] == HF_NORMALIZER_SHA
                and item['module'] == 'verl.utils.torch_functional' and item['name'] == name,
                'Actual HF normalizer callable/source differs from scientific2dd')
    indices = [i for i, item in enumerate(norm['selected_mapping'])
               if item['flat_row'] == row and item['action_token_index'] == token_index]
    require(len(indices) == 1, 'Missing/duplicate same-target HF normalizer observation')
    index = indices[0]
    flat = call['flat_micro']
    width = flat['input_ids'].shape[1] - flat['responses'].shape[1]
    prefix = flat['input_ids'][row, :width][flat['attention_mask'][row, :width].bool()].tolist()
    expected_position = len(prefix) - 1 + token_index
    kwargs = call['model_calls'][0]['kwargs']
    if call['use_remove_padding']:
        prediction_index = int(norm['selected_flat_indices'][index])
        derived_prediction_index = (sum(int(flat['attention_mask'][previous].sum()) for previous in range(row))
                                    + int(flat['attention_mask'][row, :width].sum()) - 1 + token_index)
        require(prediction_index == derived_prediction_index,
                'HF selected-flat index does not identify the derived packed prediction row')
        actual_position = int(kwargs['position_ids'].reshape(-1)[prediction_index])
        previous_id = int(kwargs['input_ids'].reshape(-1)[prediction_index])
        require(prediction_index + 1 < kwargs['input_ids'].numel(), 'HF prediction target crosses packed storage')
        actual_prediction_target = int(kwargs['input_ids'].reshape(-1)[prediction_index + 1])
    else:
        derived_prediction_index = row * flat['responses'].shape[1] + token_index
        require(int(norm['selected_flat_indices'][index]) == derived_prediction_index,
                'HF selected-flat index does not identify the derived dense prediction row')
        prediction_index = width - 1 + token_index
        actual_position = int(kwargs['position_ids'][row, prediction_index])
        previous_id = int(kwargs['input_ids'][row, prediction_index])
        actual_prediction_target = int(kwargs['input_ids'][row, prediction_index + 1])
    require(actual_position == expected_position, 'HF actual prediction position differs')
    expected_previous = prefix[-1] if token_index == 0 else int(raw['sampled_ids'][local, token_index - 1])
    require(previous_id == expected_previous, 'HF actual preceding input token differs')
    target = int(raw['sampled_ids'][local, token_index])
    require(int(norm['target_ids'][index]) == int(flat['responses'][row, token_index])
            == actual_prediction_target == target,
            'HF normalizer target differs from sampled token')
    logp = float(norm['actual_normalizer_selected_log_probs'][index])
    require(logp == float(raw['tensors']['old_log_probs'][local, token_index]),
            'Actual HF normalizer differs from saved old probability')
    vector = verified_bounded_vector(norm, int(norm['selected_flat_indices'][index]), target)
    return prefix, dict(flat_call_serial=serial, flat_row=row, target_id=target,
        selected_normalizer_flat_index=int(norm['selected_flat_indices'][index]),
        independently_derived_normalizer_flat_index=derived_prediction_index,
        selected_flat_index_and_prediction_target_independently_bound=True,
        actual_prediction_position=actual_position, preceding_input_token_id=previous_id,
        selected_logit=float(norm['pre_normalization_selected_logits_fp32'][index]),
        fp32_lse=float(norm['fp32_logsumexp'][index]), actual_normalizer_logp=logp,
        actual_normalizer_branch=norm['actual_normalizer_branch'],
        normalizer=norm['normalizer'], kernel=norm['actual_kernel'],
        logits_input_dtype=norm['logits_input_dtype'], normalizer_output_dtype=norm['actual_normalizer_output_dtype'],
        use_remove_padding=call['use_remove_padding'], use_cache=kwargs['use_cache'],
        bounded_full_vocab_vector_present=vector is not None,
        full_vocab_vector_emitted=False), vector


def vllm_token(raw, request_id, token_index, requests, modules):
    entry = requests[request_id]['tokens'][token_index]
    found = []
    runner = modules['raw_runner_observation_v1']
    for serial, call in enumerate(raw['actual_runner_observation']['calls']):
        packed = runner.validate_packed_runner_input_mapping(call)
        for sample in call['actual_sampler_calls']:
            for index, mapping in enumerate(sample['selected_mapping']):
                if mapping['request_id'] != request_id:
                    continue
                owner = runner.bind_selected_runner_input(call, mapping, packed)
                if owner['prediction_position'] != entry['prediction_position']:
                    continue
                require(mapping['target_id'] == entry['target_id'], 'vLLM same-position target differs')
                before_sha = sample['pre_processor_selected_rows_sha256']
                after_sha = sample['post_processor_selected_rows_sha256']
                require(re.fullmatch('[0-9a-f]{64}', before_sha) is not None and before_sha == after_sha,
                        'Selected pre/post processor vocabulary rows differ')
                require(sample['actual_normalizer_arguments'] == dict(dim=-1, dtype='torch.float32')
                        and sample['normalizer_path'] == 'vllm.model_executor.layers.sampler.torch.log_softmax'
                        and all(value is False for value in sample['actual_processor_flags'].values()),
                        'Actual sampler normalizer/processor branch differs')
                stats = sample['pre_normalization']
                require(int(stats['selected_flat_indices'][index]) == mapping['selected_logit_row']
                        and int(stats['target_ids'][index]) == mapping['target_id'] == entry['target_id'],
                        'vLLM statistics scalar index/target differs from selected row mapping')
                physical_index = int(call['selected_token_indices'][mapping['selected_logit_row']])
                require(physical_index == entry['runner_input_index']
                        and int(call['input_positions'][physical_index]) == owner['prediction_position']
                        == entry['prediction_position'] and owner['request_id'] == request_id
                        and owner['seq_id'] == mapping['seq_id'] == entry['seq_id'],
                        'vLLM selected scalar lacks exact physical request/sequence/prediction-position owner')
                expected_previous = (requests[request_id]['prefix'][-1] if token_index == 0
                                     else requests[request_id]['tokens'][token_index - 1]['target_id'])
                require(int(call['input_tokens'][physical_index]) == expected_previous,
                        'vLLM physical prediction input token differs from actual request context')
                require(float(stats['pre_normalization_selected_logits_fp32'][index]) == entry['selected_logit']
                        and float(stats['fp32_logsumexp'][index]) == entry['lse']
                        and float(sample['actual_normalizer_selected_log_probs'][index]) == entry['logp'],
                        'vLLM indexed selected scalars differ from exact-chain observation')
                vector = verified_bounded_vector(stats, entry['selected_logit_row'], entry['target_id'])
                found.append((dict(actual_runner_call_serial=serial, request_id=request_id,
                    seq_id=entry['seq_id'], prediction_position=entry['prediction_position'],
                    runner_input_index=entry['runner_input_index'], selected_logit_row=entry['selected_logit_row'],
                    target_id=entry['target_id'], selected_logit=entry['selected_logit'], fp32_lse=entry['lse'],
                    statistics_selected_flat_index=int(stats['selected_flat_indices'][index]),
                    statistics_target_id=int(stats['target_ids'][index]),
                    selected_statistics_and_physical_owner_independently_bound=True,
                    actual_normalizer_logp=entry['logp'], sequence_output_logp=entry['sequence_output_logp'],
                    completion_output_logp=entry['completion_output_logp'], raw_sampled_logp=entry['raw_sampled_logp'],
                    raw_sampled_dtype=entry['raw_sampled_dtype'], sampler_input_dtype=sample['input_dtype'],
                    normalizer_input_dtype=stats['logits_input_dtype'], normalizer_output_dtype=entry['dtype'],
                    normalizer=entry['normalizer'], normalizer_call_count=sample['actual_normalizer_call_count'],
                    normalizer_arguments=sample['actual_normalizer_arguments'],
                    actual_processor_flags=sample['actual_processor_flags'], actual_logits_processor=call['actual_logits_processor'],
                    processor_logits_unchanged=sample['processor_logits_unchanged'],
                    pre_processor_selected_rows_sha256=before_sha, post_processor_selected_rows_sha256=after_sha,
                    selected_probability_chain_exact=entry['selected_probability_chain_exact'],
                    selected_runner_request_sequence_owner_exact=entry['selected_runner_request_sequence_owner_exact'],
                    cache_scope=call['cache_scope'], decode_uses_cuda_graph=call['decode_uses_cuda_graph'],
                    bounded_full_vocab_vector_present=vector is not None, full_vocab_vector_emitted=False), vector))
    require(len(found) == 1, 'Missing/duplicate targeted vLLM normalizer row')
    return found[0]


def manifest_entry(manifest, path):
    matches = [item for item in manifest['files'] if item['path'] == str(path)]
    require(len(matches) == 1, 'Missing/duplicate original material path: ' + path.name)
    return matches[0]


def bound_json(manifest, path):
    entry = manifest_entry(manifest, path)
    value, receipt = read_json_snapshot(path)
    require(receipt == entry, 'Opened original JSON differs from frozen material: ' + path.name)
    return value, receipt


def main():
    require(os.environ.get('PYTHONDONTWRITEBYTECODE') == '1', 'Require PYTHONDONTWRITEBYTECODE=1')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Require an explicitly empty CUDA_VISIBLE_DEVICES')
    require(len(sys.argv) == 1, 'This fixed reader takes no path or execution arguments')
    binding_receipt = stream_receipt(BINDING, BINDING_BYTES, BINDING_SHA)
    binding, binding_snapshot = read_json_snapshot(BINDING)
    require(binding_snapshot == binding_receipt and binding['code_commit'] == SOURCE,
            'Binding snapshot/source identity differs')
    require(binding['calibration_phase'] == 'probability_probe_then_wait'
            and binding['pad_token_id'] == 151643 and binding['eos_token_id'] == 151645,
            'Frozen sampling binding differs')
    for key in ('actor_recompute_abs_tolerance', 'actor_vllm_abs_tolerance',
                'actor_recompute_joint_abs_tolerance', 'actor_vllm_joint_abs_tolerance'):
        require(binding[key] is None, 'Refuse a preselected numerical tolerance')
    normalizer_files = [item for item in binding['source_files'] if item['path'] == str(HF_NORMALIZER_SOURCE)]
    require(len(normalizer_files) == 1 and normalizer_files[0]['sha256'] == HF_NORMALIZER_SHA
            and normalizer_files[0]['bytes'] == HF_NORMALIZER_BYTES, 'Scientific2dd normalizer is not exactly bound')
    normalizer_source_receipt = stream_receipt(HF_NORMALIZER_SOURCE, HF_NORMALIZER_BYTES, HF_NORMALIZER_SHA)
    modules, helper_receipts = load_checked_modules(binding)
    material, material_snapshot = read_json_snapshot(MATERIAL)
    require(material_snapshot == dict(path=str(MATERIAL), bytes=MATERIAL_BYTES, sha256=MATERIAL_SHA)
            and material['code_commit'] == SOURCE and material['schema'] == 'RAW_ACTION_PROBABILITY_MATERIAL_V1'
            and material['actor_token_identity'] == 'sampled_action_prefix_v1',
            'Frozen actual material manifest identity differs')
    paths = [item['path'] for item in material['files']]
    require(len(paths) == len(set(paths)), 'Duplicate original material paths')
    require(manifest_entry(material, RAW) == dict(path=str(RAW), bytes=RAW_BYTES, sha256=RAW_SHA),
            'Target raw differs from frozen actual material manifest')
    require(manifest_entry(material, PREFIX) == dict(path=str(PREFIX), bytes=PREFIX_BYTES, sha256=PREFIX_SHA),
            'Original turn mapping differs from frozen actual material manifest')
    pause = material['post_sampling_pause']
    require(pause['status'] == 'POST_SAMPLING_BEFORE_FIRST_ADAM' and pause['actual_adam_writes'] == 0
            and pause['scheduler'] == 0 and pause['data_cursor'] == 512,
            'Hash-bound material is not the actual before-first-Adam gate')
    pause_workers = {item['rank']: item for item in pause['workers']}
    require(len(pause['workers']) == len(pause_workers) == 8 and set(pause_workers) == set(range(8))
            and len({item['pid'] for item in pause['workers']}) == 8,
            'Eight-rank actual paused-worker ownership differs')
    # Only the target PT is opened. These eight small original JSON receipts
    # close the DP partition against the hash-bound actual second turn.
    rank_receipts, rank_receipt_snapshots = [], []
    for rank in range(8):
        path = RAW.parent / ('call-' + str(SERIAL) + '-rank-' + str(rank) + '.json')
        value, snapshot = bound_json(material, path)
        worker = pause_workers[rank]
        require(value['source_commit'] == SOURCE and value['phase'] == 'train'
                and value['serial'] == SERIAL and value['rank'] == rank and value['pid'] == worker['pid']
                and value['rows'] > 0 and value['comparison_scope'] == 'SAME_PREFIX_SAME_TOKENS',
                'Actual serial1 DP receipt/process identity differs')
        for identity in (value['repeated_probe_identity_before'], value['repeated_probe_identity_after']):
            require(identity['rank'] == rank and identity['pid'] == worker['pid']
                    and identity['start_ticks'] == worker['start_ticks']
                    and identity['scheduler_step'] == 0 and identity['learning_rates'] == [0.0],
                    'Actual serial1 DP receipt start/process/zero-scheduler identity differs')
        rank_receipts.append(value)
        rank_receipt_snapshots.append(snapshot)
    chunk = rank_receipts[0]['rows']
    require(all(value['rows'] == chunk for value in rank_receipts), 'Actual eight-rank second-turn chunks differ')
    receipt, receipt_snapshot = rank_receipts[RANK], rank_receipt_snapshots[RANK]
    require(receipt['pid'] == PID and receipt['repeated_probe_identity_before']['start_ticks'] == START_TICKS,
            'Fixed actual rank2 process identity differs')
    decisive, decisive_snapshot = bound_json(material, RUN / 'actual-decisive-normalizer-evidence.json')
    largest = decisive['largest_positions']
    require(largest and largest[0]['source'] == TARGET_SOURCE and largest[0]['step_id'] == SERIAL
            and largest[0]['rank'] == RANK and largest[0]['token_index'] == TARGET_TOKEN
            and largest[0]['target_id'] == TARGET_ID and largest[0]['request_id'] == TARGET_REQUEST
            and all(abs(item['actual_log_prob_delta']) <= abs(largest[0]['actual_log_prob_delta']) for item in largest),
            'Hash-bound complete-gate reported largest coordinate differs')
    global_metadata_worst = largest[0]
    import torch
    raw_receipt = stream_receipt(RAW, RAW_BYTES, RAW_SHA)
    require(receipt['raw'] == raw_receipt and receipt['source_commit'] == SOURCE
            and receipt['phase'] == 'train' and receipt['serial'] == SERIAL
            and receipt['rank'] == RANK and receipt['pid'] == PID and receipt['rows'] == chunk,
            'Actual target serial1 receipt identity differs')
    # The exact hashed artifact was produced by the frozen trusted worker. Its
    # recorded NumPy RNG metadata requires general deserialization. CPU mapping
    # loads saved tensors; it performs no model forward, state restore or draw.
    verified_bytes = read_verified_raw_bytes()
    raw = torch.load(io.BytesIO(verified_bytes), map_location='cpu', weights_only=False)
    del verified_bytes
    require(raw['source_commit'] == SOURCE and raw['binding_sha256'] == BINDING_SHA
            and raw['pid'] == PID and raw['phase'] == 'train' and raw['serial'] == SERIAL and raw['rank'] == RANK,
            'Actual target raw serial1 identity differs')
    digest = modules['calibration_state_v1'].state_digest
    state_digests = []
    for state in (raw['repeated_probe_state_before'], raw['repeated_probe_state_after']):
        identity = state['identity']
        require(identity['rank'] == RANK and identity['pid'] == PID and identity['start_ticks'] == START_TICKS
                and identity['scheduler_step'] == 0 and identity['learning_rates'] == [0.0],
                'Repeated probe process/zero-scheduler identity differs')
        without_complete = {key: value for key, value in identity.items() if key != 'complete_state_sha256'}
        require(digest(dict(identity=without_complete, metadata=state['metadata'])) == identity['complete_state_sha256'],
                'Raw complete probe-state digest differs')
        state_digests.append(digest(state))
    require(state_digests[0] == state_digests[1]
            and receipt['repeated_probability_probe_rng_unchanged'] is True
            and receipt['repeated_probability_probe_complete_state_unchanged'] is True
            and receipt['repeated_probe_identity_before'] == raw['repeated_probe_state_before']['identity']
            and receipt['repeated_probe_identity_after'] == raw['repeated_probe_state_after']['identity'],
            'Actual same-model repeated probe changed complete state')
    modules['raw_alignment_v1'].validate_real_sample(raw, binding['pad_token_id'], binding['eos_token_id'])
    requests = modules['decisive_normalizer_v1']._index_engine_observation(raw)
    original_turns, prefix_snapshot = bound_json(material, PREFIX)
    require(isinstance(original_turns, list) and len(original_turns) > SERIAL
            and len(original_turns[0]) == 2560,
            'Missing original actual turn source closure')
    initial_sources = [item['source'] for item in original_turns[0]]
    require(len(set(initial_sources)) == len(initial_sources) and set(initial_sources) == set(range(2560))
            and all(item['step_id'] == 0 for item in original_turns[0]),
            'Original actual trajectory source closure differs')
    turn = original_turns[SERIAL]
    sources = [item['source'] for item in turn]
    require(len(set(sources)) == len(sources) and set(sources).issubset(set(initial_sources))
            and all(item['step_id'] == SERIAL for item in turn)
            and len(turn) <= 8 * chunk and 0 <= 8 * chunk - len(turn) < 8,
            'Original second-turn source/step/DP padding closure differs')
    target_indices = [index for index, item in enumerate(turn) if item['source'] == TARGET_SOURCE]
    require(len(target_indices) == 1, 'Target source is absent or duplicated in original second turn')
    target_index = target_indices[0]
    derived_rank, derived_local = divmod(target_index, chunk)
    require(derived_rank == RANK and derived_local == TARGET_LOCAL,
            'Actual second-turn target source does not belong to fixed rank2/local17')
    partition_begin, partition_end = RANK * chunk, (RANK + 1) * chunk
    partition = turn[partition_begin:partition_end]
    require(len(partition) == chunk and len(raw['sampled_ids']) == chunk,
            'Target rank2 actual source partition or raw row count differs')
    width = raw['sampled_ids'].shape[1]
    support = raw['sampled_mask'][:, :width].bool()
    old = raw['tensors']['old_log_probs'][:, :width].double()
    sampled = raw['sampled_log_probs'].double()
    repeated = raw['actor_repeated_log_probs'][:, :width].double()
    require(old.shape == sampled.shape == repeated.shape == support.shape and bool(support.any()),
            'Actual probability/support shapes differ or empty')
    require(bool(torch.isfinite(old[support]).all() and torch.isfinite(sampled[support]).all()
                 and torch.isfinite(repeated[support]).all()), 'Nonfinite actual sampled-support probability')
    signed_delta = old - sampled
    absolute = signed_delta.abs()
    retained_lengths = []
    for local, observed in enumerate(partition):
        ids = observed['retained_action_tokens']
        length = int(raw['sampled_lengths'][local])
        require(observed['step_id'] == SERIAL and observed['raw_actor_record']['step_id'] == SERIAL
                and len(ids) <= length and ids == raw['sampled_ids'][local, :len(ids)].tolist()
                and observed['raw_actor_record']['actor_action_tokens'] == ids
                and observed['raw_actor_record']['sampled_length'] == length,
                'Original second-turn rank2 retained actor/sample support differs')
        prompt_width = raw['tensors']['prompts'].shape[1]
        attention = raw['tensors']['attention_mask'][local, :prompt_width].bool()
        actual_prefix = raw['tensors']['prompts'][local][attention].tolist()
        actual_positions = raw['tensors']['position_ids'][local, :prompt_width][attention].tolist()
        require(actual_prefix == raw['actual_engine_prompt_token_ids'][local]
                == observed['generation_prefix'] == observed['raw_actor_record']['generation_prefix']
                and actual_positions == list(range(len(actual_prefix))),
                'Original second-turn source/HF/engine prefix/positions differ')
        require(not ids or ids[-1] != binding['eos_token_id'], 'Terminal EOS has retained action mass')
        retained_lengths.append(len(ids))
    retained_support = torch.arange(width)[None, :] < torch.tensor(retained_lengths)[:, None]
    require(not bool((retained_support & ~support).any()), 'Retained support exceeds actual sample support')

    def maximum(mask):
        if not bool(mask.any()):
            return None
        index = int(absolute.masked_fill(~mask, -1.).reshape(-1).argmax())
        return divmod(index, width)

    retained_max = maximum(retained_support)
    require(retained_max == (TARGET_LOCAL, TARGET_TOKEN),
            'Actual target raw retained maximum is not the complete-gate reported coordinate')
    local, token = TARGET_LOCAL, TARGET_TOKEN
    observed = partition[local]
    retained_count = retained_lengths[local]
    require(observed['source'] == TARGET_SOURCE and token < retained_count
            and int(raw['sampled_ids'][local, token]) == TARGET_ID,
            'Fixed token is not retained by actual source661 step1')
    request_output = raw['actual_runner_observation']['request_outputs'][local]
    request_id = request_output['request_id']
    require(request_id == TARGET_REQUEST, 'Actual local17 request is not request498')
    hf_prefix, hf, hf_vector = hf_token(raw, local, token)
    engine_prefix = raw['actual_engine_prompt_token_ids'][local]
    require(hf_prefix == engine_prefix == observed['generation_prefix']
            == observed['raw_actor_record']['generation_prefix'] == requests[request_id]['prefix'],
            'Actual prepared HF/engine/source generation prefixes differ')
    vllm, vllm_vector = vllm_token(raw, request_id, token, requests, modules)
    require(hf['target_id'] == vllm['target_id'] == TARGET_ID
            and hf['actual_prediction_position'] == vllm['prediction_position']
            and vllm['seq_id'] == 498 and vllm['runner_input_index'] == 17 and vllm['selected_logit_row'] == 17,
            'Fixed same-target actual HF/vLLM physical identity differs')
    actual_delta = hf['actual_normalizer_logp'] - vllm['actual_normalizer_logp']
    selected_delta = hf['selected_logit'] - vllm['selected_logit']
    lse_delta = hf['fp32_lse'] - vllm['fp32_lse']
    hf_residual = hf['actual_normalizer_logp'] - (hf['selected_logit'] - hf['fp32_lse'])
    vllm_residual = vllm['actual_normalizer_logp'] - (vllm['selected_logit'] - vllm['fp32_lse'])
    require(actual_delta == float(signed_delta[local, token]) == global_metadata_worst['actual_log_prob_delta']
            and hf['selected_logit'] == global_metadata_worst['hf']['selected_logit']
            and hf['fp32_lse'] == global_metadata_worst['hf']['lse']
            and hf['actual_normalizer_logp'] == global_metadata_worst['hf']['logp']
            and vllm['selected_logit'] == global_metadata_worst['vllm']['selected_logit']
            and vllm['fp32_lse'] == global_metadata_worst['vllm']['lse']
            and vllm['actual_normalizer_logp'] == global_metadata_worst['vllm']['logp']
            and vllm['prediction_position'] == global_metadata_worst['vllm']['prediction_position'],
            'Exact target raw normalizer scalars differ from hash-bound complete-gate ledger')
    target = dict(local_row=local, source=TARGET_SOURCE, step_id=SERIAL, token_index=token,
        original_turn_index=target_index, actual_dp_rank=derived_rank, actual_dp_local=derived_local,
        source_second_turn_dp_owner_independently_bound=True,
        real_sampled_length=int(raw['sampled_lengths'][local]), retained_action_length=retained_count,
        token_in_retained_action=True, token_is_terminal_eos=False,
        signed_saved_hf_minus_sampled_logp=float(signed_delta[local, token]),
        absolute_saved_hf_sampled_delta=float(absolute[local, token]),
        repeated_saved_hf_logp=float(repeated[local, token]),
        repeated_hf_minus_old_logp=float(repeated[local, token] - old[local, token]),
        actual_hf_engine_generation_prefix_exact=True, prefix_length=len(hf_prefix),
        prefix_sha256=ids_sha(hf_prefix), hf=hf, vllm=vllm,
        decomposition=dict(actual_normalizer_hf_minus_vllm=actual_delta,
            selected_logit_delta=selected_delta, fp32_lse_delta=lse_delta,
            fp32_formula_delta=selected_delta - lse_delta,
            hf_normalizer_minus_fp32_formula=hf_residual,
            vllm_normalizer_minus_fp32_formula=vllm_residual,
            normalizer_delta_minus_fp32_formula_delta=actual_delta - (selected_delta - lse_delta),
            residual_identity_error=actual_delta - (selected_delta - lse_delta + hf_residual - vllm_residual)),
        saved_normalizer_full_vocab_summary=normalizer_vector_summary(hf_vector, vllm_vector, TARGET_ID),
        retained_action_joint_sums=dict(effective_tokens=retained_count,
            hf_perturn_old_sum=float(old[local, :retained_count].sum()),
            sampled_sum=float(sampled[local, :retained_count].sum()),
            signed_hf_old_minus_sampled_sum=float(signed_delta[local, :retained_count].sum()),
            signed_repeated_hf_minus_old_sum=float((repeated[local, :retained_count] - old[local, :retained_count]).sum()),
            scope='ACTUAL_SOURCE661_STEP1_RETAINED_PREFIX_SAVED_PERTURN_HF_NOT_COMPLETE_ALL_TURN_HF'))
    measured_old = float((old[support] - repeated[support]).abs().max())
    measured_vllm = float(absolute[support].max())
    require(int(support.sum()) == receipt['valid_tokens'] and receipt['nonfinite_effective_count'] == 0
            and measured_old == receipt['actor_old_max_abs_delta']
            and measured_vllm == receipt['actor_vllm_max_abs_delta'], 'Independent raw diagnostics differ from target receipt')
    result = dict(schema='JOB3332_ACTUAL_GLOBALWORST_READONLY_DIAGNOSTIC_V4',
        status='ACTUAL_SAVED_EVIDENCE_OBSERVATION_ONLY_NOT_ACCEPTANCE', job_id='3332', run=str(RUN),
        source_commit=SOURCE, native_binding_sha256=BINDING_SHA, rank=RANK, serial=SERIAL, phase='train', pid=PID,
        binding=binding_receipt, raw=raw_receipt, actual_target_receipt=receipt_snapshot,
        actual_material_manifest=material_snapshot, actual_decisive_normalizer_metadata=decisive_snapshot,
        actual_second_turn_dp_receipts=rank_receipt_snapshots,
        actual_second_turn_source_dp_closure=dict(turn_serial=SERIAL, sources=len(sources), unique_sources=len(set(sources)),
            initial_source_count=len(initial_sources), actual_eight_rank_chunk=chunk,
            padded_rows=8 * chunk - len(turn), target_rank_partition_begin=partition_begin,
            target_rank_partition_end_exclusive=partition_end, original_source_index=target_index,
            target_rank=derived_rank, target_local=derived_local,
            first_turn_source_arithmetic_used=False, actual_source_list_used=True),
        actual_before_first_adam_snapshot=dict(status=pause['status'], actual_adam_writes=pause['actual_adam_writes'],
            scheduler=pause['scheduler'], data_cursor=pause['data_cursor'], live_process_requeried=False,
            training_failure_or_job_cancellation_inferred=False),
        training_prefix_snapshot=prefix_snapshot, frozen_helper_receipts=helper_receipts,
        scientific_normalizer_source=normalizer_source_receipt,
        raw_complete_probe_state_unchanged=True, repeated_probe_state_digests=state_digests,
        deserialized_bytes_exactly_sha_bound=True, raw_deserialization='CPU_ONLY_FROM_SAME_VERIFIED_IMMUTABLE_BYTES',
        sampled_support_tokens=int(support.sum()), retained_second_turn_rank2_support_tokens=int(retained_support.sum()),
        hf_repeat_max_abs_delta=measured_old, sampled_support_hf_vllm_max_abs_delta=measured_vllm,
        retained_support_hf_vllm_max_abs_delta=float(absolute[retained_support].max()) if bool(retained_support.any()) else None,
        reported_complete_gate_globalworst_verified_at_target_raw=True,
        target_raw_retained_maximum=target,
        probability_accepted=False, approval_emitted=False, selected_tolerances=None,
        production_thresholds_selected=False, added_forward_calls=0, added_sampling_calls=0,
        rng_draws=0, state_restores=0, remote_artifact_writes=0, full_vocab_vectors_emitted=False,
        limitations=['Only actual serial1 rank2 PT is opened; globalworst identity comes from the hash-bound complete-gate ledger, not an independent scan of all raw PTs.',
                     'Retained action sums use saved perturn HF; this reader does not independently reload complete-action old/current/reference evidence.',
                     'Original training-prefix-turns and all eight second-turn JSON receipts are exact frozen-manifest inputs; no first-turn source arithmetic is used.',
                     'FP32 decomposition residuals identify a source of difference and are not parity acceptance.',
                     'Full-vocabulary statistics are available only if both exact target vectors were present in the original bounded capture; missing vectors are not reconstructed.',
                     'Saved source identity is bound to the actual raw and frozen binding; no new git checkout audit or live process query is performed.',
                     'The actual job snapshot is before first Adam; numerical admission remains BLOCK pending independent justification and no approval is emitted.'])
    result['raw_after_inspection'] = stream_receipt(RAW, RAW_BYTES, RAW_SHA)
    require(result['raw_after_inspection'] == raw_receipt, 'Raw changed during inspection')
    return result


if __name__ == '__main__':
    try:
        payload = json.dumps(main(), ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n'
        require(len(payload.encode('utf-8')) <= MAX_STDOUT_BYTES, 'Scalar metadata output exceeds bound')
        sys.stdout.write(payload)
    except BaseException as error:
        message = dict(schema='JOB3332_ACTUAL_GLOBALWORST_READONLY_DIAGNOSTIC_V4',
            status='READONLY_DIAGNOSTIC_REFUSED_NO_ACCEPTANCE', job_id='3332', run=str(RUN),
            error_type=type(error).__name__, error=str(error)[:1000], probability_accepted=False,
            approval_emitted=False, selected_tolerances=None, remote_artifact_writes=0,
            full_vocab_vectors_emitted=False)
        payload = json.dumps(message, ensure_ascii=False, allow_nan=False) + '\n'
        if len(payload.encode('utf-8')) > MAX_STDOUT_BYTES:
            payload = '{"status":"READONLY_DIAGNOSTIC_REFUSED_OUTPUT_BOUND","probability_accepted":false}\n'
        sys.stdout.write(payload)
        sys.exit(2)
