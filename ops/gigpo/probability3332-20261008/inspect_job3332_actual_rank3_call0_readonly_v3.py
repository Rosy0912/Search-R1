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
RAW = RUN / 'probabilities/train/call-0-rank-3.pt'
RAW_SHA = '313d2902783e1e0e0fd2ff1560565a82cff92bd74cb0d900201f0acbd9082085'
RAW_BYTES = 91213434
RECEIPT = RAW.with_suffix('.json')
PREFIX = RUN / 'training-prefix-turns.json'
PID = 1805680
START_TICKS = '2237476977'
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
    # opened snapshot, including its first turn, without asserting it is final.
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


def checked_vector_presence(statistics, selected_index, target):
    """Validate an existing saved CPU vector in place, never copy or emit it."""
    import torch
    matches = [item for item in statistics['bounded_full_vocab_vectors']
               if int(item['selected_flat_index']) == selected_index and int(item['target_id']) == target]
    require(len(matches) <= 1, 'Duplicate bounded selected vector metadata')
    if not matches:
        return False
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
    return True


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
    vector_present = checked_vector_presence(norm, int(norm['selected_flat_indices'][index]), target)
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
        bounded_full_vocab_vector_present=vector_present,
        full_vocab_vector_emitted=False)


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
                present = checked_vector_presence(stats, entry['selected_logit_row'], entry['target_id'])
                found.append(dict(actual_runner_call_serial=serial, request_id=request_id,
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
                    bounded_full_vocab_vector_present=present, full_vocab_vector_emitted=False))
    require(len(found) == 1, 'Missing/duplicate targeted vLLM normalizer row')
    return found[0]


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
    import torch
    raw_receipt = stream_receipt(RAW, RAW_BYTES, RAW_SHA)
    receipt, receipt_snapshot = read_json_snapshot(RECEIPT)
    require(receipt['raw'] == raw_receipt and receipt['source_commit'] == SOURCE
            and receipt['phase'] == 'train' and receipt['serial'] == 0
            and receipt['rank'] == 3 and receipt['pid'] == PID and receipt['rows'] == 320,
            'Actual call0 receipt identity differs')
    # The exact hashed artifact was produced by the frozen trusted worker. Its
    # recorded NumPy RNG metadata requires general deserialization. CPU mapping
    # loads saved tensors; it performs no model forward, state restore or draw.
    verified_bytes = read_verified_raw_bytes()
    raw = torch.load(io.BytesIO(verified_bytes), map_location='cpu', weights_only=False)
    del verified_bytes
    require(raw['source_commit'] == SOURCE and raw['binding_sha256'] == BINDING_SHA
            and raw['pid'] == PID and raw['phase'] == 'train' and raw['serial'] == 0 and raw['rank'] == 3,
            'Actual raw call0 identity differs')
    digest = modules['calibration_state_v1'].state_digest
    state_digests = []
    for state in (raw['repeated_probe_state_before'], raw['repeated_probe_state_after']):
        identity = state['identity']
        require(identity['rank'] == 3 and identity['pid'] == PID and identity['start_ticks'] == START_TICKS
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
    turns, prefix_snapshot = read_json_snapshot(PREFIX)
    require(isinstance(turns, list) and turns and len(turns[0]) == 2560,
            'Missing complete actual first-turn prefix mapping')
    first = turns[0]
    require({item['source'] for item in first} == set(range(2560)), 'First-turn source coverage differs')
    require(len(raw['sampled_ids']) == 320, 'Rank3 actual DP row partition differs')
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
    for local in range(320):
        source = 960 + local
        observed = first[source]
        ids = observed['retained_action_tokens']
        length = int(raw['sampled_lengths'][local])
        require(observed['source'] == source and observed['step_id'] == 0
                and len(ids) <= length and ids == raw['sampled_ids'][local, :len(ids)].tolist()
                and observed['raw_actor_record']['actor_action_tokens'] == ids
                and observed['raw_actor_record']['sampled_length'] == length,
                'Actual rank3 first-turn retained action mapping differs')
        require(not ids or ids[-1] != binding['eos_token_id'], 'Terminal EOS has retained action mass')
        retained_lengths.append(len(ids))
    retained_support = torch.arange(width)[None, :] < torch.tensor(retained_lengths)[:, None]
    require(not bool((retained_support & ~support).any()), 'Retained support exceeds actual sample support')

    def maximum(mask):
        if not bool(mask.any()):
            return None
        index = int(absolute.masked_fill(~mask, -1.).reshape(-1).argmax())
        return divmod(index, width)

    def targeted(coordinate):
        if coordinate is None:
            return None
        local, token = coordinate
        source = 960 + local
        observed = first[source]
        retained_count = retained_lengths[local]
        request_output = raw['actual_runner_observation']['request_outputs'][local]
        request_id = request_output['request_id']
        hf_prefix, hf = hf_token(raw, local, token)
        engine_prefix = raw['actual_engine_prompt_token_ids'][local]
        require(hf_prefix == engine_prefix == observed['generation_prefix']
                == observed['raw_actor_record']['generation_prefix'] == requests[request_id]['prefix'],
                'Actual HF/engine/source generation prefixes differ')
        vllm = vllm_token(raw, request_id, token, requests, modules)
        require(hf['target_id'] == vllm['target_id']
                and hf['actual_prediction_position'] == vllm['prediction_position'],
                'Same-target HF/vLLM identity differs')
        actual_delta = hf['actual_normalizer_logp'] - vllm['actual_normalizer_logp']
        selected_delta = hf['selected_logit'] - vllm['selected_logit']
        lse_delta = hf['fp32_lse'] - vllm['fp32_lse']
        hf_residual = hf['actual_normalizer_logp'] - (hf['selected_logit'] - hf['fp32_lse'])
        vllm_residual = vllm['actual_normalizer_logp'] - (vllm['selected_logit'] - vllm['fp32_lse'])
        return dict(local_row=local, source=source, step_id=0, token_index=token,
            real_sampled_length=int(raw['sampled_lengths'][local]), retained_action_length=retained_count,
            token_in_retained_action=token < retained_count,
            token_is_terminal_eos=(token == int(raw['sampled_lengths'][local]) - 1
                                   and hf['target_id'] == binding['eos_token_id']),
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
            retained_action_joint_sums=dict(effective_tokens=retained_count,
                hf_perturn_old_sum=float(old[local, :retained_count].sum()),
                sampled_sum=float(sampled[local, :retained_count].sum()),
                signed_hf_old_minus_sampled_sum=float(signed_delta[local, :retained_count].sum()),
                signed_repeated_hf_minus_old_sum=float((repeated[local, :retained_count] - old[local, :retained_count]).sum()),
                scope='ACTUAL_FIRST_TURN_RETAINED_PREFIX_NOT_COMPLETE_ALL_TURN_HF_ACTION_CHECK'))

    sampled_max = maximum(support)
    retained_max = maximum(retained_support)
    measured_old = float((old[support] - repeated[support]).abs().max())
    measured_vllm = float(absolute[support].max())
    require(int(support.sum()) == receipt['valid_tokens'] and receipt['nonfinite_effective_count'] == 0
            and measured_old == receipt['actor_old_max_abs_delta']
            and measured_vllm == receipt['actor_vllm_max_abs_delta'], 'Independent raw diagnostics differ from call0 receipt')
    result = dict(schema='JOB3332_ACTUAL_RANK3_CALL0_READONLY_DIAGNOSTIC_V3',
        status='ACTUAL_SAVED_EVIDENCE_OBSERVATION_ONLY_NOT_ACCEPTANCE', job_id='3332', run=str(RUN),
        source_commit=SOURCE, native_binding_sha256=BINDING_SHA, rank=3, serial=0, phase='train', pid=PID,
        binding=binding_receipt, raw=raw_receipt, call0_receipt=receipt_snapshot,
        training_prefix_snapshot=prefix_snapshot, frozen_helper_receipts=helper_receipts,
        scientific_normalizer_source=normalizer_source_receipt,
        raw_complete_probe_state_unchanged=True, repeated_probe_state_digests=state_digests,
        deserialized_bytes_exactly_sha_bound=True, raw_deserialization='CPU_ONLY_FROM_SAME_VERIFIED_IMMUTABLE_BYTES',
        sampled_support_tokens=int(support.sum()), retained_first_turn_support_tokens=int(retained_support.sum()),
        hf_repeat_max_abs_delta=measured_old, sampled_support_hf_vllm_max_abs_delta=measured_vllm,
        retained_support_hf_vllm_max_abs_delta=float(absolute[retained_support].max()) if bool(retained_support.any()) else None,
        sampled_support_maximum=targeted(sampled_max), retained_support_maximum=targeted(retained_max),
        probability_accepted=False, approval_emitted=False, selected_tolerances=None,
        production_thresholds_selected=False, added_forward_calls=0, added_sampling_calls=0,
        rng_draws=0, state_restores=0, remote_artifact_writes=0, full_vocab_vectors_emitted=False,
        limitations=['Only actual serial0 rank3 is inspected; full eight-rank all-turn gate remains pending.',
                     'Retained action sums use saved perturn HF; complete-action old/current/reference parity is pending.',
                     'First-turn prefix file is an opened actual snapshot; it may later contain additional turns.',
                     'FP32 decomposition residuals identify a source of difference and are not parity acceptance.',
                     'Bounded original vectors are checked only for presence; no vector values are emitted.',
                     'Saved source identity is bound to the actual raw and frozen binding; no new git checkout audit is performed.'])
    result['raw_after_inspection'] = stream_receipt(RAW, RAW_BYTES, RAW_SHA)
    require(result['raw_after_inspection'] == raw_receipt, 'Raw changed during inspection')
    return result


if __name__ == '__main__':
    try:
        payload = json.dumps(main(), ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n'
        require(len(payload.encode('utf-8')) <= MAX_STDOUT_BYTES, 'Scalar metadata output exceeds bound')
        sys.stdout.write(payload)
    except BaseException as error:
        message = dict(schema='JOB3332_ACTUAL_RANK3_CALL0_READONLY_DIAGNOSTIC_V3',
            status='READONLY_DIAGNOSTIC_REFUSED_NO_ACCEPTANCE', job_id='3332', run=str(RUN),
            error_type=type(error).__name__, error=str(error)[:1000], probability_accepted=False,
            approval_emitted=False, selected_tolerances=None, remote_artifact_writes=0,
            full_vocab_vectors_emitted=False)
        payload = json.dumps(message, ensure_ascii=False, allow_nan=False) + '\n'
        if len(payload.encode('utf-8')) > MAX_STDOUT_BYTES:
            payload = '{"status":"READONLY_DIAGNOSTIC_REFUSED_OUTPUT_BOUND","probability_accepted":false}\n'
        sys.stdout.write(payload)
        sys.exit(2)
