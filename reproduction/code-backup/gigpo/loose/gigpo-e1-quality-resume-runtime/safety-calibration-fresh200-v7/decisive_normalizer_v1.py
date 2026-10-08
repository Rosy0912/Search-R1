"""Passive selected-logit/LSE observations of existing normalizers.

No model forward, sampling call, RNG draw, state restore or normalizer replacement
is performed. Original callables are invoked exactly once and their original
outputs returned. Full vocabulary copies are bounded and remain in H100 PT files.
Unknown private schemas are recorded and rejected before Adam, never inferred.
"""
from contextlib import contextmanager
import hashlib
import inspect
from pathlib import Path
import torch


def callable_identity(fn):
    path = inspect.getsourcefile(fn)
    return dict(module=fn.__module__, name=fn.__qualname__, source_path=path,
        source_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else None)


def selected_statistics(logits, labels, selected, *, vector_budget):
    """Bound FP32 temporary allocation to eight vocabulary rows at a time."""
    flat = logits.detach().reshape(-1, logits.shape[-1])
    ids = labels.detach().reshape(-1)
    index = torch.as_tensor(selected, device=flat.device, dtype=torch.long)
    chosen = ids.index_select(0, index)
    values, lse = [], []
    vectors = []
    for start in range(0, len(index), 8):
        pos = index[start:start+8]
        rows = flat.index_select(0, pos).float()
        targets = chosen[start:start+8].long()
        values.append(rows.gather(-1, targets[:, None]).squeeze(-1).cpu())
        lse.append(torch.logsumexp(rows, dim=-1).cpu())
        # Six selected prediction vectors per observation context, never all
        # logits. These initial-position diagnostics are not claimed to cover
        # every future cross-model anomaly; scalars cover every active token.
        for local in range(len(pos)):
            if vector_budget[0] <= 0:
                break
            vectors.append(dict(selected_flat_index=int(pos[local]), target_id=int(targets[local]),
                                logits_fp32=rows[local].cpu().clone()))
            vector_budget[0] -= 1
        del rows, targets, pos  # Release this FP32 block before allocating the next eight rows.
    return dict(selected_flat_indices=index.cpu(), target_ids=chosen.cpu(),
        pre_normalization_selected_logits_fp32=torch.cat(values) if values else torch.empty(0),
        fp32_logsumexp=torch.cat(lse) if lse else torch.empty(0),
        logits_input_dtype=str(logits.dtype), statistics_dtype='torch.float32',
        vocabulary_size=logits.shape[-1], bounded_full_vocab_vectors=vectors,
        full_vocab_storage_policy='H100_ONLY_NO_MAC_TRANSFER',
        added_model_forward_calls=0, rng_draws=0)


@contextmanager
def observe_hf_normalizer(actor, record, vector_budget, *, flat_function):
    """An unavailable alias becomes missing evidence, never a skipped original call."""
    from observer_safety_v2 import attempt, note, patches
    try:
        function = getattr(flat_function, '__func__', flat_function)
        globals_ = function.__globals__
        original = globals_['logprobs_from_logits']
        rows = record.setdefault('actual_normalizer_calls', [])
    except BaseException as error:
        note(record, 'hf_normalizer_setup', error)
        yield
        return

    def observed(logits, labels, *args, **kwargs):
        row = dict(observer_errors=[])
        rows.append(row)
        def capture():
            row['normalizer'] = callable_identity(original)
            flat = record['flat_micro']
            prompt_width = flat['input_ids'].shape[-1] - flat['responses'].shape[-1]
            active = flat['attention_mask'][:, prompt_width:].bool()
            selected, mapping = [], []
            if actor.use_remove_padding:
                offset = 0
                for r in range(len(flat['responses'])):
                    prefix = int(flat['attention_mask'][r, :prompt_width].sum())
                    count = int(active[r].sum())
                    for t in range(count):
                        selected.append(offset + prefix - 1 + t)
                        mapping.append(dict(flat_row=r, action_token_index=t))
                    offset += int(flat['attention_mask'][r].sum())
                if logits.ndim != 2 or logits.shape[0] != offset:
                    raise ValueError('Unknown actual HF packed normalizer schema')
            else:
                if logits.ndim != 3 or tuple(labels.shape) != tuple(active.shape):
                    raise ValueError('Unknown actual HF dense normalizer schema')
                for r, t in active.nonzero().tolist():
                    selected.append(r * active.shape[1] + t)
                    mapping.append(dict(flat_row=r, action_token_index=t))
            stat = selected_statistics(logits, labels, selected, vector_budget=vector_budget)
            for i, position in enumerate(mapping):
                if int(stat['target_ids'][i]) != int(flat['responses'][position['flat_row'], position['action_token_index']]):
                    raise ValueError('Actual HF normalizer target crosses an action boundary')
            row.update(stat, selected_mapping=mapping)
            from verl.utils import torch_functional
            flash = bool(torch_functional.FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE)
            row['actual_normalizer_branch'] = 'flash_attn_cross_entropy' if flash else 'torch_log_softmax'
            row['actual_kernel'] = callable_identity(torch_functional.logprobs_from_logits_flash_attn if flash
                                                    else torch_functional.logprobs_from_logits_naive)
            row['processor_branch'] = dict(temperature=record['temperature'], additional_logits_processors=False)
        attempt(row, 'hf_normalizer_input', capture)
        result = original(logits, labels, *args, **kwargs)  # Original exceptions are not observation failures.
        def capture_output():
            index = row['selected_flat_indices'].to(result.device)
            row['actual_normalizer_selected_log_probs'] = result.detach().reshape(-1).index_select(0, index).cpu()
            row['actual_normalizer_output_dtype'] = str(result.dtype)
            row['normalizer_minus_fp32_formula'] = row['actual_normalizer_selected_log_probs'].float() - (
                row['pre_normalization_selected_logits_fp32'] - row['fp32_logsumexp'])
        attempt(row, 'hf_normalizer_output', capture_output)
        return result

    with patches(record) as install:
        install(globals_, 'logprobs_from_logits', observed, mapping=True)
        yield


def selected_rows_fingerprint(logits, rows):
    """Exact canonical-FP32 selected rows, permitting the real sampler dtype cast.

    Input/output dtypes are independently recorded; no numeric tolerance or
    processor modification is absorbed into this exact selected-row equality.
    Temporary copies are at most eight rows; no full logits matrix clone.
    """
    digest = hashlib.sha256()
    for start in range(0, len(rows), 8):
        index = torch.as_tensor(rows[start:start + 8], device=logits.device, dtype=torch.long)
        chunk = logits.detach().index_select(0, index).float().cpu().contiguous()
        digest.update(str((str(chunk.dtype), tuple(chunk.shape))).encode())
        digest.update(memoryview(chunk.reshape(-1).view(torch.uint8).numpy()).cast('B'))
        del chunk, index
    return digest.hexdigest()


@contextmanager
def observe_vllm_sampler(runner, model_input, record, vector_budget):
    """Passive borrowed references; bounded selected rows; never skip sampling."""
    from observer_safety_v2 import attempt, note, patches
    try:
        model = runner.model
        sampler = model.sampler
        original = sampler.forward
        module = inspect.getmodule(type(sampler))
        original_torch = module.torch
        original_compute = model.compute_logits
        records = record.setdefault('actual_sampler_calls', [])
    except BaseException as error:
        note(record, 'vllm_sampler_setup', error)
        yield
        return
    current = None

    class TorchProxy:
        def __getattr__(self, key):
            return getattr(original_torch, key)
        def log_softmax(self, logits, *args, **kwargs):
            result = original_torch.log_softmax(logits, *args, **kwargs)  # Exactly once.
            if current is not None:
                def capture():
                    current['normalizer_call_count'] += 1
                    # Borrow only until this original sampler returns. No matrix clone.
                    current['normalizer_input'] = logits.detach()
                    current['normalizer_output'] = result.detach()
                    current['normalizer_arguments'] = dict(args=args, kwargs=kwargs)
                    current['post_fingerprint'] = selected_rows_fingerprint(logits, current['selected_rows'])
                attempt(current['row'], 'vllm_log_softmax_capture', capture)
            return result

    def compute(*args, **kwargs):
        value = original_compute(*args, **kwargs)
        def capture():
            processor = model.logits_processor
            record['actual_logits_processor'] = dict(class_path=type(processor).__module__ + '.' + type(processor).__name__,
                scale=processor.scale, soft_cap=processor.soft_cap,
                vocab_size=processor.vocab_size, output_dtype=str(value.dtype), output_shape=list(value.shape))
        attempt(record, 'vllm_logits_processor_metadata', capture)
        return value

    def forward(logits, metadata, *args, **kwargs):
        nonlocal current
        row = dict(observer_errors=[], normalizer_path='vllm.model_executor.layers.sampler.torch.log_softmax',
                   original_normalizer='torch.log_softmax')
        records.append(row)
        current = dict(row=row, normalizer_call_count=0)
        def capture_pre():
            row['input_dtype'] = str(logits.dtype)
            selected = [int(index) for group in metadata.seq_groups for index in group.sample_indices]
            current.update(selected_rows=selected, pre_fingerprint=selected_rows_fingerprint(logits, selected))
        attempt(row, 'vllm_pre_sampler_capture', capture_pre)
        try:
            value = original(logits, metadata, *args, **kwargs)  # Never enclosed in an observation catch.
            def capture_post():
                groups = metadata.seq_groups
                row['actual_processor_flags'] = {key: getattr(sampler, key) for key in
                    ('_do_penalties', '_do_top_p_top_k', '_do_min_p')}
                if any(row['actual_processor_flags'].values()):
                    raise ValueError('Actual non-neutral sampler processor branch executed')
                if len(groups) != len(value.outputs):
                    raise ValueError('Unknown actual vLLM sampler group/output schema')
                row['actual_normalizer_call_count'] = current['normalizer_call_count']
                if current['normalizer_call_count'] != 1:
                    raise ValueError('Actual sampler must contain exactly one observed normalizer call')
                seq_identity = {g['seq_id']: g for g in record['groups']}
                selected_rows, targets, mapping = [], [], []
                for group, output in zip(groups, value.outputs):
                    p = group.sampling_params
                    neutral = (p.temperature == 1 and p.top_p == 1 and p.top_k == -1 and p.n == 1
                        and p.min_p == 0 and p.min_tokens == 0 and p.presence_penalty == 0
                        and p.frequency_penalty == 0 and p.repetition_penalty == 1 and not p.logits_processors)
                    if not neutral or len(group.seq_ids) != 1 or len(output.samples) != 1 or len(group.sample_indices) != 1:
                        raise ValueError('Non-neutral/unknown actual sampling processor branch')
                    sample = output.samples[0]
                    if sample.parent_seq_id != group.seq_ids[0]:
                        raise ValueError('Selected sampler parent differs from prepared request')
                    identity = seq_identity[sample.parent_seq_id]
                    selected_rows.append(int(group.sample_indices[0])); targets.append(int(sample.output_token))
                    mapping.append(dict(request_id=identity['request_id'], seq_id=sample.parent_seq_id,
                        selected_logit_row=int(group.sample_indices[0]), target_id=int(sample.output_token),
                        sequence_output_selected_log_prob=sample.logprobs[sample.output_token].logprob,
                        is_prompt=group.is_prompt, processor_branches_neutral=True))
                post = current['normalizer_input']
                if (selected_rows != current['selected_rows'] or current['pre_fingerprint'] != current['post_fingerprint']
                        or selected_rows_fingerprint(post, selected_rows) != current['post_fingerprint']):
                    raise ValueError('Actual sampling processors changed selected vocabulary logits')
                labels = torch.zeros(post.shape[0], device=post.device, dtype=torch.long)
                labels[torch.as_tensor(selected_rows, device=post.device)] = torch.as_tensor(targets, device=post.device)
                row['pre_normalization'] = selected_statistics(post, labels, selected_rows, vector_budget=vector_budget)
                # Equality fingerprints establish neutral processing; no borrowed-pre reference claim.
                row['pre_processor_selected_rows_sha256'] = current['pre_fingerprint']
                row['post_processor_selected_rows_sha256'] = current['post_fingerprint']
                row['selected_mapping'] = mapping
                if current['normalizer_arguments'] != dict(args=(), kwargs=dict(dim=-1, dtype=torch.float32)):
                    raise ValueError('Actual vLLM log_softmax path or dtype differs')
                normalized = current['normalizer_output']
                row['actual_normalizer_call_count'] = current['normalizer_call_count']
                row['actual_normalizer_output_dtype'] = str(normalized.dtype)
                row['actual_normalizer_arguments'] = dict(dim=-1, dtype='torch.float32')
                row['actual_normalizer_selected_log_probs'] = normalized[
                    torch.as_tensor(selected_rows, device=post.device), torch.as_tensor(targets, device=post.device)].cpu()
                row['processor_logits_unchanged'] = True
                row['actual_forward_calls_added'] = 0
            attempt(row, 'vllm_post_sampler_capture', capture_post)
            return value
        finally:
            current = None

    with patches(record) as install:
        install(sampler, 'forward', forward)
        install(model, 'compute_logits', compute)
        install(module, 'torch', TorchProxy())
        yield


def _index_engine_observation(raw):
    """Build once per serial/rank, rather than scanning every decode for each action."""
    from raw_hf_observation_v1 import require
    observation = raw['actual_runner_observation']
    requests = {}
    for local, request in enumerate(observation['request_outputs']):
        request_id = request['request_id']
        require(request_id not in requests, 'Duplicate actual completion request')
        require(len(request['completions']) == 1, 'Actual completion count differs')
        requests[request_id] = dict(local=local, prefix=request['prompt_token_ids'],
                                   completion=request['completions'][0], tokens={})
    for call in observation['calls']:
        require(not call['observer_errors'], 'Unknown actual runner/restore schema')
        processor = call['actual_logits_processor']
        require(processor['scale'] == 1 and processor['soft_cap'] is None,
                'Non-neutral actual model logits processor')
        for sample in call['actual_sampler_calls']:
            require(not sample['observer_errors'] and sample['actual_normalizer_call_count'] == 1,
                    'Missing/unknown/multiple actual sampler normalizer calls')
            require(sample['processor_logits_unchanged'], 'Actual sampling processors changed logits')
            for i, mapping in enumerate(sample['selected_mapping']):
                request = requests[mapping['request_id']]
                selected = mapping['selected_logit_row']
                runner_index = int(call['selected_token_indices'][selected])
                prediction_position = int(call['input_positions'][runner_index])
                token_index = prediction_position - len(request['prefix']) + 1
                require(token_index not in request['tokens'], 'Duplicate decisive request prediction position')
                require(0 <= token_index < len(request['completion']['token_ids']), 'Selected token exceeds actual CompletionOutput')
                require(mapping['target_id'] == request['completion']['token_ids'][token_index]
                        == int(raw['sampled_ids'][request['local'], token_index]),
                        'SequenceOutput/CompletionOutput/raw target differs')
                normalizer = sample['actual_normalizer_selected_log_probs'][i]
                sequence = mapping['sequence_output_selected_log_prob']
                completion = request['completion']['chosen_token_log_probs'][token_index]
                sampled = raw['sampled_log_probs'][request['local'], token_index]
                # Exact actual scalar equality before the engine tensor conversion,
                # then exact equality after its actual dtype conversion. No tolerance.
                require(float(normalizer) == sequence == completion,
                        'SequenceOutput/CompletionOutput/normalizer chosen probability differs')
                converted = torch.as_tensor(sequence, dtype=raw['sampled_log_probs'].dtype)
                require(torch.equal(converted, sampled.cpu()), 'Raw sampled probability conversion differs')
                stats = sample['pre_normalization']
                request['tokens'][token_index] = dict(target_id=mapping['target_id'], seq_id=mapping['seq_id'],
                    selected_logit_row=selected, runner_input_index=runner_index, prediction_position=prediction_position,
                    selected_logit=float(stats['pre_normalization_selected_logits_fp32'][i]),
                    lse=float(stats['fp32_logsumexp'][i]), logp=float(normalizer),
                    sequence_output_logp=sequence, completion_output_logp=completion, raw_sampled_logp=float(sampled),
                    raw_sampled_dtype=str(raw['sampled_log_probs'].dtype),
                    selected_probability_chain_exact=True, normalizer=sample['normalizer_path'],
                    dtype=sample['actual_normalizer_output_dtype'])
    for request in requests.values():
        require(set(request['tokens']) == set(range(len(request['completion']['token_ids']))),
                'Incomplete exact chosen-probability chain coverage')
    return requests


def collect_decisive(turns, raw_turns, old_actions):
    """Decompose the same exact probability admitted by the raw sampling chain.

    FP32 formula residuals are diagnostics, never proof of HF/vLLM parity or a
    replacement for the separate independently chosen actual probability gate.
    """
    from raw_hf_observation_v1 import require
    count = 0; top = []; normalizer_error = 0.; decomposition_error = 0.
    indexes, hf_indexes = {}, {}
    for serial, turn in enumerate(turns):
        chunk = len(raw_turns[(serial, 0)]['sampled_ids'])
        for rank in range(8):
            indexes[(serial, rank)] = _index_engine_observation(raw_turns[(serial, rank)])
        for index, observed in enumerate(turn):
            rank, local = divmod(index, chunk)
            raw = raw_turns[(serial, rank)]
            request_id = raw['actual_runner_observation']['request_outputs'][local]['request_id']
            request = indexes[(serial, rank)][request_id]
            action = old_actions[(observed['source'], observed['step_id'])]
            hf = action['normalizer_observation']
            key = id(hf)
            if key not in hf_indexes:
                mapping = {}
                for i, position in enumerate(hf['selected_mapping']):
                    mapping.setdefault(position['flat_row'], []).append(i)
                hf_indexes[key] = mapping
            hf_indices = hf_indexes[key][action['normalizer_flat_row']]
            require(len(hf_indices) == len(action['tokens']), 'Incomplete complete-action HF normalization support')
            require(request['prefix'] == action['prefix'], 'Decisive raw request/action prefix differs')
            for token_index, hf_index in enumerate(hf_indices):
                require(token_index in request['tokens'], 'Missing actual request/target normalization')
                e = request['tokens'][token_index]; target = action['tokens'][token_index]
                require(e['target_id'] == int(hf['target_ids'][hf_index]) == target,
                        'Normalizer selected target identity differs')
                h = dict(selected_logit=float(hf['pre_normalization_selected_logits_fp32'][hf_index]),
                         lse=float(hf['fp32_logsumexp'][hf_index]),
                         logp=float(hf['actual_normalizer_selected_log_probs'][hf_index]))
                require(h['logp'] == float(action['log_probs'][token_index]), 'HF actual normalizer differs from actual scorer')
                actual = h['logp'] - e['logp']; formula = (h['selected_logit'] - e['selected_logit']) - (h['lse'] - e['lse'])
                normalizer_error = max(normalizer_error, abs(h['logp']-(h['selected_logit']-h['lse'])),
                                       abs(e['logp']-(e['selected_logit']-e['lse'])))
                decomposition_error = max(decomposition_error, abs(actual-formula))
                item = dict(source=observed['source'], step_id=observed['step_id'], token_index=token_index,
                    rank=rank, request_id=request_id, target_id=target, hf=h, vllm=e,
                    actual_log_prob_delta=actual, selected_logit_delta=h['selected_logit']-e['selected_logit'],
                    fp32_lse_delta=h['lse']-e['lse'], formula_delta=formula,
                    hf_normalizer=hf['normalizer'], hf_branch=hf['actual_normalizer_branch'],
                    hf_input_dtype=hf['logits_input_dtype'], hf_output_dtype=hf['actual_normalizer_output_dtype'])
                top.append(item); top.sort(key=lambda value: abs(value['actual_log_prob_delta']), reverse=True)
                del top[20:]; count += 1
    require(count > 0, 'No decisive actual probabilities observed')
    return dict(schema='MATCHED_ACTUAL_HF_VLLM_NORMALIZER_OBSERVATION_V2', valid_tokens=count,
        actual_normalizer_inputs_observed=True, actual_selected_target_ids_equal=True,
        all_effective_selected_probability_chains_exact=True,
        chain='SequenceOutput->CompletionOutput->raw_sampled_tensor->actual_normalizer_selected_logp',
        normalizer_calls_per_sampler=1, added_model_forward_calls=0, rng_restores=0,
        thresholds_selected=False, probability_parity_accepted=False,
        decomposition_scope='ACTUAL_ADMITTED_CHAIN_WITH_FP32_RESIDUALS_NOT_PARITY_PROOF',
        max_normalizer_vs_fp32_formula_delta=normalizer_error,
        max_cross_model_decomposition_error=decomposition_error,
        maximum_actual_token_delta=abs(top[0]['actual_log_prob_delta']), largest_positions=top,
        collector_index_policy='ONCE_PER_SERIAL_RANK_REQUEST_AND_HF_CALL',
        full_vocab_vectors='BOUNDED_ORIGINAL_H100_PT_ONLY_NOT_DOWNLOADED_TO_MAC',
        vector_selection_limitation='bounded initial predictions; no claim all cross-model anomalous vectors captured')
