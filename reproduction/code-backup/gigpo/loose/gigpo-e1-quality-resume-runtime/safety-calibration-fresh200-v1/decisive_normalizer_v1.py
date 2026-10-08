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
    return dict(selected_flat_indices=index.cpu(), target_ids=chosen.cpu(),
        pre_normalization_selected_logits_fp32=torch.cat(values) if values else torch.empty(0),
        fp32_logsumexp=torch.cat(lse) if lse else torch.empty(0),
        logits_input_dtype=str(logits.dtype), statistics_dtype='torch.float32',
        vocabulary_size=logits.shape[-1], bounded_full_vocab_vectors=vectors,
        full_vocab_storage_policy='H100_ONLY_NO_MAC_TRANSFER',
        added_model_forward_calls=0, rng_draws=0)


@contextmanager
def observe_hf_normalizer(actor, record, vector_budget, *, flat_function):
    """Intercept the actor's imported alias, not global torch or a new scorer."""
    original_flat = flat_function
    function = getattr(original_flat, '__func__', original_flat)
    globals_ = function.__globals__
    original = globals_['logprobs_from_logits']
    rows = record.setdefault('actual_normalizer_calls', [])

    def observed(logits, labels, *args, **kwargs):
        row = dict(normalizer=callable_identity(original), observer_errors=[])
        rows.append(row)
        try:
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
        except Exception as error:
            row['observer_errors'].append(repr(error))
        result = original(logits, labels, *args, **kwargs)
        try:
            index = row['selected_flat_indices'].to(result.device)
            row['actual_normalizer_selected_log_probs'] = result.detach().reshape(-1).index_select(0, index).cpu()
            row['actual_normalizer_output_dtype'] = str(result.dtype)
            row['normalizer_minus_fp32_formula'] = row['actual_normalizer_selected_log_probs'].float() - (
                row['pre_normalization_selected_logits_fp32'] - row['fp32_logsumexp'])
        except Exception as error:
            row['observer_errors'].append('normalizer_output: ' + repr(error))
        return result

    globals_['logprobs_from_logits'] = observed
    try:
        yield
    finally:
        globals_['logprobs_from_logits'] = original


@contextmanager
def observe_vllm_sampler(runner, model_input, record, vector_budget):
    """Observe actual sampler row and selected-target logit; return original samples."""
    sampler = runner.model.sampler
    original = sampler.forward
    existed = 'forward' in vars(sampler)
    module = inspect.getmodule(type(sampler))
    original_torch = module.torch
    original_compute = runner.model.compute_logits
    existed_compute = 'compute_logits' in vars(runner.model)
    records = record.setdefault('actual_sampler_calls', [])
    current = None

    class TorchProxy:
        def __getattr__(self, key):
            return getattr(original_torch, key)
        def log_softmax(self, logits, *args, **kwargs):
            result = original_torch.log_softmax(logits, *args, **kwargs)
            if current is not None:
                current['normalizer_input'] = logits.detach().clone()
                current['normalizer_output'] = result.detach().clone()
                current['normalizer_arguments'] = dict(args=args, kwargs=kwargs)
            return result

    def compute(*args, **kwargs):
        value = original_compute(*args, **kwargs)
        processor = runner.model.logits_processor
        record['actual_logits_processor'] = dict(class_path=type(processor).__module__ + '.' + type(processor).__name__,
            scale=processor.scale, soft_cap=processor.soft_cap,
            vocab_size=processor.vocab_size, output_dtype=str(value.dtype), output_shape=list(value.shape))
        return value

    def forward(logits, metadata, *args, **kwargs):
        nonlocal current
        row = dict(observer_errors=[], normalizer_path='vllm.model_executor.layers.sampler.torch.log_softmax',
                   original_normalizer='torch.log_softmax', input_dtype=str(logits.dtype))
        records.append(row)
        current = dict(pre_logits=logits.detach().clone())
        try:
            value = original(logits, metadata, *args, **kwargs)
            try:
                groups = metadata.seq_groups
                row['actual_processor_flags'] = {key: getattr(sampler, key) for key in
                    ('_do_penalties', '_do_top_p_top_k', '_do_min_p')}
                if any(row['actual_processor_flags'].values()):
                    raise ValueError('Actual non-neutral sampler processor branch executed')
                if len(groups) != len(value.outputs):
                    raise ValueError('Unknown actual vLLM sampler group/output schema')
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
                    selected_rows.append(group.sample_indices[0]); targets.append(sample.output_token)
                    mapping.append(dict(request_id=identity['request_id'], seq_id=sample.parent_seq_id,
                        selected_logit_row=group.sample_indices[0], target_id=sample.output_token,
                        completion_selected_log_prob=sample.logprobs[sample.output_token].logprob,
                        is_prompt=group.is_prompt, processor_branches_neutral=True))
                post, pre = current['normalizer_input'], current['pre_logits']
                # Each selected row has a target; create labels for original
                # vocabulary rows without assuming selected_token_indices order.
                labels = torch.zeros(post.shape[0], device=post.device, dtype=torch.long)
                labels[torch.as_tensor(selected_rows, device=post.device)] = torch.as_tensor(targets, device=post.device)
                row['pre_processor'] = selected_statistics(pre, labels, selected_rows, vector_budget=vector_budget)
                row['pre_normalization'] = selected_statistics(post, labels, selected_rows, vector_budget=vector_budget)
                row['selected_mapping'] = mapping
                if current['normalizer_arguments'] != dict(args=(), kwargs=dict(dim=-1, dtype=torch.float32)):
                    raise ValueError('Actual vLLM log_softmax path or dtype differs')
                row['actual_normalizer_output_dtype'] = str(current['normalizer_output'].dtype)
                row['actual_normalizer_arguments'] = dict(dim=-1, dtype='torch.float32')
                row['actual_normalizer_arguments_repr'] = repr(current['normalizer_arguments'])
                row['actual_normalizer_selected_log_probs'] = current['normalizer_output'][
                    torch.as_tensor(selected_rows, device=post.device), torch.as_tensor(targets, device=post.device)].cpu()
                row['processor_logits_unchanged'] = torch.equal(pre, post)
                row['actual_forward_calls_added'] = 0
            except Exception as error:
                row['observer_errors'].append(repr(error))
            return value
        finally:
            current = None

    sampler.forward = forward
    runner.model.compute_logits = compute
    module.torch = TorchProxy()
    try:
        yield
    finally:
        module.torch = original_torch
        if existed: sampler.forward = original
        else: del sampler.forward
        if existed_compute: runner.model.compute_logits = original_compute
        else: del runner.model.compute_logits


def collect_decisive(turns, raw_turns, old_actions):
    """Bind actual HF and sampler scalar evidence by source/action/request/target.

    Original PTs retain all-token scalars and bounded vocabulary vectors on
    H100. This JSON view contains only bounded largest residual positions.
    """
    from raw_hf_observation_v1 import require
    count = 0; top = []; normalizer_error = 0.; decomposition_error = 0.
    for serial, turn in enumerate(turns):
        chunk = len(raw_turns[(serial, 0)]['sampled_ids'])
        for index, observed in enumerate(turn):
            rank, local = divmod(index, chunk)
            raw = raw_turns[(serial, rank)]
            observation = raw['actual_runner_observation']
            request = observation['request_outputs'][local]
            request_id = request['request_id']
            action = old_actions[(observed['source'], observed['step_id'])]
            hf = action['normalizer_observation']
            hf_indices = [i for i, value in enumerate(hf['selected_mapping'])
                          if value['flat_row'] == action['normalizer_flat_row']]
            require(len(hf_indices) == len(action['tokens']), 'Incomplete complete-action HF normalization support')
            engine = {}
            for call in observation['calls']:
                for sample in call['actual_sampler_calls']:
                    require(not sample['observer_errors'], 'Unknown sampler normalization schema')
                    require(sample['processor_logits_unchanged'], 'Actual sampling processors changed logits')
                    processor = call['actual_logits_processor']
                    require(processor['scale'] == 1 and processor['soft_cap'] is None,
                            'Non-neutral actual model logits processor')
                    for i, mapping in enumerate(sample['selected_mapping']):
                        if mapping['request_id'] != request_id: continue
                        selected = mapping['selected_logit_row']
                        runner_index = int(call['selected_token_indices'][selected])
                        prediction_position = int(call['input_positions'][runner_index])
                        token_index = prediction_position - len(action['prefix']) + 1
                        require(token_index not in engine, 'Duplicate decisive request prediction position')
                        stats = sample['pre_normalization']
                        engine[token_index] = dict(target_id=mapping['target_id'], selected_logit_row=selected,
                            runner_input_index=runner_index, prediction_position=prediction_position,
                            selected_logit=float(stats['pre_normalization_selected_logits_fp32'][i]),
                            lse=float(stats['fp32_logsumexp'][i]),
                            logp=float(sample['actual_normalizer_selected_log_probs'][i]),
                            completion_logp=mapping['completion_selected_log_prob'],
                            normalizer=sample['normalizer_path'], dtype=sample['actual_normalizer_output_dtype'])
            for token_index, hf_index in enumerate(hf_indices):
                require(token_index in engine, 'Missing actual matched request/target normalization')
                e = engine[token_index]; target = action['tokens'][token_index]
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
    return dict(schema='MATCHED_ACTUAL_HF_VLLM_NORMALIZER_OBSERVATION_V1', valid_tokens=count,
        actual_normalizer_inputs_observed=True, actual_selected_target_ids_equal=True,
        added_model_forward_calls=0, rng_restores=0, thresholds_selected=False,
        max_normalizer_vs_fp32_formula_delta=normalizer_error,
        max_cross_model_decomposition_error=decomposition_error,
        maximum_actual_token_delta=abs(top[0]['actual_log_prob_delta']), largest_positions=top,
        full_vocab_vectors='BOUNDED_ORIGINAL_H100_PT_ONLY_NOT_DOWNLOADED_TO_MAC',
        vector_selection_limitation='bounded initial predictions; no claim all cross-model anomalous vectors captured')
