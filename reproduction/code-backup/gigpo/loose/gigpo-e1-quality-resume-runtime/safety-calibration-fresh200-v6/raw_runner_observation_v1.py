"""Read existing vLLM prepare/execute/model inputs, without an extra forward.

This uses the exact installed 0.6.3 ModelRunner interface already preserved as
17685c52... .py. Actual GPU/private-interface acceptance remains outstanding.
Nothing here infers observed runner positions from an environment request.
"""
from contextlib import contextmanager
import inspect
import torch
from raw_hf_observation_v1 import clone_cpu, require


@contextmanager
def observe_actual_runner(engine):
    """Observer lookup/install/copy/restore failures return missing evidence only."""
    from observer_safety_v2 import attempt, note, patches, install_pre_hook
    from contextlib import nullcontext
    output = dict(schema='ACTUAL_VLLM_RUNNER_INPUT_OBSERVATION_V1', calls=[], request_outputs=[], observer_errors=[],
                  actual_forward_calls_added=0, rng_restores=0)
    try:
        runner = engine.llm_engine.model_executor.worker.model_runner
        original_prepare, original_execute = runner.prepare_model_input, runner.execute_model
        original_postprocess = engine._post_process_outputs
    except BaseException as error:
        note(output, 'runner_setup', error)
        yield output
        return
    prepared, vector_budget = {}, [6]

    def prepare(*args, **kwargs):
        value = original_prepare(*args, **kwargs)
        record = dict(groups=[], observer_errors=[])
        def capture():
            groups = inspect.signature(original_prepare).bind(*args, **kwargs).arguments['seq_group_metadata_list']
            index = 0
            for group in groups:
                for seq_id, data in group.seq_data.items():
                    record['groups'].append(dict(request_id=group.request_id, seq_id=seq_id,
                        is_prompt=group.is_prompt, sequence_length=data.get_len(), prompt_length=data.get_prompt_len(),
                        num_computed_tokens=data.get_num_computed_tokens(), query_length=value.query_lens[index],
                        mrope_position_delta=data.mrope_position_delta))
                    index += 1
            require(index == len(value.query_lens), 'Prepared sequence/query metadata coverage differs')
            prepared[id(value)] = record
        attempt(record, 'prepare_metadata', capture)
        if record['observer_errors']:
            output['observer_errors'].extend(record['observer_errors'])
        return value

    def execute(*args, **kwargs):
        record = dict(actual_model_calls=[], observer_errors=[])
        output['calls'].append(record)
        model_input = None
        def capture():
            nonlocal model_input
            model_input = inspect.signature(original_execute).bind(*args, **kwargs).arguments['model_input']
            prepare_record = prepared.pop(id(model_input), None)
            require(prepare_record is not None, 'Actual execute input lacks corresponding observed preparation')
            record.update(prepare_record)
            record.update(input_tokens=clone_cpu(model_input.input_tokens),
                input_positions=clone_cpu(model_input.input_positions), query_lens=list(model_input.query_lens),
                seq_lens=list(model_input.seq_lens), slot_mapping=clone_cpu(model_input.attn_metadata.slot_mapping),
                selected_token_indices=clone_cpu(model_input.sampling_metadata.selected_token_indices),
                num_prefills=model_input.attn_metadata.num_prefills,
                num_prefill_tokens=model_input.attn_metadata.num_prefill_tokens,
                num_decode_tokens=model_input.attn_metadata.num_decode_tokens)
            decode = model_input.attn_metadata.decode_metadata
            record['decode_uses_cuda_graph'] = bool(decode.use_cuda_graph) if decode is not None else False
            record['actual_cache_metadata'] = {key: clone_cpu(getattr(model_input.attn_metadata, key))
                for key in ('block_tables', 'seq_lens_tensor', 'query_start_loc', 'seq_start_loc')}
            record['actual_decode_cache_metadata'] = {key: clone_cpu(getattr(decode, key))
                for key in ('block_tables', 'seq_lens_tensor', 'slot_mapping')} if decode is not None else None
            record['cache_scope'] = 'ACTUAL_FLASH_METADATA_FIELDS_NOT_DIRECT_KERNEL_KV_CONTENTS'
        attempt(record, 'execute_metadata', capture)

        def before_model(module, call_args, call_kwargs):
            attempt(record, 'actual_vllm_model_input', lambda: record['actual_model_calls'].append(
                dict(args=clone_cpu(call_args), input_ids=clone_cpu(call_kwargs['input_ids']),
                     positions=clone_cpu(call_kwargs['positions']),
                     slot_mapping=clone_cpu(call_kwargs['attn_metadata'].slot_mapping))))
            return None
        hook_cleanup = None
        def hook_setup():
            nonlocal hook_cleanup
            hook_cleanup = install_pre_hook(record, runner.model, before_model, 'vllm_model_hook')
        attempt(record, 'vllm_model_hook_lookup', hook_setup)
        context = nullcontext()
        try:
            from decisive_normalizer_v1 import observe_vllm_sampler
            context = observe_vllm_sampler(runner, model_input, record, vector_budget)
        except BaseException as error:
            note(record, 'sampler_observer_import', error)
        try:
            with context:
                value = original_execute(*args, **kwargs)  # Exactly once, including failed metadata binding.
        finally:
            if hook_cleanup is not None:
                attempt(record, 'vllm_model_hook_restore', hook_cleanup)
        return value

    def postprocess(request_outputs):
        def capture():
            output['request_outputs'] = [dict(request_id=request.request_id,
                prompt_token_ids=list(request.prompt_token_ids),
                completions=[dict(index=completion.index, token_ids=list(completion.token_ids),
                    chosen_token_log_probs=[values[token].logprob
                        for token, values in zip(completion.token_ids, completion.logprobs)])
                             for completion in request.outputs]) for request in request_outputs]
        attempt(output, 'request_output_mapping', capture)
        return original_postprocess(request_outputs)

    with patches(output) as install:
        install(runner, 'prepare_model_input', prepare)
        install(runner, 'execute_model', execute)
        install(engine, '_post_process_outputs', postprocess)
        yield output


def validate_actual_runner_observation(observation, engine_prompts, sampled_ids, sampled_lengths, sampled_log_probs):
    """Strict observed engine tokens/positions and sampled-target request mapping."""
    require(observation['schema'] == 'ACTUAL_VLLM_RUNNER_INPUT_OBSERVATION_V1'
            and not observation['observer_errors'] and observation['actual_forward_calls_added'] == 0
            and observation['rng_restores'] == 0, 'Incomplete/non-neutral actual runner observer')
    requests = observation['request_outputs']
    require(len(requests) == len(engine_prompts) == len(sampled_ids), 'Actual runner request-output coverage differs')
    by_request, observed_inputs, observed_predictions = {}, {}, {}
    for row, request in enumerate(requests):
        request_id = request['request_id']
        require(request_id not in by_request and request['prompt_token_ids'] == engine_prompts[row],
                'Actual engine request identity/prefix differs')
        require(len(request['completions']) == 1 and request['completions'][0]['index'] == 0,
                'Actual n=1 request output differs')
        sampled = sampled_ids[row, :int(sampled_lengths[row])].tolist()
        require(request['completions'][0]['token_ids'] == sampled, 'Actual CompletionOutput tokens differ')
        chosen = torch.tensor(request['completions'][0]['chosen_token_log_probs'], dtype=sampled_log_probs.dtype)
        require(len(chosen) == len(sampled) and torch.equal(chosen, sampled_log_probs[row, :len(sampled)]),
                'Actual CompletionOutput chosen-ID log-probability differs from engine return')
        by_request[request_id] = (row, list(engine_prompts[row]), sampled)
        observed_inputs[request_id], observed_predictions[request_id] = set(), set()
    require(observation['calls'], 'Missing actual vLLM runner calls')
    for serial, call in enumerate(observation['calls']):
        require(not call['observer_errors'] and call['decode_uses_cuda_graph'] is False,
                'Unknown runner inputs or unobserved CUDA-graph path')
        require(call.get('actual_sampler_calls') and all(not row['observer_errors'] and row.get('actual_normalizer_call_count') == 1
                    for row in call['actual_sampler_calls']),
                'Missing/unknown actual sampler normalization observation')
        require(call.get('actual_cache_metadata') is not None, 'Missing actual cache metadata')
        require(len(call['actual_model_calls']) == 1, 'Actual vLLM model forward input was not observed')
        model = call['actual_model_calls'][0]
        require(model['args'] == () and torch.equal(model['input_ids'], call['input_tokens'])
                and torch.equal(model['positions'], call['input_positions'])
                and torch.equal(model['slot_mapping'], call['slot_mapping']), 'Actual runner/model tokens, positions or cache slots differ')
        require(call['input_tokens'].ndim == call['input_positions'].ndim == 1
                and len(call['input_tokens']) == len(call['input_positions']), 'Unsupported actual runner position tensor')
        require(len(call['groups']) == len(call['query_lens']) == len(call['seq_lens']), 'Actual packed request mapping differs')
        selected = set(call['selected_token_indices'].tolist())
        cursor = 0
        for group, query_length, sequence_length in zip(call['groups'], call['query_lens'], call['seq_lens']):
            require(group['request_id'] in by_request and group['query_length'] == query_length
                    and group['mrope_position_delta'] is None, 'Unknown/multiaxis runner request identity')
            row, prompt, sampled = by_request[group['request_id']]
            prefix_length = len(prompt)
            require(group['prompt_length'] == prefix_length and query_length > 0,
                    'Actual runner prompt/query length differs')
            positions = call['input_positions'][cursor:cursor+query_length].tolist()
            ids = call['input_tokens'][cursor:cursor+query_length].tolist()
            require(positions == list(range(sequence_length-query_length, sequence_length)),
                    'Actual runner query positions are not the bound natural sequence')
            context = prompt + sampled
            require(ids == [context[position] for position in positions], 'Actual runner token path differs from sampled context')
            for local, position in enumerate(positions):
                observed_inputs[group['request_id']].add(position)
                if cursor+local in selected:
                    target_index = position - (prefix_length-1)
                    require(0 <= target_index < len(sampled), 'Actual sampled prediction crosses a request or support boundary')
                    require(target_index not in observed_predictions[group['request_id']], 'Duplicate actual sampled prediction position')
                    observed_predictions[group['request_id']].add(target_index)
            cursor += query_length
        require(cursor == len(call['input_tokens']), 'Unreviewed padded/missing actual runner tokens')
        require(all(0 <= index < cursor for index in selected), 'Actual sampler selected logit position exceeds runner input')
    result = []
    for request_id, (row, prompt, sampled) in by_request.items():
        require(set(range(len(sampled))) == observed_predictions[request_id], 'Incomplete actual sampled prediction coverage')
        require(set(range(len(prompt))) <= observed_inputs[request_id], 'Actual prefill prefix input was not fully observed')
        result.append(dict(row=row, request_id=request_id, actual_vllm_runner_positions_observed=True,
            actual_vllm_model_tokens_observed=True, sampled_prediction_positions=list(range(len(prompt)-1,len(prompt)+len(sampled)-1)),
            actual_vllm_slot_mapping_observed=True,
            actual_prefill_and_decode_coverage=True, added_forward_calls=0,
            internal_cu_seqlens_tensor_observed=False, probability_accepted=False))
    return result
