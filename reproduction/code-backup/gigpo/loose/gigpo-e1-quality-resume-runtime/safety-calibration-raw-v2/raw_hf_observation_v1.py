"""Read-only observations of existing HF calls, not a replacement scorer.

The actual module pre-hook observes kwargs AFTER the actor's remove-padding
transformation. The surrounding flat-call wrapper observes the input that
prepare_action_prefix_micro constructed. Neither hook changes an argument,
output, model, RNG, optimizer or sampling distribution. Observer errors survive
in the record and are rejected by the driver before its first Adam RPC.
"""
from contextlib import contextmanager
import copy
import torch


def require(condition, message):
    if not condition:
        raise ValueError(message)


def clone_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: clone_cpu(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(clone_cpu(item) for item in value)
    return copy.deepcopy(value)


@contextmanager
def capture_existing_flat_calls(actor):
    """Return original outputs unchanged; all recorded tensors are copies."""
    calls = []
    original = actor._forward_flat_micro_batch
    had_instance_override = '_forward_flat_micro_batch' in vars(actor)

    def observed(micro, temperature):
        record = dict(flat_micro=None, temperature=temperature, model_calls=[], observer_errors=[],
                      use_remove_padding=actor.use_remove_padding,
                      ulysses_sequence_parallel_size=actor.ulysses_sequence_parallel_size)
        calls.append(record)
        try:
            record['flat_micro'] = clone_cpu(dict(micro.items()))
        except Exception as error:
            record['observer_errors'].append('flat_input_capture: ' + repr(error))

        def before_model(module, args, kwargs):
            try:
                record['model_calls'].append(dict(args=clone_cpu(args), kwargs=clone_cpu(kwargs)))
            except Exception as error:
                record['observer_errors'].append('actual_model_input_capture: ' + repr(error))
            return None  # PyTorch preserves original kwargs and args.

        hook = actor.actor_module.register_forward_pre_hook(before_model, with_kwargs=True)
        try:
            result = original(micro, temperature)
        finally:
            hook.remove()
        try:
            record['flat_entropy'], record['flat_log_probs'] = clone_cpu(result)
        except Exception as error:
            record['observer_errors'].append('flat_output_capture: ' + repr(error))
        return result

    actor._forward_flat_micro_batch = observed
    try:
        yield calls
    finally:
        if had_instance_override:
            actor._forward_flat_micro_batch = original
        else:
            del actor._forward_flat_micro_batch


def validate_actual_model_inputs(record):
    """Verify the actual one HF module call, including unpadded positions."""
    require(not record['observer_errors'], 'HF observation incomplete: ' + repr(record['observer_errors']))
    require(record['temperature'] == 1. and record['ulysses_sequence_parallel_size'] == 1,
            'Unexpected HF distribution or sequence-parallel input path')
    require(len(record['model_calls']) == 1, 'Missing/multiple actual HF module calls')
    actual = record['model_calls'][0]
    require(actual['args'] == (), 'Actual HF module used unbound positional arguments')
    kwargs, flat = actual['kwargs'], record['flat_micro']
    require(kwargs.get('use_cache') is False, 'Actual HF cache behavior differs')
    if record['use_remove_padding']:
        selected = flat['attention_mask'].bool()
        expected_ids = flat['input_ids'][selected].reshape(1, -1)
        expected_positions = flat['position_ids'][selected].reshape(1, -1)
        require(kwargs.get('attention_mask') is None, 'Unpadded HF mask must be absent')
    else:
        expected_ids, expected_positions = flat['input_ids'], flat['position_ids']
        require(torch.equal(kwargs['attention_mask'], flat['attention_mask']), 'Actual HF attention mask changed')
    require(torch.equal(kwargs['input_ids'], expected_ids), 'Actual HF model tokens differ from prepared prefixes/actions')
    require(torch.equal(kwargs['position_ids'], expected_positions), 'Actual HF model positions differ from prepared prefixes/actions')
    shape = flat['responses'].shape
    require(tuple(record['flat_log_probs'].shape) == tuple(shape), 'HF flat output mapping differs')
    # Prediction labels for an action are read at the preceding token. Verify
    # their exact packed-row boundaries without claiming cu_seqlens observation.
    prompt_width = flat['input_ids'].shape[1] - flat['responses'].shape[1]
    row_lengths = flat['attention_mask'].sum(-1).tolist()
    packed_start = 0
    boundaries = []
    for row, row_length in enumerate(row_lengths):
        prefix_length = int(flat['attention_mask'][row, :prompt_width].sum())
        action_length = int(flat['attention_mask'][row, prompt_width:].sum())
        action = flat['responses'][row, :action_length]
        if record['use_remove_padding'] and action_length:
            indices = torch.arange(packed_start + prefix_length - 1,
                                   packed_start + prefix_length + action_length - 1)
            labels = kwargs['input_ids'].reshape(-1).roll(-1).index_select(0, indices)
            require(torch.equal(labels, action), 'Actual packed prediction labels cross a row or lose an action token')
        boundaries.append(dict(flat_row=row, packed_start=packed_start, packed_end=packed_start+int(row_length),
            prefix_length=prefix_length, action_length=action_length,
            target_prediction_packed_positions=list(range(packed_start+prefix_length-1,
                packed_start+prefix_length+action_length-1)) if action_length else [],
            boundary_scope='DERIVED_FROM_OBSERVED_HF_INPUT_AND_ACTUAL_PREPARED_MASK',
            internal_cu_seqlens_tensor_observed=False))
        packed_start += int(row_length)
    return dict(actual_hf_model_input_observed=True, actual_hf_model_positions_observed=True,
                actual_hf_model_attention_observed=True,
                actual_vllm_runner_positions_observed=False,
                sampler_position_evidence_scope='DERIVED_BACKEND_CONTRACT_EXPECTATION', packed_row_boundaries=boundaries)


def unpack_complete_action_calls(raw, *, prepare, pad_id, eos_id):
    """Map captured HF calls back to (source, step), never a raw timeline prefix."""
    require(raw['schema'] == 'RAW_ACTION_HF_FORWARD_OBSERVATION_V1', 'Wrong HF observation schema')
    require(raw['role'] in ('old', 'current', 'reference'), 'Unknown observed HF role')
    require(raw['actor_token_identity'] == 'sampled_action_prefix_v1', 'Canonical scorer cannot claim raw probability')
    require(raw['actual_pad_token_id'] == pad_id and raw['actual_eos_token_id'] == eos_id
            and raw['temperature'] == 1., 'Observed HF tokenizer/distribution identity differs')
    require(not raw['observer_errors'], 'Incomplete HF observer')
    data, records, calls = raw['input_tensors'], raw['action_prefix_records'], raw['flat_calls']
    sources = raw['trajectory_sources']
    questions = list(raw['trajectory_question_ids'])
    require(len(sources) == len(records) == len(questions) == len(data['responses']), 'HF source/provenance rows differ')
    require(len(set(sources)) == len(sources), 'Duplicate HF trajectory sources')
    require(all(type(source) is int and source >= 0 for source in sources), 'Invalid HF trajectory source')
    require(raw['micro_batch_size'] > 0 and len(sources) % raw['micro_batch_size'] == 0,
            'Unreviewed partial/dynamic HF micro-batch path')
    require(len(calls) == len(sources) // raw['micro_batch_size'], 'Incomplete existing HF micro forwards')
    actions = {}
    for serial, call in enumerate(calls):
        start = serial * raw['micro_batch_size']
        end = start + raw['micro_batch_size']
        micro = {key: tensor[start:end] for key, tensor in data.items()}
        prepared = prepare(micro, pad_id, eos_id)
        require(call['flat_micro'].keys() == prepared.flat_micro.keys(), 'Actual prepared HF keys differ')
        for key, expected in prepared.flat_micro.items():
            require(torch.equal(call['flat_micro'][key], expected.cpu()), 'Actual prepared HF input changed: ' + key)
        evidence = validate_actual_model_inputs(call)
        for local, source in enumerate(sources[start:end]):
            provenance = records[start + local]
            count = int(data['actor_action_count'][start + local])
            require(len(provenance) == count, 'HF action provenance count differs')
            for slot, item in enumerate(provenance):
                require(item['step_id'] == slot, 'HF action slot changed')
                row = local * 4 + slot
                prefix_width = prepared.flat_micro['prompts'].shape[1]
                plen = int(data['actor_action_prefix_lengths'][start + local, slot])
                length = int(data['actor_action_lengths'][start + local, slot])
                flat = call['flat_micro']
                prefix = flat['prompts'][row, -plen:].tolist()
                tokens = flat['responses'][row, :length].tolist()
                offset = int(data['actor_action_output_offsets'][start + local, slot])
                require(prefix == item['generation_prefix'] and tokens == item['actor_action_tokens']
                        and offset == item['actor_output_offset'], 'HF input/provenance mapping differs')
                require(flat['attention_mask'][row, -flat['responses'].shape[1]:][:length].tolist() == [1] * length,
                        'Raw sampled PAD/special token lost its effective HF mask')
                positions = flat['position_ids'][row, prefix_width:prefix_width + length].tolist()
                require(positions == list(range(plen, plen + length)), 'HF action positions differ')
                probs = call['flat_log_probs'][row, :length].double()
                scattered = raw['output_log_probs'][start + local, offset:offset + length].double()
                require(torch.equal(probs, scattered), 'Observed flat HF probabilities differ from scattered output')
                require((source, slot) not in actions, 'Duplicate HF action mapping')
                actions[(source, slot)] = dict(source=source, step_id=slot, prefix=prefix, tokens=tokens,
                    positions=positions, mask=[1] * length, offset=offset, log_probs=probs,
                    question_id=questions[start+local], response_group=source//5, response_slot=source%5,
                    role=raw['role'], rank=raw['rank'], pid=raw['pid'], observation=evidence)
    return actions
