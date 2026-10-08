"""UNEXECUTED scalar chain fixtures; no parity, model, sampling or GPU evidence."""
import copy
from types import SimpleNamespace
import pytest
import torch
from decisive_normalizer_v1 import _index_engine_observation
from raw_runner_observation_v1 import build_packed_runner_input_mapping, validate_actual_runner_observation


def fixture():
    sample = dict(observer_errors=[], actual_normalizer_call_count=1, processor_logits_unchanged=True,
        selected_mapping=[dict(request_id='r', seq_id=0, selected_logit_row=0, target_id=7,
                               sequence_output_selected_log_prob=-.5, is_prompt=True)],
        actual_normalizer_selected_log_probs=torch.tensor([-.5]),
        pre_normalization=dict(pre_normalization_selected_logits_fp32=torch.tensor([1.]),
                               fp32_logsumexp=torch.tensor([1.5])),
        normalizer_path='actual observed callable', actual_normalizer_output_dtype='torch.float32')
    call = dict(observer_errors=[], actual_logits_processor=dict(scale=1, soft_cap=None),
        selected_token_indices=torch.tensor([1]), input_tokens=torch.tensor([1, 2]),
        input_positions=torch.tensor([0, 1]), query_lens=[2], seq_lens=[2],
        groups=[dict(request_id='r', seq_id=0, sequence_length=2, query_length=2,
                     prompt_length=2, is_prompt=True, mrope_position_delta=None)],
        actual_sampler_calls=[sample])
    call['packed_input_mapping'] = build_packed_runner_input_mapping(call)
    return dict(sampled_ids=torch.tensor([[7]]), sampled_log_probs=torch.tensor([[-.5]]),
        actual_runner_observation=dict(request_outputs=[dict(request_id='r', prompt_token_ids=[1, 2],
            completions=[dict(index=0, token_ids=[7], chosen_token_log_probs=[-.5])])],
            calls=[call]))


def test_one_token_exact_four_way_chosen_probability_chain():
    result = _index_engine_observation(fixture())
    assert result['r']['tokens'][0]['selected_probability_chain_exact']
    assert result['r']['tokens'][0]['seq_id'] == 0


@pytest.mark.parametrize('failure', ['sequence', 'completion', 'raw', 'normalizer', 'multiple_calls', 'target'])
def test_each_probability_chain_link_is_independently_required(failure):
    raw = fixture()
    sample = raw['actual_runner_observation']['calls'][0]['actual_sampler_calls'][0]
    if failure == 'sequence': sample['selected_mapping'][0]['sequence_output_selected_log_prob'] = -.6
    if failure == 'completion': raw['actual_runner_observation']['request_outputs'][0]['completions'][0]['chosen_token_log_probs'] = [-.6]
    if failure == 'raw': raw['sampled_log_probs'][0, 0] = -.6
    if failure == 'normalizer': sample['actual_normalizer_selected_log_probs'][0] = -.6
    if failure == 'multiple_calls': sample['actual_normalizer_call_count'] = 2
    if failure == 'target': sample['selected_mapping'][0]['target_id'] = 8
    with pytest.raises(ValueError):
        _index_engine_observation(raw)


def two_request_fixture():
    """Both requests have length2, target7 and logp-.5; numerics cannot identify ownership."""
    raw = fixture()
    call = raw['actual_runner_observation']['calls'][0]
    call.update(input_tokens=torch.tensor([1, 2, 3, 4]), input_positions=torch.tensor([0, 1, 0, 1]),
        selected_token_indices=torch.tensor([1, 3]), query_lens=[2, 2], seq_lens=[2, 2],
        groups=[dict(request_id=name, seq_id=seq, sequence_length=2, query_length=2, prompt_length=2,
                     is_prompt=True, mrope_position_delta=None) for name, seq in [('A', 10), ('B', 11)]],
        decode_uses_cuda_graph=False, actual_cache_metadata={}, slot_mapping=torch.tensor([0, 1, 2, 3]))
    call['packed_input_mapping'] = build_packed_runner_input_mapping(call)
    call['actual_model_calls'] = [dict(args=(), input_ids=call['input_tokens'].clone(),
        positions=call['input_positions'].clone(), slot_mapping=call['slot_mapping'].clone())]
    sample = call['actual_sampler_calls'][0]
    sample.update(selected_mapping=[dict(request_id=name, seq_id=seq, selected_logit_row=selected,
        target_id=7, sequence_output_selected_log_prob=-.5, is_prompt=True)
        for selected, (name, seq) in enumerate([('A', 10), ('B', 11)])],
        actual_normalizer_selected_log_probs=torch.tensor([-.5, -.5]),
        pre_normalization=dict(pre_normalization_selected_logits_fp32=torch.tensor([1., 1.]),
                               fp32_logsumexp=torch.tensor([1.5, 1.5])))
    raw.update(sampled_ids=torch.tensor([[7], [7]]), sampled_log_probs=torch.tensor([[-.5], [-.5]]))
    raw['actual_runner_observation'].update(schema='ACTUAL_VLLM_RUNNER_INPUT_OBSERVATION_V1',
        observer_errors=[], actual_forward_calls_added=0, rng_restores=0,
        request_outputs=[dict(request_id=name, prompt_token_ids=prefix,
            completions=[dict(index=0, token_ids=[7], chosen_token_log_probs=[-.5])])
            for name, prefix in [('A', [1, 2]), ('B', [3, 4])]])
    return raw


def test_two_equal_length_requests_have_exact_physical_sequence_owners():
    raw = two_request_fixture()
    result = _index_engine_observation(raw)
    for name, seq, runner_index in [('A', 10, 1), ('B', 11, 3)]:
        token = result[name]['tokens'][0]
        assert token['seq_id'] == seq and token['runner_input_index'] == runner_index
        assert token['prediction_position'] == 1 and token['selected_runner_request_sequence_owner_exact']
        assert token['selected_probability_chain_exact'] and token['logp'] == -.5
    evidence = validate_actual_runner_observation(raw['actual_runner_observation'], [[1, 2], [3, 4]],
        raw['sampled_ids'], torch.tensor([1, 1]), raw['sampled_log_probs'])
    assert len(evidence) == 2 and all(row['actual_vllm_runner_positions_observed'] for row in evidence)


@pytest.mark.parametrize('failure', ['packed_indices', 'request', 'seq', 'logit_rows', 'duplicate_seq',
    'duplicate_request', 'missing_owner', 'forged_owner', 'out_of_range', 'duplicate_indices', 'missing_mapping'])
def test_identical_probability_requests_cannot_hide_swapped_or_missing_owners(failure):
    raw = two_request_fixture()
    call = raw['actual_runner_observation']['calls'][0]
    selected = call['actual_sampler_calls'][0]['selected_mapping']
    if failure == 'packed_indices': call['selected_token_indices'] = torch.tensor([3, 1])
    if failure == 'request': selected[0]['request_id'], selected[1]['request_id'] = 'B', 'A'
    if failure == 'seq': selected[0]['seq_id'], selected[1]['seq_id'] = 11, 10
    if failure == 'logit_rows': selected[0]['selected_logit_row'], selected[1]['selected_logit_row'] = 1, 0
    if failure == 'duplicate_seq': call['groups'][1]['seq_id'] = 10
    if failure == 'duplicate_request': call['groups'][1]['request_id'] = 'A'
    if failure == 'missing_owner': call['packed_input_mapping'].pop()
    if failure == 'forged_owner': call['packed_input_mapping'][1]['request_id'] = 'B'
    if failure == 'out_of_range': call['selected_token_indices'] = torch.tensor([1, 4])
    if failure == 'duplicate_indices': call['selected_token_indices'] = torch.tensor([1, 1])
    if failure == 'missing_mapping': selected.pop()
    # Every link still has the same target and scalar probability. Both actual
    # runner validation and decisive collection must reject structural identity.
    assert raw['sampled_ids'].tolist() == [[7], [7]] and raw['sampled_log_probs'].tolist() == [[-.5], [-.5]]
    with pytest.raises(ValueError):
        _index_engine_observation(raw)
    with pytest.raises(ValueError):
        validate_actual_runner_observation(raw['actual_runner_observation'], [[1, 2], [3, 4]],
            raw['sampled_ids'], torch.tensor([1, 1]), raw['sampled_log_probs'])


class TwoRequestSampler:
    """CPU sampler surrogate invokes the observed normalizer and returns two real-shaped outputs."""
    _do_penalties = _do_top_p_top_k = _do_min_p = False
    def __init__(self):
        self.calls = 0
    def forward(self, logits, metadata):
        self.calls += 1
        normalized = torch.log_softmax(logits, dim=-1, dtype=torch.float32)
        return SimpleNamespace(outputs=[SimpleNamespace(samples=[SimpleNamespace(parent_seq_id=group.seq_ids[0],
            output_token=7, logprobs={7: SimpleNamespace(logprob=float(normalized[group.sample_indices[0], 7]))})])
            for group in metadata.seq_groups])


@pytest.mark.parametrize('swapped', [False, True])
def test_actual_sampler_observer_cross_binds_two_packed_requests_without_extra_call(swapped):
    from decisive_normalizer_v1 import observe_vllm_sampler
    call = two_request_fixture()['actual_runner_observation']['calls'][0]
    call.pop('actual_sampler_calls')
    if swapped: call['selected_token_indices'] = torch.tensor([3, 1])
    params = SimpleNamespace(temperature=1, top_p=1, top_k=-1, n=1, min_p=0, min_tokens=0,
        presence_penalty=0, frequency_penalty=0, repetition_penalty=1, logits_processors=[])
    metadata = SimpleNamespace(seq_groups=[SimpleNamespace(seq_ids=[seq], sample_indices=[index],
        sampling_params=params, is_prompt=True) for index, seq in enumerate([10, 11])])
    sampler = TwoRequestSampler()
    model = SimpleNamespace(sampler=sampler, compute_logits=lambda value: value,
        logits_processor=SimpleNamespace(scale=1, soft_cap=None, vocab_size=8))
    runner = SimpleNamespace(model=model)
    original_torch = torch
    logits = original_torch.tensor([[0.] * 7 + [1.], [0.] * 7 + [1.]])
    with observe_vllm_sampler(runner, None, call, [6]):
        value = sampler.forward(model.compute_logits(logits), metadata)
    sample = call['actual_sampler_calls'][0]
    assert sampler.calls == 1 and len(value.outputs) == 2 and torch is original_torch
    assert 'forward' not in vars(sampler) and model.compute_logits(logits) is logits
    if swapped:
        assert sample['observer_errors'] and 'selected_mapping' not in sample
    else:
        assert not sample['observer_errors'] and sample['actual_normalizer_call_count'] == 1
        assert [(row['request_id'], row['seq_id'], row['selected_logit_row']) for row in sample['selected_mapping']] == [
            ('A', 10, 0), ('B', 11, 1)]
