"""UNEXECUTED scalar chain fixtures; no parity, model, sampling or GPU evidence."""
import copy
import pytest
import torch
from decisive_normalizer_v1 import _index_engine_observation


def fixture():
    sample = dict(observer_errors=[], actual_normalizer_call_count=1, processor_logits_unchanged=True,
        selected_mapping=[dict(request_id='r', seq_id=0, selected_logit_row=0, target_id=7,
                               sequence_output_selected_log_prob=-.5)],
        actual_normalizer_selected_log_probs=torch.tensor([-.5]),
        pre_normalization=dict(pre_normalization_selected_logits_fp32=torch.tensor([1.]),
                               fp32_logsumexp=torch.tensor([1.5])),
        normalizer_path='actual observed callable', actual_normalizer_output_dtype='torch.float32')
    return dict(sampled_ids=torch.tensor([[7]]), sampled_log_probs=torch.tensor([[-.5]]),
        actual_runner_observation=dict(request_outputs=[dict(request_id='r', prompt_token_ids=[1, 2],
            completions=[dict(index=0, token_ids=[7], chosen_token_log_probs=[-.5])])],
            calls=[dict(observer_errors=[], actual_logits_processor=dict(scale=1, soft_cap=None),
                selected_token_indices=torch.tensor([0]), input_positions=torch.tensor([1]), actual_sampler_calls=[sample])]))


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
