"""Canonical reward provenance and actual sampler-length transport regressions."""
import ast
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest
import torch
from torch.nn.utils.rnn import pad_sequence
from search_r1.baselines.reward_view import reward_response_view

ROOT = Path(__file__).resolve().parents[2]
reward_fixture = runpy.run_path(str(ROOT / 'tests/reproduction/test_reward_and_validation.py'))
DataProto = reward_fixture['DataProto']
RewardManager = reward_fixture['RewardManager']
Tokenizer = reward_fixture['CharTokenizer']


def reward_batch(raw_text, canonical_text):
    prompt = torch.tensor([[ord('Q')]], dtype=torch.long)
    raw = torch.tensor([[ord(c) for c in raw_text]], dtype=torch.long)
    canonical = torch.tensor([[ord(c) for c in canonical_text]], dtype=torch.long)
    tensors = dict(prompts=prompt, responses=raw,
        attention_mask=torch.ones(1, 1 + raw.shape[1], dtype=torch.long),
        info_mask=torch.ones(1, 1 + raw.shape[1], dtype=torch.long),
        canonical_reward_responses=canonical,
        canonical_reward_attention_mask=torch.ones(1, 1 + canonical.shape[1], dtype=torch.long),
        canonical_reward_info_mask=torch.ones(1, 1 + canonical.shape[1], dtype=torch.long),
        canonical_reward_step_ids=torch.zeros_like(canonical))
    return DataProto.from_dict(tensors, non_tensors={
        'reward_model': [{'ground_truth': {'target': ['Paris']}}],
        'data_source': ['nq']})


@pytest.mark.parametrize('raw,canonical,expected', [
    ('<answer>Paris</answer><answer>London</answer>', '<answer>Paris</answer>', 1.),
    ('<search>x</search><answer>Paris</answer>', '<search>x</search>', 0.),
    ('<search><answer>Paris</answer></search>', '<search><answer>Paris</answer></search>', 0.),
])
def test_reward_uses_exact_canonical_span_and_raw_timeline_terminal(raw, canonical, expected):
    data = reward_batch(raw, canonical)
    manager = RewardManager(Tokenizer(), num_examine=0)
    result = manager(data)
    assert result.shape == data.batch['responses'].shape
    assert float(result.sum()) == expected
    assert float(result[0, -1]) == expected
    assert bool((result[0, :-1] == 0).all())


def test_incomplete_canonical_provenance_is_rejected():
    data = reward_batch('<answer>Paris</answer>', '<answer>Paris</answer>')
    data.batch.pop('canonical_reward_step_ids')
    with pytest.raises(ValueError, match='Incomplete canonical'):
        reward_response_view(data.batch)


def test_legacy_response_view_keeps_original_ids_masks():
    ids = torch.tensor([[1, 2, 3]])
    attention = torch.tensor([[1, 1, 1, 0]])
    info = torch.tensor([[1, 1, 0, 0]])
    actual = reward_response_view(dict(responses=ids, attention_mask=attention, info_mask=info))
    assert actual[0] is ids
    assert torch.equal(actual[1], attention[:, -3:])
    assert torch.equal(actual[2], info[:, -3:])


@pytest.mark.parametrize('version', ['vllm_v_0_5_4', 'vllm_v_0_6_3'])
def test_completion_lengths_distinguish_real_pad_from_added_pad_and_preserve_return_api(version):
    source = ROOT / ('verl/third_party/vllm/' + version + '/llm.py')
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'LLM')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_post_process_outputs')
    method.returns = None
    for arg in method.args.args:
        arg.annotation = None
    wrapper = ast.ClassDef(name='Fixture', bases=[], keywords=[], body=[method], decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[]))
    namespace = dict(torch=torch, pad_sequence=pad_sequence)
    exec(compile(module, str(source), 'exec'), namespace)
    engine = namespace['Fixture']()
    engine.llm_engine = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=151643, eos_token_id=151645))
    def completion(ids):
        return SimpleNamespace(token_ids=ids,
            logprobs=[{i: SimpleNamespace(logprob=-.5)} for i in ids])
    output = engine._post_process_outputs([
        SimpleNamespace(outputs=[completion([27, 151643, 397, 151645])]),
        SimpleNamespace(outputs=[completion([27, 151645])])])
    assert len(output) == 2
    assert engine.last_completion_token_lengths.tolist() == [4, 2]
    assert output[0][0].tolist() == [27, 151643, 397, 151645]
    assert output[0][1].tolist() == [27, 151645, 151643, 151643]
    assert output[1][0].tolist() == [-.5] * 4
    assert output[1][1].tolist() == [-.5, -.5, 151643., 151643.]


@pytest.mark.parametrize('canonical', [False, True])
def test_empty_response_does_not_score_prompt(canonical):
    data = reward_batch('', '')
    if not canonical:
        for key in list(data.batch.keys()):
            if key.startswith('canonical_reward_'):
                data.batch.pop(key)
    view = reward_response_view(data.batch)
    assert all(tuple(value.shape) == (1, 0) for value in view)
    result = RewardManager(Tokenizer(), num_examine=0)(data)
    assert tuple(result.shape) == (1, 0) and float(result.sum()) == 0
