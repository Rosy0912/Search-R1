import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest
import torch

from search_r1.qa_protocol import decode_model_spans, extract_final_answer, parse_model_action
from search_r1.baselines.bapo import BoundaryReward, BoundaryStage, decode_answers
from search_r1.baselines.baseline_alignment_data import (
    order_from_rng, order_manifest, RecordedOrderSampler, validate_emitted_batch,
    prepared_manifest, verify_prepared_manifest, verify_stage_files,
)


class CharTokenizer:
    def decode(self, ids):
        return ''.join(chr(int(i)) for i in ids)


def trajectory(spans, observations=None):
    texts, mask = [], []
    for i, span in enumerate(spans):
        texts.extend(map(ord, span)); mask.extend([1] * len(span))
        if i + 1 < len(spans):
            observation = (observations or ['<information>observation</information>'])[i]
            texts.extend(map(ord, observation)); mask.extend([0] * len(observation))
    return SimpleNamespace(batch=dict(responses=torch.tensor([texts]),
                                     info_mask=torch.tensor([mask]),
                                     attention_mask=torch.ones((1, len(texts)), dtype=torch.long)),
                           non_tensor_batch=dict(action_records=[[dict(text=s) for s in spans]],
                               reward_model=[dict(ground_truth=dict(target=['Paris']))], uid=['q']))


@pytest.fixture
def shared_decode():
    # Avoid initializing Ray; exercise the identical shared decoder used by main_ppo.
    module = SimpleNamespace(decode_model_response_spans=decode_model_spans)
    with patch.dict('sys.modules', {'verl.trainer.main_ppo': module}):
        yield


def test_aligned_terminal_grading_does_not_add_whole_chain_penalty(shared_decode):
    data = trajectory(['free reasoning <search>capital</search>', '<answer>Paris</answer>'])
    tokenizer = CharTokenizer()
    assert decode_answers(data, tokenizer, require_legal=True) == [None]
    assert decode_answers(data, tokenizer, require_legal=True, profile='baseline_aligned') == ['Paris']
    legacy = BoundaryReward(tokenizer, BoundaryStage())(data)
    aligned = BoundaryReward(tokenizer, BoundaryStage(), profile='baseline_aligned')(data)
    assert legacy.sum().item() == -1
    assert aligned.sum().item() == 1


@pytest.mark.parametrize('final', [
    '<answer>Paris<search>x</search></answer>', '<answer></answer>',
    '<answer>Paris</answer>trailing', '<search>Paris</search>',
])
def test_aligned_retains_format_penalty_and_terminal_rules(final, shared_decode):
    data = trajectory(['<answer>Paris</answer>', final])
    assert BoundaryReward(CharTokenizer(), BoundaryStage(), profile='baseline_aligned')(data).sum().item() == -1


def test_observation_answer_never_reused_or_joined(shared_decode):
    tokenizer = CharTokenizer()
    data = trajectory(['<search>x</search>', '<search>y</search>'], ['<information><answer>Paris</answer></information>'])
    assert decode_answers(data, tokenizer, profile='baseline_aligned') == [None]
    data = trajectory(['<answer>Pa', 'ris</answer>'], ['<information>x</information>'])
    assert decode_answers(data, tokenizer, profile='baseline_aligned') == [None]
    assert parse_model_action('free reasoning <search>x</search>') == ('search', 'x')
    assert extract_final_answer(['<answer>Paris</answer>', '<search>x</search>']) is None


def test_bapo_boundary_bonus_and_invalid_penalty_preserved(shared_decode):
    data = trajectory(['<answer>I do not know</answer>'])
    # Later-stage diversity gate is open for a homogeneous group of siblings.
    data.batch = {k: v.repeat(5, 1) for k, v in data.batch.items()}
    data.non_tensor_batch = {k: v * 5 for k, v in data.non_tensor_batch.items()}
    scores = BoundaryReward(CharTokenizer(), BoundaryStage(early=False), profile='baseline_aligned')(data)
    assert scores.sum(dim=1).tolist() == [.5] * 5


def test_recorded_order_matches_real_baseline_loader_after_rng_consumption():
    ids = [f'nq:train:{i}' for i in range(19)]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        torch.rand(37)  # Models/earlier code can advance the global RNG.
        state = torch.get_rng_state().clone()
        manifest = order_from_rng(ids, 4, state, epochs=2)
        assert torch.equal(torch.get_rng_state(), state)
        loader = torch.utils.data.DataLoader(list(range(19)), batch_size=4, shuffle=True,
                                           num_workers=0, drop_last=True)
        actual = [[i for batch in loader for i in batch.tolist()] for _ in range(2)]
        assert manifest['epochs'] == actual
    sampler = RecordedOrderSampler(ids, 4, manifest)
    replay = torch.utils.data.DataLoader(list(range(19)), batch_size=4, sampler=sampler,
                                        num_workers=0, drop_last=True)
    for epoch in range(2):
        got = []
        for batch_index, batch in enumerate(replay):
            got.extend(batch.tolist())
            validate_emitted_batch(manifest, epoch, batch_index, [ids[i] for i in batch.tolist()])
        assert got == actual[epoch]


def test_shared_order_resume_and_mismatch_fail_closed():
    ids = [f'nq:train:{i}' for i in range(10)]
    manifest = order_manifest(ids, 4, [[3, 5, 0, 7, 1, 8, 4, 2]])
    sampler = RecordedOrderSampler(ids, 4, manifest)
    iterator = iter(sampler)
    assert [next(iterator) for _ in range(4)] == [3, 5, 0, 7]
    resumed = RecordedOrderSampler(ids, 4, manifest)
    resumed.load_state_dict(sampler.state_dict())
    assert list(resumed) == [1, 8, 4, 2]
    with pytest.raises(ValueError, match='Actual question batch'):
        validate_emitted_batch(manifest, 0, 0, ids[:4])
    changed = copy.deepcopy(manifest); changed['epochs'][0][0] = 9
    with pytest.raises(ValueError, match='modified'):
        RecordedOrderSampler(ids, 4, changed)
    with pytest.raises(ValueError, match='dataset or batch size'):
        RecordedOrderSampler(list(reversed(ids)), 4, manifest)


def test_imported_shared_data_binds_prompt_targets_ids_and_row_order(tmp_path):
    def row(index):
        return dict(extra_info=dict(index=index), data_source='nq',
                    prompt=[dict(role='user', content='Question: ' + index)],
                    reward_model=dict(ground_truth=dict(target=['Paris'])))
    files = [tmp_path / (name + '.parquet') for name in ('train', 'validation', 'test')]
    frames = [pd.DataFrame([row('nq:train:1'), row('hotpotqa:train:3')]),
              pd.DataFrame([row('nq:train:2')]), pd.DataFrame([row('nq:test:1')])]
    for path, frame in zip(files, frames):
        frame.to_pickle(path)
    # File proof is real; substitute parquet decoding to keep this CPU test independent of Arrow.
    with patch('pandas.read_parquet', pd.read_pickle):
        manifest = prepared_manifest(*files)
        verify_prepared_manifest(manifest, *files)
        frames[0].iloc[::-1].to_pickle(files[0])
        with pytest.raises(ValueError, match='semantic_sha256'):
            verify_prepared_manifest(manifest, *files)
        frames[0].to_pickle(files[0])
        frames[0].at[0, 'reward_model'] = dict(ground_truth=dict(target=['London']))
        frames[0].to_pickle(files[0])
        with pytest.raises(ValueError, match='semantic_sha256'):
            verify_prepared_manifest(manifest, *files)


def test_stage_preserves_explicit_common_train_and_rejects_dev_final_overlap(tmp_path):
    def row(index, content):
        return dict(extra_info=dict(index=index), data_source='nq',
                    prompt=[dict(role='user', content=content)],
                    reward_model=dict(ground_truth=dict(target=['Paris'])))
    files = [tmp_path / (name + '.parquet') for name in ('train', 'validation', 'test', 'stage')]
    frames = [pd.DataFrame([row('nq:train:1', 'train question')]),
              pd.DataFrame([row('nq:train:2', 'dev question')]),
              pd.DataFrame([row('nq:test:1', 'test question')]),
              pd.DataFrame([row('nq:train:1', 'train question')])]
    for path, frame in zip(files, frames):
        frame.to_pickle(path)
    with patch('pandas.read_parquet', pd.read_pickle):
        with pytest.raises(ValueError, match='overlaps shared train'):
            prepared_manifest(*files)
        manifest = prepared_manifest(*files, stage_train_overlap_policy='preserve-common-baseline-train')
        proof = verify_stage_files(manifest, files[3], *files[:3])
        assert proof['train_rows_preserved'] == 1
        assert proof['training_derived'] and not proof['independent_holdout']
        # IDs differ, but the identical full shared prompt still detects leakage.
        pd.DataFrame([row('stage:synthetic:9', 'dev question')]).to_pickle(files[3])
        with pytest.raises(ValueError, match='dev/final'):
            prepared_manifest(*files, stage_train_overlap_policy='preserve-common-baseline-train')
