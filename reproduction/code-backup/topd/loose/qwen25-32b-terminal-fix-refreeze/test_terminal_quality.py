"""Real repaired source metrics on synthetic tokens; no model quality evidence."""
import numpy as np
import pytest
import torch
from test_eval64 import Tokenizer
from verl import DataProto
from search_r1.baselines.evaluation import batch_answer_metrics
import eval64


def metric(text, observation=None):
    tok = Tokenizer()
    ids = tok.encode(text)
    mask = [1] * len(ids)
    if observation is not None:
        inserted = tok.encode(observation)
        ids = inserted + ids
        mask = [0] * len(inserted) + mask
    data = DataProto.from_dict(tensors=dict(
        responses=torch.tensor([ids]), attention_mask=torch.ones((1, len(ids)), dtype=torch.long),
        info_mask=torch.tensor([mask])), non_tensors=dict(
        reward_model=np.array([dict(ground_truth=dict(target=['Paris']))], dtype=object),
        data_source=np.array(['synthetic-protocol'], dtype=object)))
    return batch_answer_metrics(data, tok)[0]


@pytest.mark.parametrize('text', [
    '<search><answer>Paris</answer></search>',
    '<think><answer>Paris</answer></think>',
    '<answer>Paris</answer><search>next</search>',
    '<answer>Paris</answer><answer>Paris</answer>',
    '<information><answer>Paris</answer></information>',
    '<answer>Paris</answer> trailing',
])
def test_false_terminal_answer_never_counts_toward_26(text):
    row = metric(text)
    assert row['em'] == row['f1'] == row['answer_format'] == 0
    rows = [dict(uid=str(i), sample_position=i, em=float(i < 25), f1=float(i < 25)) for i in range(64)]
    rows[25].update(em=row['em'], f1=row['f1'])
    assert eval64.quality_gate(rows)['correct'] == 25
    assert not eval64.quality_gate(rows)['candidate_quality_accepted']


def test_real_terminal_answer_after_observation_is_26th_correct():
    row = metric('<think>Found it.</think><answer>Paris</answer>', '<information><answer>London</answer></information>')
    assert row['em'] == row['f1'] == row['answer_format'] == row['format_rate'] == 1
    rows = [dict(uid=str(i), sample_position=i, em=float(i < 25), f1=float(i < 25)) for i in range(64)]
    rows[25].update(em=row['em'], f1=row['f1'])
    assert eval64.quality_gate(rows)['correct'] == 26
    assert eval64.quality_gate(rows)['candidate_quality_accepted']


def test_missing_new_protocol_source_cannot_authorize_quality(inventory, tmp_path):
    import json
    root, _, _, binding = inventory
    del binding['source_files']['search_r1/qa_protocol.py']
    path = tmp_path / 'missing-protocol.json'
    path.write_text(json.dumps(binding))
    with pytest.raises(RuntimeError, match='Missing reused-source'):
        eval64.asset_gate(path, root)


def test_old_repo_cannot_substitute_new_source(inventory):
    root, _, binding_path, _ = inventory
    (root / 'repo').mkdir()
    # Move the required new source out: a similarly named old tree cannot pass.
    source = root / 'repo-terminal-fixed413/search_r1/qa_protocol.py'
    target = root / 'repo/search_r1/qa_protocol.py'
    target.parent.mkdir(parents=True)
    source.rename(target)
    with pytest.raises((RuntimeError, FileNotFoundError)):
        eval64.asset_gate(binding_path, root)
