"""Independent CPU regression of actual update with Parquet-style alias arrays."""
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from transformers import Qwen2Config, Qwen2ForCausalLM

from search_r1.baselines.cosearch import GenerativeRanker


def _ranker():
    torch.manual_seed(71)
    torch.set_num_threads(1)
    ranker = GenerativeRanker.__new__(GenerativeRanker)
    ranker.model = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=37, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=256, attention_dropout=0.,
        attn_implementation='sdpa')).eval()
    ranker.device = torch.device('cpu')
    ranker.config = OmegaConf.create(dict(temperature=.7, response_only_logits=True,
        gradient_checkpointing=False, training_parallel=False, epochs=1, batch_size=8,
        similarity=.8, split_availability=True, min_group_size=2,
        threshold=.5, clip_ratio=.2, grad_clip=1.))
    # Zero decay: this regression must demonstrate reward-driven change.
    ranker.optimizer = torch.optim.AdamW(ranker.model.parameters(), lr=.001, weight_decay=0.)
    ranker.events = []
    for i, ids in enumerate(([2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12])):
        ids = torch.tensor(ids)
        with torch.no_grad():
            old = ranker.response_logp(ids[None], 2).squeeze(0).clone()
        ranker.events.append(dict(question_uid='question', trajectory_uid=f't-{i}',
            query='same query', candidates=['Paris', 'Rome', 'Berlin', 'London', 'Tokyo', 'Madrid'],
            ranking=[1, 2, 3] if i == 0 else [4, 5, 6], input_ids=ids,
            response_length=2, old_logp=old))
    ranker.cost = dict(training_forward_tokens=0, optimizer_steps=0)
    return ranker


@pytest.mark.parametrize('aliases', [[], ['Paris'], ['Paris', 'City of Paris'], ['yes', 'NO']])
@pytest.mark.parametrize('distinct_main_rewards', [False, True])
def test_actual_update_numpy_aliases_match_list_rewards_groups_gradients_and_adam(aliases, distinct_main_rewards, monkeypatch, tmp_path):
    monkeypatch.delenv('COSEARCH_UPDATE_EVIDENCE_ROOT', raising=False)
    expected = _ranker()
    actual = _ranker()
    baseline = {n: p.detach().clone() for n, p in actual.model.named_parameters()}
    mains = [.4, .9] if distinct_main_rewards else [.4, .4]
    expected_outcomes = {f't-{i}': (mains[i], list(aliases), True) for i in range(2)}
    actual_outcomes = {f't-{i}': (mains[i], np.array(aliases, dtype=object), True) for i in range(2)}
    # Invoke production reward preparation, grouping, forward, clip and optimizer.
    expected_metrics = expected._update(expected_outcomes)
    monkeypatch.setenv('COSEARCH_UPDATE_EVIDENCE_ROOT', str(tmp_path))
    actual_metrics = actual._update(actual_outcomes)
    import json
    evidence = json.loads(next((tmp_path / 'ranker-update-evidence').glob('*.json')).read_text())
    assert evidence['selection_unchanged'] is True
    assert len(evidence['events']) == 2
    assert evidence['events'][0]['targets'] == aliases
    assert evidence['events'][0]['ranking_response_ids'] == [5, 6]
    for key in ('ranker/training_events', 'ranker/same_reward_groups_skipped',
                'ranker/trivial_answer_events', 'ranker/informative_groups',
                'ranker/effective_nonzero_advantage_events', 'ranker/gradient_norm_max',
                'ranker/parameter_probe_delta_l1'):
        assert actual_metrics[key] == expected_metrics[key]
    assert not actual.events and not expected.events
    informative = bool(aliases and aliases[0] == 'Paris') or distinct_main_rewards
    assert actual.cost['optimizer_steps'] == int(informative)
    assert actual_metrics['ranker/training_events'] == (2 if informative else 0)
    changed = False
    for (name, p), (other_name, q) in zip(actual.model.named_parameters(), expected.model.named_parameters()):
        assert name == other_name
        torch.testing.assert_close(p, q, atol=0, rtol=0)
        changed |= not torch.equal(p, baseline[name])
        if informative:
            torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)
        else:
            assert p.grad is None and q.grad is None
    assert changed == informative
    if informative:
        assert actual_metrics['ranker/gradient_norm_max'] > 1e-5
        assert actual_metrics['ranker/parameter_probe_delta_l1'] > 0
    for left, right in zip(actual.optimizer.state.values(), expected.optimizer.state.values()):
        for key in ('step', 'exp_avg', 'exp_avg_sq'):
            torch.testing.assert_close(left[key], right[key], atol=0, rtol=0)
