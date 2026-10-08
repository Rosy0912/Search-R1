"""Regression counterexamples from the pinned CoSearch source audit."""
from types import SimpleNamespace
import torch
from search_r1.baselines.cosearch import (
    GenerativeRanker, contains_answer, answer_available, ranker_reward,
    query_f1, semantic_groups, sequence_ppo_loss,
)


def test_exact_document_labels_preserve_articles_punctuation_and_unicode():
    assert not contains_answer('I live in Hague', ['The Hague'])
    assert contains_answer('I live in THE Hague.', ['The Hague'])
    assert not contains_answer('foobar', ['foo-bar'])
    assert contains_answer('foo-bar', ['foo-bar'])
    assert contains_answer('A cafe\u0301 in Paris', ['CAFÉ'])
    assert not contains_answer('anything', [''])
    assert not contains_answer('Roman', ['Rome'])


def test_trivial_boolean_gold_never_creates_document_supervision():
    docs = ['Yes no true false this is unrelated', 'noise', 'noise']
    for targets in (['yes'], ['NO', ' True '], ['false']):
        assert not answer_available(docs, targets)
        assert ranker_reward(docs, [1, 2, 3], targets, .25) == .25
        assert ranker_reward(docs, None, targets, .25) == -.2
    assert answer_available(['Paris says yes'], ['yes', 'Paris'])


def test_query_normalization_separates_punctuation_instead_of_merging_words():
    assert query_f1('the foo-bar', 'foo bar') == 1.
    assert query_f1('foobar', 'foo bar') == 0.
    assert query_f1('the', 'a') == 0.
    keys = semantic_groups([dict(question_uid='q', query=q) for q in ['foo-bar', 'foo bar', 'foobar']])
    assert keys[0] == keys[1] and keys[0] != keys[2]


def test_sequence_objective_does_not_weight_longer_rankings_more():
    parameter = torch.tensor(0., requires_grad=True)
    short = sequence_ppo_loss(parameter.expand(1), torch.zeros(1), torch.tensor(1.), .2)
    long = sequence_ppo_loss(parameter.expand(3), torch.zeros(3), torch.tensor(-1.), .2)
    ((short + long) / 2).backward()
    assert parameter.grad == 0.
    bad = torch.tensor([float('nan')])
    try:
        sequence_ppo_loss(bad, torch.zeros(1), torch.tensor(1.), .2)
    except ValueError:
        pass
    else:
        raise AssertionError('nonfinite policy input must fail before optimizer update')


def test_same_reward_groups_skip_adamw_including_existing_momentum():
    ranker = GenerativeRanker.__new__(GenerativeRanker)
    ranker.model = torch.nn.Linear(1, 1)
    ranker.optimizer = torch.optim.AdamW(ranker.model.parameters(), lr=.1, weight_decay=.1)
    # Populate optimizer momentum, proving a skipped group cannot cause a stale update.
    ranker.model(torch.ones(1, 1)).sum().backward()
    ranker.optimizer.step()
    before = [p.detach().clone() for p in ranker.model.parameters()]
    ranker.device = torch.device('cpu')
    ranker.config = SimpleNamespace(threshold=.5, similarity=.8, split_availability=False,
                                   min_group_size=3, epochs=1, batch_size=3)
    ranker.cost = dict(optimizer_steps=0)
    ranker.events = [dict(question_uid='q', trajectory_uid=str(i), query='capital france',
                         candidates=['Paris'], ranking=[1], response_length=1) for i in range(3)]
    metrics = ranker.update({str(i): (0., ['Paris'], True) for i in range(3)})
    assert metrics['ranker/training_events'] == 0
    assert metrics['ranker/same_reward_groups_skipped'] == 1
    assert ranker.cost['optimizer_steps'] == 0
    assert all(torch.equal(p, old) for p, old in zip(ranker.model.parameters(), before))
    assert ranker.events == []


def test_unpatched_hf_generation_accepts_generic_tokenizer_and_validation_costs():
    from test_models_and_driver import TinyModels, config
    TinyModels.setUpClass()
    try:
        c = config('cosearch').baselines.ranker
        c.model_path = TinyModels.tmp.name
        c.max_new_tokens = 4
        ranker = GenerativeRanker(c)
        ranker.question_uids = ['q']
        docs = [{'document': {'contents': d}} for d in ['Paris', 'Rome', 'Berlin']]
        ranker.rank('Capital?', 'capital', docs, SimpleNamespace(source=0, trajectory='t'))
        event = ranker.events[0]
        ids = event['input_ids'][None, :]
        length = event['response_length']
        with torch.no_grad():
            expected = ranker.model(ids, use_cache=False).logits[:, -length - 1:-1].float()
            expected = torch.log_softmax(expected / c.temperature, -1).gather(-1, ids[:, -length:, None])
        torch.testing.assert_close(event['old_logp'], expected[0, :, 0])
        training_score_tokens = ranker.cost['score_forward_tokens']
        ranker.validation = True
        ranker.rank('Capital?', 'capital', docs, SimpleNamespace(source=0, trajectory='t'))
        assert len(ranker.events) == 1
        assert ranker.cost['score_forward_tokens'] == training_score_tokens
        assert ranker.cost['validation_score_forward_tokens'] > 0
    finally:
        TinyModels.tearDownClass()
