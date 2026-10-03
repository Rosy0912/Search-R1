"""Independent actual-selector preview/update and zero-signal gate regressions."""
import copy

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from search_r1.baselines.controller import Controller
from tests.baselines.test_cosearch_numpy_update import _ranker


@pytest.mark.parametrize('case', ['informative', 'constant', 'illegal_main', 'under_minimum',
                                 'different_query', 'trivial', 'zero_response', 'mixed_constant_subset'])
def test_preview_is_pure_and_exactly_matches_actual_update_selection(case, monkeypatch):
    monkeypatch.delenv('COSEARCH_UPDATE_EVIDENCE_ROOT', raising=False)
    ranker = _ranker()
    ranker.config.min_group_size = 3
    third = copy.deepcopy(ranker.events[1])
    third['trajectory_uid'] = 't-2'
    ranker.events.append(third)
    aliases = np.array(['Paris', 'City of Paris'], dtype=object)
    mains = [.4] * 3
    legal = [True] * 3
    if case == 'constant':
        for event in ranker.events:
            event['ranking'] = None
    elif case == 'illegal_main':
        legal = [False] * 3
    elif case == 'under_minimum':
        legal[-1] = False
    elif case == 'different_query':
        ranker.events[-1]['query'] = 'unrelated subject'
    elif case == 'trivial':
        aliases = np.array(['YES', 'no'], dtype=object)
        mains = [.4, .9, .2]
    elif case == 'zero_response':
        for event in ranker.events:
            event['response_length'] = 0
            event['old_logp'] = torch.empty(0)
    elif case == 'mixed_constant_subset':
        for i in range(3, 6):
            event = copy.deepcopy(third)
            event.update(question_uid='unrelated-question', trajectory_uid=f't-{i}', ranking=None)
            ranker.events.append(event)
        mains += [.4] * 3
        legal += [True] * 3
    outcomes = {f't-{i}': (mains[i], aliases, legal[i]) for i in range(len(ranker.events))}
    before = copy.deepcopy(ranker.events)
    identities = [id(e) for e in ranker.events]
    rng = torch.get_rng_state().clone()
    parameters = [p.detach().clone() for p in ranker.model.parameters()]
    events, keys, chosen, advantages, preview = ranker.select_update_events(outcomes, copy_events=True)
    assert torch.equal(torch.get_rng_state(), rng)
    assert [id(e) for e in ranker.events] == identities
    for original, saved in zip(ranker.events, before):
        assert original.keys() == saved.keys()
        for name in original:
            if torch.is_tensor(original[name]):
                assert torch.equal(original[name], saved[name])
            else:
                assert original[name] == saved[name]
    assert not ranker.optimizer.state
    assert all(torch.equal(p, q) for p, q in zip(ranker.model.parameters(), parameters))
    assert all(id(e) not in identities for e in events)
    controller = Controller.__new__(Controller)
    controller.ranker = ranker
    controller.config = OmegaConf.create(dict(require_informative_first_batch=True))
    controller.prepare_ranker_outcomes = lambda data: outcomes
    if preview['ranker/effective_nonzero_advantage_events']:
        controller.before_actor(None, {})
        assert controller._first_ranker_batch_checked
    else:
        with pytest.raises(RuntimeError, match='no informative ranker learning signal'):
            controller.before_actor(None, {})
        assert not getattr(controller, '_first_ranker_batch_checked', False)
    assert [id(e) for e in ranker.events] == identities
    assert torch.equal(torch.get_rng_state(), rng)
    original_selector = ranker.select_update_events
    observed = {}
    def observe(*args, **kwargs):
        selected = original_selector(*args, **kwargs)
        observed['keys'], observed['chosen'] = selected[1], selected[2]
        observed['advantage'] = selected[3].clone()
        return selected
    ranker.select_update_events = observe
    actual = ranker._update(outcomes)
    assert observed['keys'] == keys and observed['chosen'] == chosen
    torch.testing.assert_close(observed['advantage'], advantages, atol=0, rtol=0)
    for name, value in preview.items():
        assert actual[name] == value
    assert not ranker.events
    assert ranker.cost['optimizer_steps'] == int(bool(chosen))
    if chosen:
        assert actual['ranker/gradient_norm_max'] > 1e-5
    else:
        assert all(torch.equal(p, q) for p, q in zip(ranker.model.parameters(), parameters))
