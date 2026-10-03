"""Independent tiny-Qwen oracles for ranker generation/batching optimizations.

No mocked language-model logits: exercises fixed HF cache, real attention,
EOS=PAD, unpadding, sequence PPO gradients and Adam updates.
"""
import copy

import pytest
import torch
from transformers import GenerationConfig, Qwen2Config, Qwen2ForCausalLM

from search_r1.baselines.cosearch import (GenerativeRanker, sequence_ppo_loss,
                                         unpack_generation, eval_decoder_checkpoint)


def _model(kernel='sdpa'):
    torch.manual_seed(31)
    torch.set_num_threads(1)
    model = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=37, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=256, pad_token_id=0, eos_token_id=1,
        attention_dropout=0., attn_implementation=kernel)).eval()
    return model


def _generate(model, ids, mask, temperature=.7, eos=None, sample=False, maximum=8):
    config = GenerationConfig(max_new_tokens=maximum, do_sample=sample,
        temperature=temperature, top_k=0, top_p=1., num_beams=1,
        pad_token_id=eos if eos is not None else 0, eos_token_id=eos,
        return_dict_in_generate=True, output_logits=True)
    with torch.no_grad():
        return model.generate(input_ids=ids, attention_mask=mask,
                              generation_config=config)


def _oracle_events(output, ids, mask, eos, temperature):
    """Independent explicit per-row EOS/PAD oracle; prompt width is padded width."""
    width = ids.shape[1]
    generated = output.sequences[:, width:]
    assert generated.shape[1] == len(output.logits)
    events = []
    for row in range(ids.shape[0]):
        response = generated[row]
        length = response.numel()
        if eos is not None:
            ends = torch.nonzero(response == eos).flatten()
            if ends.numel():
                length = int(ends[0]) + 1
        response = response[:length]
        values = torch.stack([output.logits[t][row].float().div(temperature)
            .log_softmax(-1)[response[t]] for t in range(length)])
        prompt = ids[row][mask[row].bool()]
        events.append(dict(input_ids=torch.cat((prompt, response)),
                           response_length=length, old_logp=values))
    return events


def _full_single_logp(model, event, temperature):
    ids = event['input_ids'][None]
    length = event['response_length']
    logits = model(ids, use_cache=False).logits[:, -length-1:-1]
    return logits.float().div(temperature).log_softmax(-1).gather(
        -1, ids[:, -length:, None]).squeeze(-1)[0]


def _different_length_case(model):
    """Real greedy first-row EOS; find a second row that survives longer."""
    first = torch.tensor([[4, 5, 6, 7, 8]])
    with torch.no_grad():
        eos = int(model(first).logits[0, -1].argmax())
    for token in range(2, 37):
        second = torch.tensor([[token, (token+3) % 35+2, (token+11) % 35+2]])
        trial = _generate(model, second, torch.ones_like(second), eos=eos)
        if trial.sequences.shape[1] - second.shape[1] > 1:
            ids = torch.cat((first, torch.cat((torch.full((1, 2), eos), second), 1)), 0)
            mask = torch.tensor([[1, 1, 1, 1, 1], [0, 0, 1, 1, 1]])
            return ids, mask, eos
    raise AssertionError('Tiny Qwen fixture needs real unequal EOS lengths')


@pytest.mark.parametrize('kernel', ['eager', 'sdpa'])
@pytest.mark.parametrize('sample', [False, True])
@pytest.mark.parametrize('temperature', [.7, 1.3])
def test_raw_cached_logits_match_unpadded_full_policy(kernel, sample, temperature):
    model = _model(kernel)
    ids, mask, eos = _different_length_case(model)
    # Sampling tests also test PAD=EOS but do not require random early EOS.
    torch.manual_seed(113)
    output = _generate(model, ids, mask, temperature, eos, sample)
    oracle = _oracle_events(output, ids, mask, eos, temperature)
    events = unpack_generation(output, ids, mask, eos, temperature)
    for row, event in enumerate(events):
        assert event['input_ids'].device.type == 'cpu'
        assert event['old_logp'].device.type == 'cpu'
        assert event['response_length'] == oracle[row]['response_length']
        assert torch.equal(event['input_ids'], oracle[row]['input_ids'])
        torch.testing.assert_close(event['old_logp'], oracle[row]['old_logp'], atol=0, rtol=0)
        with torch.no_grad():
            full = _full_single_logp(model, event, temperature)
        torch.testing.assert_close(event['old_logp'], full, atol=2e-6, rtol=2e-6)
        assert torch.isfinite(event['old_logp']).all()
        assert event['input_ids'].numel() == int(mask[row].sum()) + event['response_length']
    if not sample:
        assert events[0]['response_length'] == 1
        assert events[1]['response_length'] > 1
        assert int(events[0]['input_ids'][-1]) == eos
        # The first row's subsequent tokens are forced PAD and must NOT be scored.
        assert (output.sequences[0, ids.shape[1]+1:] == eos).all()
        assert len(output.logits) > events[0]['old_logp'].numel()
        for row, event in enumerate(events):
            prompt = ids[row][mask[row].bool()][None]
            single = _generate(model, prompt, torch.ones_like(prompt), temperature, eos)
            assert torch.equal(single.sequences[0], event['input_ids'])


def test_left_padding_position_ids_preserve_batch_full_logp():
    model = _model()
    ids, mask, eos = _different_length_case(model)
    output = _generate(model, ids, mask, eos=eos)
    events = _oracle_events(output, ids, mask, eos, .7)
    length = len(output.logits)
    response_mask = torch.stack([torch.arange(length) < e['response_length'] for e in events]).long()
    full_mask = torch.cat((mask, response_mask), 1)
    position_ids = full_mask.cumsum(-1) - 1
    position_ids.masked_fill_(full_mask == 0, 1)
    with torch.no_grad():
        logits = model(output.sequences, attention_mask=full_mask,
                       position_ids=position_ids, use_cache=False).logits
        response = output.sequences[:, ids.shape[1]:]
        logp = logits[:, ids.shape[1]-1:-1].div(.7).log_softmax(-1).gather(-1, response[:, :, None]).squeeze(-1)
        for row, event in enumerate(events):
            torch.testing.assert_close(logp[row, :event['response_length']],
                _full_single_logp(model, event, .7), atol=2e-6, rtol=2e-6)


def test_cached_old_policy_preserves_all_gradients_and_adam_step():
    model = _model()
    ids, mask, eos = _different_length_case(model)
    output = _generate(model, ids, mask, eos=eos)
    events = _oracle_events(output, ids, mask, eos, .7)
    reference, cached = copy.deepcopy(model), copy.deepcopy(model)
    optimizers = [torch.optim.AdamW(m.parameters(), lr=.001, weight_decay=0.)
                  for m in (reference, cached)]
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    gradients = []
    for use_cached, candidate, optimizer in zip((False, True), (reference, cached), optimizers):
        optimizer.zero_grad()
        for index, event in enumerate(events):
            current = _full_single_logp(candidate, event, .7)
            with torch.no_grad():
                old = event['old_logp'] if use_cached else _full_single_logp(model, event, .7)
            # Non-unit ratio and opposite nonzero advantages; sequence-equal weights.
            loss = sequence_ppo_loss(current, old + .03, torch.tensor((.9, -.7)[index]), .2) / len(events)
            loss.backward()
        torch.nn.utils.clip_grad_norm_(candidate.parameters(), 1.)
        gradients.append({n: p.grad.detach().clone() for n, p in candidate.named_parameters()})
        optimizer.step()
    assert gradients[0].keys() == gradients[1].keys()
    assert sum(float(g.norm()) for g in gradients[0].values()) > 1e-4
    for name in gradients[0]:
        torch.testing.assert_close(gradients[0][name], gradients[1][name], atol=2e-6, rtol=2e-5)
    after = [dict(m.named_parameters()) for m in (reference, cached)]
    for name in before:
        torch.testing.assert_close(after[0][name], after[1][name], atol=2e-6, rtol=2e-5)
    assert sum(float((after[1][n].detach()-p).abs().sum()) for n, p in before.items()) > 1e-5
    for state_a, state_b in zip(optimizers[0].state.values(), optimizers[1].state.values()):
        for key in ('step', 'exp_avg', 'exp_avg_sq'):
            torch.testing.assert_close(state_a[key], state_b[key], atol=2e-6, rtol=2e-5)


def test_actual_eval_decoder_checkpoint_matches_policy_and_all_gradients():
    baseline = _model()
    ids, mask, eos = _different_length_case(baseline)
    output = _generate(baseline, ids, mask, eos=eos)
    events = unpack_generation(output, ids, mask, eos, .7)
    checkpointed = copy.deepcopy(baseline)
    calls = []
    policies, gradients = [], []
    for enabled, candidate in ((False, baseline), (True, checkpointed)):
        candidate.zero_grad()
        values = []
        with eval_decoder_checkpoint(candidate, enabled):
            if enabled:
                assert candidate.model.training
                assert all(not m.training for m in candidate.modules() if m is not candidate.model)
                original = candidate.model._gradient_checkpointing_func

                def counted_checkpoint(*args, **kwargs):
                    calls.append(True)
                    return original(*args, **kwargs)

                candidate.model._gradient_checkpointing_func = counted_checkpoint
            for index, event in enumerate(events):
                current = _full_single_logp(candidate, event, .7)
                values.append(current.detach())
                loss = sequence_ppo_loss(current, event['old_logp'] + .03,
                    torch.tensor((.9, -.7)[index]), .2) / len(events)
                loss.backward()
        assert all(not m.training for m in candidate.modules())
        policies.append(values)
        gradients.append({n: p.grad.detach().clone() for n, p in candidate.named_parameters()})
    assert len(calls) == checkpointed.config.num_hidden_layers * len(events)
    for left, right in zip(policies[0], policies[1]):
        torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-6)
    for name in gradients[0]:
        torch.testing.assert_close(gradients[0][name], gradients[1][name], atol=2e-6, rtol=2e-5)


def test_eval_decoder_checkpoint_restores_eval_after_exception():
    model = _model()
    with pytest.raises(RuntimeError, match='deliberate'):
        with eval_decoder_checkpoint(model, True):
            assert model.model.training
            raise RuntimeError('deliberate')
    assert all(not m.training for m in model.modules())


def test_unpack_multiple_eos_ids_include_first_eos_only():
    model = _model()
    ids, mask, eos = _different_length_case(model)
    output = _generate(model, ids, mask, eos=eos)
    single = unpack_generation(output, ids, mask, eos, .7)
    multiple = unpack_generation(output, ids, mask, [eos, 36 if eos != 36 else 35], .7)
    assert multiple[0]['response_length'] == single[0]['response_length'] == 1
    assert torch.equal(multiple[0]['input_ids'], single[0]['input_ids'])
    torch.testing.assert_close(multiple[0]['old_logp'], single[0]['old_logp'])


def test_real_rank_batch_restores_original_events_and_skips_validation_scoring():
    from types import SimpleNamespace
    from tests.baselines.test_models_and_driver import TinyModels, config
    TinyModels.setUpClass()
    try:
        settings = config('cosearch').baselines.ranker
        settings.model_path = TinyModels.tmp.name
        settings.attn_implementation = 'sdpa'
        settings.cpu_threads = 1
        settings.reuse_generation_logits = True
        settings.phase_gpu = False
        settings.inference_batch_size = 2
        settings.inference_token_budget = 10000
        settings.max_new_tokens = 5
        ranker = GenerativeRanker(settings)
        ranker.model.config.max_position_embeddings = 4096
        ranker.question_uids = ['question-0', 'question-1', 'question-2']
        questions = ['Question Paris ' * 15, 'Question', 'Question Rome ' * 4]
        queries = ['query-0', 'query-1', 'query-2']
        docs = [[{'document': {'contents': f'Paris row {row} doc {j}'}}
                 for j in range(50)] for row in range(3)]
        states = [SimpleNamespace(source=i, trajectory=f'trajectory-{i}') for i in range(3)]
        actual_generate = ranker.model.generate
        calls = []

        def observe_generate(*args, **kwargs):
            calls.append(dict(mask=kwargs['attention_mask'].detach().clone(),
                              logits=kwargs['generation_config'].output_logits))
            return actual_generate(*args, **kwargs)

        ranker.model.generate = observe_generate

        def forbid_second_score(*args, **kwargs):
            raise AssertionError('Optimized generation must not run response_logp')

        ranker.response_logp = forbid_second_score
        selected = ranker.rank_batch(questions, queries, docs, states)
        assert len(selected) == 3 and all(len(row) == settings.keep for row in selected)
        assert len(calls) == 2
        assert calls[0]['mask'].shape[0] == 2 and calls[1]['mask'].shape[0] == 1
        assert calls[0]['mask'].sum(1).max() < calls[1]['mask'].sum(1).max()
        assert len(ranker.events) == 3
        for index, event in enumerate(ranker.events):
            assert event['question_uid'] == f'question-{index}'
            assert event['trajectory_uid'] == f'trajectory-{index}'
            assert event['query'] == queries[index]
            assert event['candidates'] == [d['document']['contents'] for d in docs[index]]
            with torch.no_grad():
                full = _full_single_logp(ranker.model, event, settings.temperature)
            torch.testing.assert_close(event['old_logp'], full, atol=3e-6, rtol=3e-6)
        assert ranker.cost['score_forward_tokens'] == 0
        previous = len(ranker.events)
        calls.clear()
        ranker.validation = True
        ranker.rank_batch(questions, queries, docs, states)
        assert len(ranker.events) == previous
        assert len(calls) == 2 and all(not call['logits'] for call in calls)
        assert ranker.cost.get('validation_score_forward_tokens', 0) == 0
        assert ranker.cost['validation_rank_calls'] == 3
    finally:
        TinyModels.tearDownClass()


def test_production_update_preserves_sequence_equal_gradient_and_adam_state():
    from omegaconf import OmegaConf
    from search_r1.baselines.advantages import grouped_center
    from search_r1.baselines.cosearch import ranker_reward
    model = _model()
    ids, mask, eos = _different_length_case(model)
    output = _generate(model, ids, mask, eos=eos)
    events = unpack_generation(output, ids, mask, eos, .7)
    for i, event in enumerate(events):
        event.update(question_uid='same-question', trajectory_uid=f't-{i}',
                     query='same query', candidates=['Paris', 'Rome', 'Berlin', 'London', 'Tokyo', 'Madrid'],
                     ranking=[1, 2, 3] if i == 0 else [4, 5, 6])
    reference = copy.deepcopy(model)
    expected = torch.optim.AdamW(reference.parameters(), lr=.001, weight_decay=0.)
    rewards = [ranker_reward(e['candidates'], e['ranking'], ['Paris'], .4, .5) for e in events]
    advantage = grouped_center(torch.tensor(rewards), [0, 0])
    assert advantage.abs().min() > 0
    expected.zero_grad()
    for event, adv in zip(events, advantage):
        current = _full_single_logp(reference, event, .7)
        (sequence_ppo_loss(current, event['old_logp'], adv, .2) / len(events)).backward()
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.)
    reference_grad = {n: p.grad.detach().clone() for n, p in reference.named_parameters()}
    expected.step()
    ranker = GenerativeRanker.__new__(GenerativeRanker)
    ranker.model = copy.deepcopy(model)
    ranker.device = torch.device('cpu')
    ranker.config = OmegaConf.create(dict(temperature=.7, response_only_logits=True,
        phase_gpu=False, gradient_checkpointing=False, epochs=1, batch_size=8,
        similarity=.8, split_availability=False, min_group_size=2,
        threshold=.5, clip_ratio=.2, grad_clip=1.))
    ranker.optimizer = torch.optim.AdamW(ranker.model.parameters(), lr=.001, weight_decay=0.)
    ranker.events = copy.deepcopy(events)
    ranker.cost = dict(training_forward_tokens=0, optimizer_steps=0)
    metrics = ranker.update({f't-{i}': (.4, ['Paris'], True) for i in range(2)})
    assert metrics['ranker/training_events'] == 2
    assert ranker.cost['optimizer_steps'] == 1
    assert metrics['ranker/gradient_norm_max'] > 1e-5
    assert metrics['ranker/parameter_probe_delta_l1'] > 0
    assert not ranker.events
    for name, parameter in ranker.model.named_parameters():
        torch.testing.assert_close(parameter.grad, reference_grad[name], atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(parameter, dict(reference.named_parameters())[name], atol=2e-6, rtol=2e-5)
    for expected_state, actual_state in zip(expected.state.values(), ranker.optimizer.state.values()):
        for key in ('step', 'exp_avg', 'exp_avg_sq'):
            torch.testing.assert_close(expected_state[key], actual_state[key], atol=2e-6, rtol=2e-5)


def test_gpu_phase_rejects_missing_owned_allocation_before_cuda(monkeypatch):
    from omegaconf import OmegaConf
    ranker = GenerativeRanker.__new__(GenerativeRanker)
    ranker.config = OmegaConf.create(dict(phase_gpu=True))
    for key in ('SLURM_JOB_ID', 'SLURM_JOB_NODELIST', 'COSEARCH_RANKER_GPU_UUID',
                'COSEARCH_RANKER_LEASE_UUIDS'):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(RuntimeError, match='owned h100-16'):
        with ranker.execution_phase():
            raise AssertionError('Unowned phase must not execute')
