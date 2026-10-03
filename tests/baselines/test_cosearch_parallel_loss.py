"""Independent CPU oracle for production batched ranker sequence PPO.

This verifies math, not CUDA DataParallel scatter/reduction, eight-GPU memory,
HF replica thread safety, or native GPU performance.
"""
import copy

import pytest
import torch
import torch.nn.functional as F
from transformers import Qwen2Config, Qwen2ForCausalLM

from search_r1.baselines.cosearch import (RankerSequenceLoss,
    eval_decoder_checkpoint, sequence_ppo_loss)


def _model(kernel):
    torch.manual_seed(902)
    torch.set_num_threads(1)
    return Qwen2ForCausalLM(Qwen2Config(vocab_size=43, hidden_size=16,
        intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=2, max_position_embeddings=256,
        eos_token_id=0, pad_token_id=0, attention_dropout=0.,
        attn_implementation=kernel)).eval()


def _single_logp(model, ids, response_length, temperature):
    logits = model(ids[None], use_cache=False).logits[:, -response_length-1:-1]
    return logits[0].float().div(temperature).log_softmax(-1).gather(
        -1, ids[-response_length:, None]).squeeze(-1)


def _examples(model, rows, temperature):
    events = []
    for i in range(rows):
        prompt = torch.tensor([2 + (i*3+j) % 39 for j in range(1 + (i*7) % 8)])
        response_length = 1 + (i*3) % 6
        # PAD=EOS appears as the actual final response action and stays scored.
        response = torch.tensor([3 + (i+j*5) % 37 for j in range(response_length-1)] + [0])
        ids = torch.cat((prompt, response))
        with torch.no_grad():
            logp = _single_logp(model, ids, response_length, temperature)
        # Exercise both clipped and unclipped ratios, not only ratio=1.
        offset = (0.0, .5, -.4, .03)[i % 4]
        events.append(dict(input_ids=ids, response_length=response_length,
                           old_logp=logp + offset))
    advantages = torch.tensor([(.8, -.7, 0., 1.1, -.2, .4, -.6, .3)[i % 8]
                               for i in range(rows)])
    return events, advantages


def _pack(events):
    width = max(e['input_ids'].numel() for e in events)
    response_width = max(e['response_length'] for e in events)
    ids = torch.stack([F.pad(e['input_ids'], (width-e['input_ids'].numel(), 0), value=0)
                       for e in events])
    masks = torch.stack([F.pad(torch.ones_like(e['input_ids']),
                         (width-e['input_ids'].numel(), 0)) for e in events])
    old = torch.stack([F.pad(e['old_logp'], (response_width-e['response_length'], 0))
                       for e in events])
    lengths = torch.tensor([e['response_length'] for e in events])
    return ids, masks, lengths, old, response_width


@pytest.mark.parametrize('rows', [8, 3, 1])
@pytest.mark.parametrize('kernel', ['eager', 'sdpa'])
@pytest.mark.parametrize('checkpoint', [False, True])
@pytest.mark.parametrize('temperature', [.7, 1.3])
def test_production_batched_loss_all_gradients_clip_two_adam_steps(rows, kernel, checkpoint, temperature):
    clip_ratio = .2
    initial = _model(kernel)
    events, advantages = _examples(initial, rows, temperature)
    ids, mask, lengths, old, response_width = _pack(events)
    reference, batched = copy.deepcopy(initial), copy.deepcopy(initial)
    before = {n: p.detach().clone() for n, p in initial.named_parameters()}
    optimizers = [torch.optim.AdamW(m.parameters(), lr=.00001, weight_decay=.01)
                  for m in (reference, batched)]
    for step in range(2):
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        expected_losses = []
        # Original mathematical definition: unpadded, one event at a time,
        # token mean within event, equal weighting of all rows including zero adv.
        for event, advantage in zip(events, advantages):
            loss = sequence_ppo_loss(_single_logp(reference, event['input_ids'],
                event['response_length'], temperature), event['old_logp'], advantage, clip_ratio)
            expected_losses.append(loss.detach())
            (loss / rows).backward()
        with eval_decoder_checkpoint(batched, checkpoint):
            actual_losses = RankerSequenceLoss(batched, temperature, clip_ratio)(
                ids, mask, lengths, old, advantages, response_width)
            assert actual_losses.shape == (rows,)
            torch.testing.assert_close(actual_losses.detach(), torch.stack(expected_losses),
                                        atol=3e-6, rtol=3e-6)
            if rows >= 3:
                assert actual_losses[2] == 0.
            (actual_losses.sum() / rows).backward()
        assert all(not m.training for m in batched.modules())
        parameters = [dict(m.named_parameters()) for m in (reference, batched)]
        assert parameters[0].keys() == parameters[1].keys()
        for name in parameters[0]:
            assert parameters[0][name].grad is not None
            assert parameters[1][name].grad is not None
            torch.testing.assert_close(parameters[0][name].grad, parameters[1][name].grad,
                                        atol=3e-6, rtol=3e-5)
        assert sum(float(p.grad.norm()) for p in parameters[1].values()) > 1e-4
        # Force real clipping, not merely call a no-op clip with a high limit.
        norms = [torch.nn.utils.clip_grad_norm_(m.parameters(), .05)
                 for m in (reference, batched)]
        assert norms[0] > .05
        torch.testing.assert_close(norms[0], norms[1], atol=3e-6, rtol=3e-5)
        for name in parameters[0]:
            torch.testing.assert_close(parameters[0][name].grad, parameters[1][name].grad,
                                        atol=3e-6, rtol=3e-5)
        for optimizer in optimizers:
            optimizer.step()
        for name in parameters[0]:
            torch.testing.assert_close(parameters[0][name], parameters[1][name],
                                        atol=2e-7, rtol=2e-6)
        for left, right in zip(optimizers[0].state.values(), optimizers[1].state.values()):
            assert float(left['step']) == float(right['step']) == step + 1
            for key in ('step', 'exp_avg', 'exp_avg_sq'):
                torch.testing.assert_close(left[key], right[key], atol=2e-7, rtol=3e-5)
    assert sum(float((p.detach()-before[n]).abs().sum())
               for n, p in batched.named_parameters()) > 1e-5


def test_zero_advantage_sequence_has_no_gradient_but_counts_in_expectation():
    model = _model('sdpa')
    events, _ = _examples(model, 3, 1.3)
    ids, mask, lengths, old, response_width = _pack(events)
    losses = RankerSequenceLoss(model, 1.3, .2)(ids, mask, lengths, old,
                                              torch.zeros(3), response_width)
    assert torch.equal(losses, torch.zeros(3))
    losses.sum().backward()
    assert all(p.grad is not None and torch.equal(p.grad, torch.zeros_like(p.grad))
               for p in model.parameters())


def test_terminal_eos_equal_to_pad_is_scored_while_old_logp_padding_is_masked():
    model = _model('eager')
    events, advantages = _examples(model, 3, .7)
    ids, mask, lengths, old, response_width = _pack(events)
    loss = RankerSequenceLoss(model, .7, .2)
    original = loss(ids, mask, lengths, old, advantages, response_width)
    # Deliberately change only old_logp left-padding. Real EOS at the last
    # position remains an action, and changing its score must change the loss.
    padding = torch.arange(response_width)[None] < response_width-lengths[:, None]
    changed = old.clone()
    changed[padding] = -.1
    torch.testing.assert_close(original, loss(ids, mask, lengths, changed,
                                             advantages, response_width), atol=0, rtol=0)
    changed = old.clone()
    changed[0, -1] += .05
    altered = loss(ids, mask, lengths, changed, advantages, response_width)
    assert altered[0] != original[0]
