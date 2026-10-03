"""CPU mathematical checks for unscaled static Qwen beyond metadata length.

Not 3B resource/quality evidence or a joint-RL acceptance update.
"""
import copy

import pytest
import torch
import torch.nn.functional as F
from transformers import GenerationConfig, Qwen2Config, Qwen2ForCausalLM

from search_r1.baselines.cosearch import (RankerSequenceLoss, check_candidate_context,
    eval_decoder_checkpoint, sequence_ppo_loss, unpack_generation)


@pytest.mark.parametrize('kernel', ['eager', 'sdpa'])
def test_beyond_config_keeps_rope_and_full_sequence_ppo_gradient(kernel):
    torch.manual_seed(44)
    torch.set_num_threads(1)
    config = Qwen2Config(vocab_size=43, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=32, attention_dropout=0., eos_token_id=None,
        pad_token_id=0, attn_implementation=kernel)
    model = Qwen2ForCausalLM(config).eval()
    original_config = copy.deepcopy(model.config.to_dict())
    frequency = model.model.rotary_emb.inv_freq.detach().clone()
    prompts = [(torch.arange(n).remainder(40)+2) for n in (48, 39)]
    ids = torch.stack([F.pad(p, (48-p.numel(), 0)) for p in prompts])
    mask = torch.stack([F.pad(torch.ones_like(p), (48-p.numel(), 0)) for p in prompts])
    with torch.no_grad():
        output = model.generate(ids, attention_mask=mask,
            generation_config=GenerationConfig(max_new_tokens=4, do_sample=False,
                eos_token_id=None, pad_token_id=0, return_dict_in_generate=True,
                output_logits=True))
    events = unpack_generation(output, ids, mask, None, 1.)
    assert [e['input_ids'].numel() for e in events] == [52, 43]
    reference, batched = copy.deepcopy(model), copy.deepcopy(model)
    advantages = torch.tensor([.7, -.4])
    losses = []
    for e, advantage in zip(events, advantages):
        logits = reference(e['input_ids'][None], use_cache=False,
                           num_logits_to_keep=5).logits[:, :-1]
        logp = logits.log_softmax(-1).gather(-1, e['input_ids'][-4:][None, :, None]).squeeze(-1)[0]
        torch.testing.assert_close(logp.detach(), e['old_logp'], atol=2e-6, rtol=2e-6)
        loss = sequence_ppo_loss(logp, e['old_logp']+.03, advantage, .2)
        losses.append(loss.detach())
        (loss / 2).backward()
    width = 52
    full_ids = torch.stack([F.pad(e['input_ids'], (width-e['input_ids'].numel(), 0)) for e in events])
    full_mask = torch.stack([F.pad(torch.ones_like(e['input_ids']),
        (width-e['input_ids'].numel(), 0)) for e in events])
    old = torch.stack([e['old_logp']+.03 for e in events])
    with eval_decoder_checkpoint(batched, True):
        batch_losses = RankerSequenceLoss(batched, 1., .2)(full_ids, full_mask,
            torch.tensor([4, 4]), old, advantages, 4)
        torch.testing.assert_close(batch_losses.detach(), torch.stack(losses), atol=2e-6, rtol=2e-6)
        batch_losses.mean().backward()
    for left, right in zip(reference.parameters(), batched.parameters()):
        assert left.grad is not None and right.grad is not None
        assert right.grad.isfinite().all()
        torch.testing.assert_close(left.grad, right.grad, atol=2e-6, rtol=2e-5)
    assert sum(float(p.grad.norm()) for p in batched.parameters()) > 1e-4
    for candidate in (model, reference, batched):
        assert candidate.config.to_dict() == original_config
        assert candidate.model.rotary_emb.rope_type == 'default'
        assert torch.equal(candidate.model.rotary_emb.inv_freq, frequency)


def test_full_context_optin_preserves_complete_oversized_input(tmp_path, monkeypatch, capsys):
    import json
    monkeypatch.setenv('SEARCH_R1_BASELINE_OUTPUT', str(tmp_path))
    ids = list(range(48))
    docs = [dict(document=dict(contents='full candidate ' + str(i))) for i in range(50)]
    check_candidate_context([ids], 4, 32, ['full question'], ['query'], [docs], True)
    record = json.loads(next((tmp_path/'ranker-context-failures').glob('*.json')).read_text())
    assert record['cases'][0]['input_ids'] == ids
    assert record['cases'][0]['documents'] == docs
    assert len(ids) == 48 and len(docs) == 50
    assert 'COSEARCH_FULL_CONTEXT_EXTRAPOLATION_JSON=' in capsys.readouterr().out
    with pytest.raises(ValueError, match='context exceeds model limit'):
        check_candidate_context([ids], 4, 32, ['full question'], ['query'], [docs])
