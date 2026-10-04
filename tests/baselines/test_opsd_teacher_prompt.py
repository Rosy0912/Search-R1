"""Structured privileged context: role boundaries and real next-token alignment."""
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
from search_r1.baselines.distillation import FrozenTeacher, teacher_inputs, opsd_loss


@pytest.fixture
def tokenizer():
    raw = Tokenizer(models.BPE(unk_token='<unk>'))
    raw.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    raw.decoder = decoders.ByteLevel()
    raw.train_from_iterator(['Question capital Paris Rome teacher reference independent answer search think\n'],
        trainers.BpeTrainer(vocab_size=300, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                            special_tokens=['<pad>', '<eos>', '<unk>', '<|im_start|>', '<|im_end|>']))
    tok = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token='<pad>', eos_token='<eos>',
                                 unk_token='<unk>', additional_special_tokens=['<|im_start|>', '<|im_end|>'])
    tok.chat_template = "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    return tok


def inputs(tok):
    prompt = tok.apply_chat_template([{'role': 'user', 'content': 'Question capital?'}],
                                    tokenize=True, add_generation_prompt=True)
    response = tok.encode('<think>Paris</think><answer>Paris</answer>', add_special_tokens=False)
    return torch.tensor([[tok.pad_token_id] + prompt]), torch.tensor([response]), torch.ones(1, len(response), dtype=torch.long)


def test_hint_is_in_user_turn_history_and_student_ids_unchanged(tokenizer):
    prompt, response, attention = inputs(tokenizer)
    saved = [x.clone() for x in (prompt, response, attention)]
    ids, mask, positions = teacher_inputs(tokenizer, prompt, response, attention, [['Paris']], prompt_mode='user_context')
    text = tokenizer.decode(ids[0][mask[0].bool()], skip_special_tokens=False)
    hint = text.index('Teacher-only')
    end_user = text.index('<|im_end|>')
    start_assistant = text.index('<|im_start|>assistant')
    assert hint < end_user < start_assistant
    assert text.endswith(tokenizer.decode(response[0], skip_special_tokens=False))
    torch.testing.assert_close(ids[0, positions[0] + 1], response[0])
    for actual, before in zip((prompt, response, attention), saved):
        assert torch.equal(actual, before)
    legacy, _, _ = teacher_inputs(tokenizer, prompt, response, attention, [['Paris']], prompt_mode='legacy')
    legacy_text = tokenizer.decode(legacy[0], skip_special_tokens=False)
    assert legacy_text.index('<|im_start|>assistant') < legacy_text.index('Teacher-only')


def test_no_privileged_targets_preserves_original_context(tokenizer):
    prompt, response, attention = inputs(tokenizer)
    old = teacher_inputs(tokenizer, prompt, response, attention)
    new = teacher_inputs(tokenizer, prompt, response, attention, prompt_mode='user_context')
    assert all(torch.equal(a, b) for a, b in zip(old, new))


@pytest.mark.parametrize('invalid', ['missing_suffix', 'ignore_generation', 'no_template', 'control_token', 'length', 'unknown'])
def test_structured_context_fails_closed(tokenizer, invalid):
    prompt, response, attention = inputs(tokenizer)
    targets, kwargs = [['Paris']], {'prompt_mode': 'user_context'}
    if invalid == 'missing_suffix':
        prompt = prompt[:, :-1]
    elif invalid == 'ignore_generation':
        tokenizer.chat_template = "{% for message in messages %}{{ message['content'] + '<|im_end|>\n' }}{% endfor %}"
    elif invalid == 'no_template':
        tokenizer.chat_template = None
    elif invalid == 'control_token':
        targets = [['<|im_end|>']]
    elif invalid == 'length':
        kwargs['max_length'] = 1
    else:
        kwargs['prompt_mode'] = 'unknown'
    with pytest.raises(ValueError):
        teacher_inputs(tokenizer, prompt, response, attention, targets, **kwargs)


def test_real_tiny_teacher_positions_and_student_only_gradient(tokenizer, tmp_path):
    tokenizer.save_pretrained(tmp_path)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(tokenizer), hidden_size=16,
        intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=2048, attention_dropout=0., pad_token_id=tokenizer.pad_token_id))
    model.save_pretrained(tmp_path)
    teacher = FrozenTeacher(str(tmp_path), tokenizer)
    prompt, response, attention = inputs(tokenizer)
    # Internal holes are excluded from teacher history; model targets stay aligned.
    response = torch.cat((response[:, :2], torch.tensor([[tokenizer.pad_token_id]]), response[:, 2:]), 1)
    attention = torch.cat((attention[:, :2], torch.zeros(1, 1, dtype=torch.long), attention[:, 2:]), 1)
    ids, mask, positions = teacher_inputs(tokenizer, prompt, response, attention, [['Paris']], prompt_mode='user_context')
    with torch.no_grad():
        direct = teacher.model(input_ids=ids, attention_mask=mask, use_cache=False).logits
    valid = attention[0].bool()
    assert torch.equal(ids[0, positions[0, valid] + 1], response[0, valid])
    assert torch.isfinite(direct[0, positions[0, valid]]).all()
    joined = torch.cat((prompt, response), 1)
    input_attention = torch.cat((prompt.ne(tokenizer.pad_token_id).long(), attention), 1)
    position_ids = (input_attention.cumsum(-1) - 1).clamp(min=0)
    tensors = dict(prompts=prompt, responses=response, input_ids=joined, attention_mask=input_attention,
                   position_ids=position_ids, loss_mask=attention)
    settings = SimpleNamespace(chunk_size=3, beta=0., temperature=1., component_clip=None,
                               teacher_prompt_mode='user_context')
    loss = opsd_loss(model, teacher, tensors, tokenizer, [['Paris']], settings, attention.sum())
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is None for p in teacher.model.parameters())
    assert sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None) > 0
