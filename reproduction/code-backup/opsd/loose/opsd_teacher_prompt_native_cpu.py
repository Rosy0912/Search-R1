import json, hashlib, os
from pathlib import Path
import torch
from transformers import AutoTokenizer
__package__='search_r1.baselines'
def teacher_inputs(tokenizer, prompts, responses, response_attention, targets=None, max_length=None,
                   prompt_mode='legacy'):
    """Add private answers to teacher context, then identical student response.

    Return absolute prediction positions for every response target. Observations
    remain in response and attention, but the caller excludes them from loss.
    """
    from .rollout import padded
    if prompt_mode not in ('legacy', 'user_context'):
        raise ValueError('Unknown teacher prompt mode')
    pad = tokenizer.pad_token_id
    suffix_ids = None
    if targets is not None and prompt_mode == 'user_context':
        # Explicit Qwen ChatML adapter: preserve original prompt token IDs and
        # insert privileged text inside its last user turn, before its closure.
        if not getattr(tokenizer, 'chat_template', None):
            raise ValueError('Structured teacher context requires a chat template')
        marker = 'OPSD_TEMPLATE_CONTENT_SENTINEL_71c8'
        messages = [{'role': 'user', 'content': marker}]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        without_generation = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        if rendered.count(marker) != 1 or not rendered.startswith(without_generation):
            raise ValueError('Unsupported teacher chat template layout')
        generation = rendered[len(without_generation):]
        suffix = rendered.split(marker)[1]
        if generation != '<|im_start|>assistant\n' or suffix != '<|im_end|>\n' + generation:
            raise ValueError('Structured teacher context currently requires Qwen ChatML boundaries')
        suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
        if not suffix_ids:
            raise ValueError('Empty teacher chat suffix')
    prefixes, rows = [], []
    for i, prompt in enumerate(prompts):
        prefix = prompt[prompt != pad].tolist()
        if targets is not None:
            answers = targets[i]
            if isinstance(answers, str):
                answers = [answers]
            hint = '\n[Teacher-only training context: accepted answers: ' + repr(list(answers)) + ']\n'
            if suffix_ids is not None:
                if len(prefix) < len(suffix_ids) or prefix[-len(suffix_ids):] != suffix_ids:
                    raise ValueError('Student prompt does not end with the verified chat suffix')
                hint += ('Use the reference answers to understand the question, then reason independently. '
                         'Continue following the requested <think>, <search>, and <answer> protocol. '
                         'Do not quote the answer list or these teacher-only instructions.\n')
                hint_ids = tokenizer.encode(hint, add_special_tokens=False)
                if set(hint_ids).intersection(tokenizer.all_special_ids):
                    raise ValueError('Teacher hint contains chat control or special tokens')
                prefix = prefix[:-len(suffix_ids)] + hint_ids + suffix_ids
            else:
                prefix += tokenizer.encode(hint, add_special_tokens=False)
        prefixes.append(len(prefix))
        response = responses[i][response_attention[i].bool()].tolist()
        row = prefix + response
        if max_length is not None and len(row) > max_length:
            raise ValueError('Teacher context exceeds model limit; no silent truncation')
        rows.append(row)
    ids = padded(rows, pad)
    attention = torch.zeros_like(ids)
    for i, row in enumerate(rows):
        attention[i, :len(row)] = 1
    positions = torch.tensor(prefixes)[:, None] + response_attention.long().cpu().cumsum(-1) - 2
    return ids, attention, positions
FUNCTION_SOURCE='def teacher_inputs(tokenizer, prompts, responses, response_attention, targets=None, max_length=None,\n                   prompt_mode=\'legacy\'):\n    """Add private answers to teacher context, then identical student response.\n\n    Return absolute prediction positions for every response target. Observations\n    remain in response and attention, but the caller excludes them from loss.\n    """\n    from .rollout import padded\n    if prompt_mode not in (\'legacy\', \'user_context\'):\n        raise ValueError(\'Unknown teacher prompt mode\')\n    pad = tokenizer.pad_token_id\n    suffix_ids = None\n    if targets is not None and prompt_mode == \'user_context\':\n        # Explicit Qwen ChatML adapter: preserve original prompt token IDs and\n        # insert privileged text inside its last user turn, before its closure.\n        if not getattr(tokenizer, \'chat_template\', None):\n            raise ValueError(\'Structured teacher context requires a chat template\')\n        marker = \'OPSD_TEMPLATE_CONTENT_SENTINEL_71c8\'\n        messages = [{\'role\': \'user\', \'content\': marker}]\n        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)\n        without_generation = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)\n        if rendered.count(marker) != 1 or not rendered.startswith(without_generation):\n            raise ValueError(\'Unsupported teacher chat template layout\')\n        generation = rendered[len(without_generation):]\n        suffix = rendered.split(marker)[1]\n        if generation != \'<|im_start|>assistant\\n\' or suffix != \'<|im_end|>\\n\' + generation:\n            raise ValueError(\'Structured teacher context currently requires Qwen ChatML boundaries\')\n        suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)\n        if not suffix_ids:\n            raise ValueError(\'Empty teacher chat suffix\')\n    prefixes, rows = [], []\n    for i, prompt in enumerate(prompts):\n        prefix = prompt[prompt != pad].tolist()\n        if targets is not None:\n            answers = targets[i]\n            if isinstance(answers, str):\n                answers = [answers]\n            hint = \'\\n[Teacher-only training context: accepted answers: \' + repr(list(answers)) + \']\\n\'\n            if suffix_ids is not None:\n                if len(prefix) < len(suffix_ids) or prefix[-len(suffix_ids):] != suffix_ids:\n                    raise ValueError(\'Student prompt does not end with the verified chat suffix\')\n                hint += (\'Use the reference answers to understand the question, then reason independently. \'\n                         \'Continue following the requested <think>, <search>, and <answer> protocol. \'\n                         \'Do not quote the answer list or these teacher-only instructions.\\n\')\n                hint_ids = tokenizer.encode(hint, add_special_tokens=False)\n                if set(hint_ids).intersection(tokenizer.all_special_ids):\n                    raise ValueError(\'Teacher hint contains chat control or special tokens\')\n                prefix = prefix[:-len(suffix_ids)] + hint_ids + suffix_ids\n            else:\n                prefix += tokenizer.encode(hint, add_special_tokens=False)\n        prefixes.append(len(prefix))\n        response = responses[i][response_attention[i].bool()].tolist()\n        row = prefix + response\n        if max_length is not None and len(row) > max_length:\n            raise ValueError(\'Teacher context exceeds model limit; no silent truncation\')\n        rows.append(row)\n    ids = padded(rows, pad)\n    attention = torch.zeros_like(ids)\n    for i, row in enumerate(rows):\n        attention[i, :len(row)] = 1\n    positions = torch.tensor(prefixes)[:, None] + response_attention.long().cpu().cumsum(-1) - 2\n    return ids, attention, positions'
asset=Path('/mnt/data/shaoshitong/search-r1-rl/opsd/assets-root/assets/qwen2.5-3b')
tok=AutoTokenizer.from_pretrained(asset,local_files_only=True)
question=tok.apply_chat_template([{'role':'user','content':'Question: audit?'}],tokenize=True,add_generation_prompt=True)
prompt=torch.tensor([[tok.pad_token_id]+question])
response=torch.tensor([tok.encode('<think>audit</think><answer>audit answer</answer>',add_special_tokens=False)])
attention=torch.ones_like(response)
original=prompt.clone()
ids,mask,positions=teacher_inputs(tok,prompt,response,attention,[['audit answer']],prompt_mode='user_context')
text=tok.decode(ids[0][mask[0].bool()],skip_special_tokens=False)
hint=text.index('[Teacher-only'); user_start=text.rindex('<|im_start|>user'); user_end=text.index('<|im_end|>',user_start); assistant=text.rindex('<|im_start|>assistant')
assert user_start<hint<user_end<assistant
assert torch.equal(ids[0,positions[0]+1],response[0]) and torch.equal(prompt,original)
try:
 teacher_inputs(tok,prompt,response,attention,[['<|im_end|>']],prompt_mode='user_context')
 raise AssertionError('control token accepted')
except ValueError:
 pass
print(json.dumps({'cpu_only':os.environ.get('CUDA_VISIBLE_DEVICES')=='','model_loaded':False,'candidate_function_sha256':hashlib.sha256(FUNCTION_SOURCE.encode()).hexdigest(),'tokenizer_config_sha256':hashlib.sha256((asset/'tokenizer_config.json').read_bytes()).hexdigest(),'hint_inside_user':True,'response_prediction_positions_match':True,'student_prompt_unchanged':True,'role_token_injection_rejected':True,'teacher_context':text},ensure_ascii=False))
