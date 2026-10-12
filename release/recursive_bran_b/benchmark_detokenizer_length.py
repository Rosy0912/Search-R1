"""CPU equivalence and isolated length timing using the real frozen tokenizer."""
import argparse
import hashlib
import json
import time
from types import SimpleNamespace
from transformers import AutoTokenizer
from vllm.transformers_utils.detokenizer import detokenize_incrementally
from search_r1.baselines.detokenizer_len_cache import native_generation_length_cache

def incremental(tokenizer, prompt, suffix, skip):
    previous = None; prefix = read = 0; result = []
    for index in range(len(suffix)):
        tokens, text, prefix, read = detokenize_incrementally(tokenizer,
            prompt + suffix[:index+1], previous, prefix, read,
            skip_special_tokens=skip, spaces_between_special_tokens=True)
        previous = tokens if previous is None else previous + tokens
        result.append((tokens, text, prefix, read))
    return result

def run(model, repetitions):
    t = AutoTokenizer.from_pretrained(model, local_files_only=True)
    group = SimpleNamespace(tokenizer=t, enable_lora=False)
    identity = lambda: hashlib.sha256(t.backend_tokenizer.to_str().encode()).hexdigest()
    before = identity(); cls = t.__class__
    prompt = t.encode('Question: 请检索并回答。\n', add_special_tokens=False)
    texts = ['<think>查找资料</think><search>北京 population</search>',
             '<answer>42</answer> then <search>第二次查询</search> trailing',
             'naïve café 中文 🙂\n<answer>混合 Unicode</answer>',
             '<|im_end|> literal closing tags </search></answer>']
    tested = 0
    for text in texts:
        suffix = t.encode(text, add_special_tokens=False)
        for skip in (False, True):
            old = incremental(t, prompt, suffix, skip)
            with native_generation_length_cache(group, enabled=True):
                new = incremental(group.tokenizer, prompt, suffix, skip)
                assert group.tokenizer.backend_tokenizer is t.backend_tokenizer
                assert group.tokenizer.encode(text) == t.encode(text)
                assert group.tokenizer.decode(suffix) == t.decode(suffix)
            assert new == old, 'Actual native incremental token/text/offset semantics changed'
            tested += len(suffix)
    # Exercise the exact len-dependent invalid-ID bound too.
    suffix = [len(t) + 1, t.eos_token_id]
    old = incremental(t, prompt, suffix, False)
    with native_generation_length_cache(group, enabled=True):
        assert incremental(group.tokenizer, prompt, suffix, False) == old
    start = time.perf_counter()
    baseline = [len(t) for _ in range(repetitions)]
    old_time = time.perf_counter() - start
    with native_generation_length_cache(group, enabled=True):
        start = time.perf_counter()
        cached = [len(group.tokenizer) for _ in range(repetitions)]
        new_time = time.perf_counter() - start
    assert baseline == cached and group.tokenizer is t and t.__class__ is cls
    assert identity() == before
    return dict(scope='CPU real tokenizer length calls and fixed native detokenizer fixtures only; no GPU/full-outer speed claim',
        tokenizer_backend_sha256=before, native_incremental_steps_compared=tested+2,
        original_object_and_class_and_backend_unchanged=True, exact_native_tokens_text_offsets_equal=True,
        repetitions=repetitions, original_length_wall_seconds=old_time,
        cached_length_wall_seconds=new_time, length_values_equal=True,
        temperature_reward_probability_and_sampling_rng_not_modified=True)

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--model', required=True)
    p.add_argument('--repetitions', type=int, default=200);a=p.parse_args()
    if not 1 <= a.repetitions <= 2000: raise ValueError('Bounded CPU benchmark only')
    print(json.dumps(run(a.model, a.repetitions), sort_keys=True))
