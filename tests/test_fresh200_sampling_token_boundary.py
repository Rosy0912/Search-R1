"""Official byte-bound Qwen tokenizer counterexamples; not actual3306 tokens."""
import hashlib
import os
from pathlib import Path
import pytest
from tokenizers import Tokenizer
from search_r1.llm_agent.sampling_trace import prefix

SHA='c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539'

@pytest.fixture(scope='module')
def tokenizer():
    path=Path(os.environ['FRESH200_OFFICIAL_TOKENIZER_JSON'])
    assert hashlib.sha256(path.read_bytes()).hexdigest()==SHA
    return Tokenizer.from_file(str(path))

@pytest.mark.parametrize('tag',['search','answer'])
@pytest.mark.parametrize('suffix',['x','>','<answer>','\n','中文'])
def test_real_official_token_boundary_is_whole_raw_prefix(tokenizer,tag,suffix):
    action=f'<{tag}>x</{tag}>';full=action+suffix+' trailing'
    raw=tokenizer.encode(full,add_special_tokens=False).ids
    decode=lambda ids:tokenizer.decode(ids,skip_special_tokens=False)
    n,text=prefix(raw,decode)
    assert text==action and decode(raw[:n]).startswith(action)
    assert all(not decode(raw[:k]).startswith(action) for k in range(1,n))
    assert raw[:n]+raw[n:]==raw # prefix returns count; no token edits.
    if suffix in ('x','>','<answer>'):
        assert decode(raw[:n])!=text # authentic BPE token spans past closing tag.
    if suffix=='x':assert raw[n-1]==28257 # official token decodes '>x'.


def test_real_no_closed_action_and_embedded_special_unchanged(tokenizer):
    full='<search>中文 <|im_end|> unfinished'
    raw=tokenizer.encode(full,add_special_tokens=False).ids
    assert prefix(raw,lambda ids:tokenizer.decode(ids,skip_special_tokens=False))==(len(raw),full)
