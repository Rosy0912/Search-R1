"""Compare actual fast-tokenizer semantics across HF serialization formats."""
import json
from pathlib import Path


def validate_tokenizer_identity(original, checkpoint):
    from tokenizers import Tokenizer
    original,checkpoint=Path(original),Path(checkpoint)
    # Loading with the very same target runtime normalizes the older BPE merge
    # strings versus newer pairs and materializes default ignore_merges=False.
    # Compare the ENTIRE normalized backend, not selected vocabulary samples.
    first=json.loads(Tokenizer.from_file(str(original/'tokenizer.json')).to_str())
    second=json.loads(Tokenizer.from_file(str(checkpoint/'tokenizer.json')).to_str())
    assert first==second, 'Saved tokenizer backend semantics differ from original base'
    assert json.loads((original/'tokenizer_config.json').read_text())==json.loads((checkpoint/'tokenizer_config.json').read_text()), 'Saved tokenizer config/chat template differs'
    # Also exercise the actual Transformers loader: auxiliary saved special-token
    # or chat-template files must not override the equivalent JSON configuration.
    from transformers import AutoTokenizer
    a=AutoTokenizer.from_pretrained(original,local_files_only=True)
    b=AutoTokenizer.from_pretrained(checkpoint,local_files_only=True)
    assert type(a) is type(b) and a.is_fast and b.is_fast, 'Tokenizer loader class differs'
    assert json.loads(a.backend_tokenizer.to_str())==json.loads(b.backend_tokenizer.to_str()), 'Loaded tokenizer backend differs'
    for name in ('special_tokens_map','all_special_ids','chat_template','padding_side','truncation_side','model_max_length','model_input_names'):
        assert getattr(a,name)==getattr(b,name), 'Loaded tokenizer setting differs: '+name
    return dict(verified=True,comparison='entire normalized backend and parsed config; actual AutoTokenizer backend/special-token/chat/settings identity; byte hashes retained separately')
