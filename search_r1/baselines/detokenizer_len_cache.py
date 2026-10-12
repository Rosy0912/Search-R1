"""Cache only the unchanged HF vocabulary length within one native generation.

The original tokenizer object, backend, decode/encode methods and special-token
properties remain untouched. LoRA and non-fast tokenizers are out of scope.
"""
from contextlib import contextmanager
import os
import sys

class LengthView:
    def __init__(self, original):
        self.original = original
        self.length = len(original)

    def __len__(self):
        return self.length

    def __getattr__(self, name):
        return getattr(self.original, name)

@contextmanager
def native_generation_length_cache(group, enabled=None):
    if enabled is None:
        enabled = os.environ.get('RECURSIVE_B_CACHE_DETOKENIZER_LENGTH') == '1'
    if not enabled:
        yield
        return
    if group.enable_lora or not getattr(group.tokenizer, 'is_fast', False):
        raise ValueError('Length cache requires the fixed no-LoRA fast tokenizer')
    original = group.tokenizer
    if isinstance(original, LengthView):
        raise ValueError('Concurrent/nested tokenizer length view is forbidden')
    view = LengthView(original)
    group.tokenizer = view
    try:
        yield
    finally:
        error = sys.exc_info()[1]
        group.tokenizer = original
        if len(original) != view.length:
            if error is not None:
                # Preserve the real programming failure, never turn it into a skip.
                if hasattr(error, 'add_note'):
                    error.add_note('Tokenizer vocabulary length also changed during generation')
            else:
                raise ValueError('Tokenizer vocabulary changed during native generation')
