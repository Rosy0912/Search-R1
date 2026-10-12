"""Length-only proxy lifetime and exact delegation; no GPU speed claim."""
import unittest
from types import SimpleNamespace
from search_r1.baselines.detokenizer_len_cache import native_generation_length_cache, LengthView

class Tokenizer:
    is_fast = True
    def __init__(self):
        self.size = 100; self.calls = 0; self.backend_tokenizer = object()
        self.all_special_ids = [0, 1]
    def __len__(self):
        self.calls += 1
        return self.size
    def decode(self, ids):
        return str(ids)

class Check(unittest.TestCase):
    def make(self):
        t = Tokenizer()
        return t, SimpleNamespace(tokenizer=t, enable_lora=False)

    def test_only_two_original_length_calls_and_backend_methods_unchanged(self):
        t, g = self.make()
        before_class = t.__class__
        with native_generation_length_cache(g, enabled=True):
            self.assertIsInstance(g.tokenizer, LengthView)
            self.assertEqual([len(g.tokenizer) for _ in range(100)], [100]*100)
            self.assertIs(g.tokenizer.backend_tokenizer, t.backend_tokenizer)
            self.assertIs(g.tokenizer.decode.__self__, t)
            self.assertEqual(g.tokenizer.decode([1, 2]), t.decode([1, 2]))
            self.assertEqual(g.tokenizer.all_special_ids, t.all_special_ids)
        self.assertIs(g.tokenizer, t); self.assertIs(t.__class__, before_class)
        self.assertEqual(t.calls, 2)

    def test_drift_rejected_and_original_restored(self):
        t, g = self.make()
        with self.assertRaisesRegex(ValueError, 'vocabulary changed'):
            with native_generation_length_cache(g, enabled=True):
                t.size += 1
        self.assertIs(g.tokenizer, t)

    def test_programming_failure_preserved_and_original_restored(self):
        t, g = self.make()
        with self.assertRaisesRegex(RuntimeError, 'real fault'):
            with native_generation_length_cache(g, enabled=True):
                t.size += 1
                raise RuntimeError('real fault')
        self.assertIs(g.tokenizer, t)

    def test_disabled_has_original_dynamic_length(self):
        t, g = self.make()
        with native_generation_length_cache(g, enabled=False):
            self.assertIs(g.tokenizer, t)
            t.size += 1
            self.assertEqual(len(g.tokenizer), 101)

    def test_lora_refused_before_group_change(self):
        t, g = self.make(); g.enable_lora = True
        with self.assertRaisesRegex(ValueError, 'no-LoRA'):
            with native_generation_length_cache(g, enabled=True): pass
        self.assertIs(g.tokenizer, t)

if __name__ == '__main__': unittest.main()
