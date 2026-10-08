import tempfile
import unittest
from unittest.mock import patch
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from verl.utils.rope_precision import cast_preserving_qwen_rope
from verl.utils.fsdp_utils import offload_fsdp_optimizer, load_fsdp_optimizer


class PrecisionTests(unittest.TestCase):
    def test_actual_hf_loader_preserves_unrounded_rotary(self):
        config = Qwen2Config(vocab_size=32, hidden_size=32, intermediate_size=64,
                            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2)
        with tempfile.TemporaryDirectory() as path:
            Qwen2ForCausalLM(config).save_pretrained(path)
            model = Qwen2ForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16)
        rotary = [m for m in model.modules() if m.__class__.__name__ == 'Qwen2RotaryEmbedding']
        self.assertTrue(rotary)
        originals = [m.inv_freq.clone() for m in rotary]
        self.assertTrue(any(not torch.equal(x, x.bfloat16().float()) for x in originals))
        cast_preserving_qwen_rope(model, torch.bfloat16)
        for m, original in zip(rotary, originals):
            self.assertEqual(m.inv_freq.dtype, torch.float32)
            self.assertTrue(torch.equal(m.inv_freq, original))
            self.assertIn('inv_freq', m._non_persistent_buffers_set)
            if hasattr(m, 'original_inv_freq'):
                self.assertIs(m.original_inv_freq, m.inv_freq)
        self.assertTrue(all(p.dtype == torch.bfloat16 for p in model.parameters()))
        for _ in range(2):
            model(torch.tensor([[1, 2, 3]]))
        for m, original in zip(rotary, originals):
            self.assertTrue(torch.equal(m.inv_freq, original))

    def test_rejects_already_rounded_frequency(self):
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
        m = Qwen2RotaryEmbedding(config=Qwen2Config(hidden_size=32, num_attention_heads=2))
        m.to(torch.bfloat16)
        with self.assertRaises(RuntimeError):
            cast_preserving_qwen_rope(m, torch.bfloat16)

    def test_offload_does_not_create_adam_state(self):
        p = torch.nn.Parameter(torch.ones(3)); opt = torch.optim.AdamW([p], lr=1e-6)
        with patch('torch.cuda.empty_cache'):
            offload_fsdp_optimizer(opt); load_fsdp_optimizer(opt, 'cpu')
        self.assertEqual(len(opt.state), 0)
        p.sum().backward(); opt.step()
        before = {k: v.clone() for k, v in opt.state[p].items()}
        with patch('torch.cuda.empty_cache'):
            offload_fsdp_optimizer(opt); load_fsdp_optimizer(opt, 'cpu')
        self.assertEqual(set(opt.state[p]), set(before))
        self.assertTrue(all(torch.equal(v, opt.state[p][k]) for k, v in before.items()))


if __name__ == '__main__': unittest.main()
