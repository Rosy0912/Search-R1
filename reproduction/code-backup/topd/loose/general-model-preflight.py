"""Offline real base-model inference and flash kernel check, independent of TOP-D."""
import os
import torch
from flash_attn import flash_attn_func
from vllm import LLM, SamplingParams

torch.manual_seed(42)
q = torch.randn(2, 128, 4, 64, device='cuda', dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn_like(q)
actual = flash_attn_func(q, k, v, causal=True)
reference = torch.nn.functional.scaled_dot_product_attention(
    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True).transpose(1, 2)
assert torch.isfinite(actual).all()
torch.testing.assert_close(actual, reference, atol=.03, rtol=.03)
print('FLASH_ATTN_BF16_REFERENCE_PASS', flush=True)
del q, k, v, actual, reference
torch.cuda.empty_cache()
root = os.environ['AGENT_ROOT']
model = LLM(model=root+'/assets-root/assets/qwen2.5-3b', tensor_parallel_size=1,
            gpu_memory_utilization=.12, max_model_len=1024, enforce_eager=True)
out = model.generate(['The capital of France is'], SamplingParams(temperature=0, max_tokens=16))
assert out[0].outputs[0].token_ids
print('REAL_BASE_3B_VLLM_INFERENCE_PASS', repr(out[0].outputs[0].text), flush=True)
