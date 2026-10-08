"""Fixed-production-model inference only, not a QA benchmark or algorithm smoke."""
from vllm import LLM, SamplingParams
model = LLM(model='/dataset/shaoshitong/search-r1-rl/cosearch/assets-root/assets/qwen2.5-3b',
            tensor_parallel_size=1, gpu_memory_utilization=.12,
            max_model_len=1024, enforce_eager=True)
result = model.generate(['The capital of France is'], SamplingParams(temperature=0, max_tokens=16))
assert result[0].outputs[0].token_ids
print('REAL_BASE_3B_VLLM_CUDA_OK', repr(result[0].outputs[0].text), flush=True)
