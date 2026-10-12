"""A short real-model CUDA inference check, not a QA benchmark."""
from vllm import LLM, SamplingParams
import importlib.metadata

model = LLM(model='/data/search-r1/assets/qwen2.5-3b',
            tensor_parallel_size=1, gpu_memory_utilization=0.12,
            max_model_len=1024, enforce_eager=True)
out = model.generate(['The capital of France is'], SamplingParams(temperature=0,max_tokens=16))
assert out[0].outputs[0].token_ids
print('REAL_MODEL_VLLM_CUDA_OK', importlib.metadata.version('vllm'), repr(out[0].outputs[0].text))
