import json,os,torch
from flash_attn import flash_attn_func
q=torch.randn(2,32,4,32,device='cuda',dtype=torch.bfloat16,requires_grad=True)
y=flash_attn_func(q,q,q,causal=True);y.float().square().mean().backward()
assert torch.isfinite(y).all() and torch.isfinite(q.grad).all()
print('REAL_FLASH_BF16_FORWARD_BACKWARD_OK',flush=True)
from vllm import LLM,SamplingParams
model=LLM(model=os.environ['OPSD_TEACHER'],tensor_parallel_size=1,gpu_memory_utilization=.12,max_model_len=1024,enforce_eager=True)
out=model.generate(['The capital of France is'],SamplingParams(temperature=0,max_tokens=16))
assert out[0].outputs[0].token_ids
print('REAL_3B_VLLM_CUDA_OK',repr(out[0].outputs[0].text),flush=True)
