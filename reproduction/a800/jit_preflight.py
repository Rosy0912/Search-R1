"""Exercise real CUDA Inductor/Triton extension compilation, not only eager inference."""
import torch

x=torch.arange(1024,device='cuda',dtype=torch.float32)/1000
def f(x):return x.sin()+x.cos()
result=torch.compile(f,fullgraph=True)(x)
torch.cuda.synchronize()
assert torch.isfinite(result).all()
assert torch.allclose(result,f(x),rtol=1e-5,atol=1e-5)
print('ACTUAL_TORCH_TRITON_CUDA_JIT_OK',torch.__version__,flush=True)
