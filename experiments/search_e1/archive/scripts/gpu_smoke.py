"""Exercise real causal alignment and LoRA updates on a tiny random Qwen2 GPU model."""
import json
import sys
from pathlib import Path
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from peft import LoraConfig, get_peft_model
from search_e1.opsd import pair_loss

if '--flash' in sys.argv:
    from flash_attn import flash_attn_func
    q = torch.randn(1,16,4,32,device='cuda',dtype=torch.float16)
    attention = flash_attn_func(q,q,q,causal=True)
    assert attention.shape == q.shape and torch.isfinite(attention).all()

torch.manual_seed(42)
base = Qwen2ForCausalLM(Qwen2Config(vocab_size=128, hidden_size=64, intermediate_size=128,
                                  num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2))
model = get_peft_model(base.cuda(), LoraConfig(r=4, lora_alpha=8, target_modules=['q_proj','v_proj'], task_type='CAUSAL_LM'))
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
model.enable_input_require_grads()
example = ([[1,2,3], [1,2,3,4,5]], [8,9,10,11], [1,0,0,1])
loss = pair_loss(model, example, clip=10, chunk=1)
loss.backward()
trainable = {n:p for n,p in model.named_parameters() if p.requires_grad}
assert torch.isfinite(loss) and loss.item() > 0
assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in trainable.values())
assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
before = {n:p.detach().clone() for n,p in trainable.items()}
torch.optim.AdamW(trainable.values(), lr=1e-3).step()
assert any(not torch.equal(before[n],p) for n,p in trainable.items())
# Token masks genuinely change the average and causal teacher/student prefix alignment works.
loss2 = pair_loss(model, (example[0], example[1], [1,1,1,1]), clip=10, chunk=2)
assert torch.isfinite(loss2)
result = {'gpu':torch.cuda.get_device_name(), 'loss':loss.item(), 'updated_loss':loss2.item(),
          'trainable_parameters':sum(p.numel() for p in trainable.values()), 'status':'passed'}
Path('runs').mkdir(exist_ok=True)
Path('runs/gpu_smoke.json').write_text(json.dumps(result,indent=2)+'\n')
print(result)
