"""Compare distributed LoRA gradients against the exact single-device global mean."""
import os
import json
from pathlib import Path
import torch
import torch.distributed as dist
from transformers import Qwen2Config, Qwen2ForCausalLM
from peft import LoraConfig, get_peft_model
from search_e1.opsd import pair_loss

rank=int(os.environ['RANK'])
world=int(os.environ['WORLD_SIZE'])
torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
dist.init_process_group('nccl')
torch.manual_seed(42)
base=Qwen2ForCausalLM(Qwen2Config(vocab_size=128,hidden_size=64,intermediate_size=128,
                                num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2)).cuda()
model=get_peft_model(base,LoraConfig(r=4,lora_alpha=8,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'))
parameters=[p for p in model.parameters() if p.requires_grad]
examples=[([[1,2,3],[1,2,3,4,5]], [8,9,10,11], [1,0,0,1]),
          ([[1,2],[1,2,4]], [12,13,14], [1,1,1]),
          ([[2,3],[2,3,9,8]], [22,23], [1,1])]
# Three samples on four ranks exercises an empty rank in the last global batch.
for example in examples:
    (pair_loss(model,example,10.,2)/len(examples)).backward()
expected=[p.grad.detach().clone() for p in parameters]
model.zero_grad(set_to_none=True)
for example in examples[rank::world]:
    (pair_loss(model,example,10.,1)/len(examples)).backward()
for p,target in zip(parameters,expected):
    if p.grad is None:
        p.grad=torch.zeros_like(p)
    dist.all_reduce(p.grad)
    torch.testing.assert_close(p.grad,target,rtol=1e-3,atol=1e-6)
if rank==0:
    result = {'distributed_gradient':'passed','world_size':world,'global_samples':len(examples)}
    Path('runs/distributed_smoke.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result,flush=True)
dist.destroy_process_group()
