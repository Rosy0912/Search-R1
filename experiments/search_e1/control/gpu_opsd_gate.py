"""Real four-rank OPSD memory/update gate; synthetic inputs, never scientific EM."""
import os,json,random
from pathlib import Path
from datetime import timedelta
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import LoraConfig,get_peft_model
from search_e1.opsd import pair_loss,_opsd_collective_reject
base=Path(os.environ['E1_BASE']);profile=os.environ['E1_PROFILE']
rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
assert world==4
random.seed(42);torch.manual_seed(42);torch.cuda.set_device(local)
dist.init_process_group('nccl',timeout=timedelta(hours=3))
model_path=str(base/'assets/model')
tokenizer=AutoTokenizer.from_pretrained(model_path,use_fast=True)
base_model=AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda()
model=get_peft_model(base_model,LoraConfig(r=16,lora_alpha=32,lora_dropout=0.,target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],task_type='CAUSAL_LM'))
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads()
trainable=[p for p in model.parameters() if p.requires_grad]
optimizer=torch.optim.AdamW(trainable,lr=1e-5)
# Full legal context8192, longer masked response than the rollout policy budget.
# Teacher/student share aligned response but use different4096-token prefixes.
rng=random.Random(42+rank)
ids=lambda n:[rng.randrange(10,10000) for _ in range(n)]
encoded=([ids(4096),ids(4096)],ids(4096),[1]*4096)
tracked=next(p for name,p in model.named_parameters() if 'lora_B' in name);initial=tracked.detach().cpu().clone()
torch.cuda.reset_peak_memory_stats();optimizer.zero_grad(set_to_none=True)
losses=[];invalid=False
for i in range(8):
 loss=pair_loss(model,encoded,10.,32)
 invalid=invalid or not torch.isfinite(loss).all().item()
 if torch.isfinite(loss).all().item():(loss/32).backward()
 losses.append(float(loss.detach()));print(json.dumps(dict(rank=rank,local_pair=i+1,loss=losses[-1])),flush=True)
_opsd_collective_reject(invalid,world,'OPSD gate nonfinite loss')
for p in trainable:
 if p.grad is None:p.grad=torch.zeros_like(p)
 dist.all_reduce(p.grad,op=dist.ReduceOp.SUM)
grad_norm=torch.nn.utils.clip_grad_norm_(trainable,1.)
_opsd_collective_reject(not torch.isfinite(grad_norm).all().item(),world,'OPSD gate nonfinite gradient')
optimizer.step();torch.cuda.synchronize()
changed=not torch.equal(initial,tracked.detach().cpu());_opsd_collective_reject(not changed,world,'OPSD gate weight did not change')
update_peak=torch.cuda.max_memory_reserved()
# Preserve optimizer residency when probing worst-case native HF generation.
model.eval();prefix=torch.tensor([ids(8192-500)],device='cuda');attention=torch.ones_like(prefix)
with torch.no_grad():output=model.generate(input_ids=prefix,attention_mask=attention,min_new_tokens=500,max_new_tokens=500,do_sample=False,pad_token_id=tokenizer.eos_token_id)
torch.cuda.synchronize();assert output.shape[1]==8192
peak=torch.cuda.max_memory_reserved();total=torch.cuda.get_device_properties(local).total_memory
_opsd_collective_reject(total-peak<6*1024**3,world,'OPSD gate insufficient6GiB allocator headroom')
report=dict(rank=rank,world_size=world,global_pairs=32,local_pairs=8,losses=losses,grad_norm=float(grad_norm),changed=changed,update_peak_reserved_bytes=update_peak,generation_peak_reserved_bytes=peak,total_memory_bytes=total,synthetic_gate=True,context=8192,response=4096,generation_new_tokens=500,clip=10.,chunk=32,lr=1e-5)
rows=[None]*world;dist.all_gather_object(rows,report)
if rank==0:
 (base/'receipts'/f'{profile}-opsd-memory-update-gate.json').write_text(json.dumps(dict(status='ACCEPTED',ranks=rows,scope='Memory and actual pair_loss/AdamW update, no reward or EM inference'),indent=2))
dist.destroy_process_group()
