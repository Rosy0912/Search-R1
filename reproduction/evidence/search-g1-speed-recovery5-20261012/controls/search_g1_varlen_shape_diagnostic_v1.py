import os,json,pathlib,datetime,time
os.environ['CUDA_VISIBLE_DEVICES']='0'
import torch
from transformers import AutoModelForCausalLM
cp='/mnt/data/shaoshitong/search-g1-mixed-2de313e-20261011/checkpoints/search-g1-common8/actor/global_step_5'
out=pathlib.Path(os.environ['DIAGNOSTIC_RESULT']);assert not out.exists()
torch.backends.cuda.matmul.allow_tf32=False
torch.manual_seed(137)
m=AutoModelForCausalLM.from_pretrained(cp,torch_dtype=torch.float32,attn_implementation='flash_attention_2',local_files_only=True).eval()
for p in m.parameters():p.data=p.data.to(torch.bfloat16)
m=m.cuda()
assert {p.dtype for p in m.parameters()}=={torch.bfloat16}
assert all(b.dtype==torch.float32 for n,b in m.named_buffers() if 'inv_freq' in n)
a=torch.arange(10,17,device='cuda').unsqueeze(0);b=torch.arange(30,41,device='cuda').unsqueeze(0);ids=torch.cat((a,b),1);pos=torch.cat((torch.arange(7,device='cuda'),torch.arange(11,device='cuda'))).unsqueeze(0)
def f(ids,pos,mask=None):
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):y=m(input_ids=ids,position_ids=pos,attention_mask=mask,use_cache=False).logits
 assert y.dtype==torch.bfloat16 and torch.isfinite(y).all()
 return y.float()
p=f(ids,pos);single=torch.cat((f(a,pos[:,:7]),f(b,pos[:,7:])),1)
# Same flattened GEMM shape18 for every forward; mask trailing pads out of attention.
references=[]
for seq in (a,b):
 n=seq.shape[1];padded=torch.nn.functional.pad(seq,(0,18-n),value=0);pp=torch.zeros_like(padded);pp[:,:n]=torch.arange(n,device='cuda');mask=torch.zeros_like(padded);mask[:,:n]=1
 references.append(f(padded,pp,mask)[:,:n])
same=torch.cat(references,1);changed=ids.clone();changed[:,:7]+=50;q=f(changed,pos)
def stats(x,y):
 d=(x-y).abs();xp=x.log_softmax(-1);yp=y.log_softmax(-1);kl=(xp.exp()*(xp-yp)).sum(-1)
 return dict(max=d.max().item(),mean=d.mean().item(),rms=(d.square().mean().sqrt().item()),p99=torch.quantile(d.flatten(),.99).item(),max_probability_difference=(xp.exp()-yp.exp()).abs().max().item(),mean_token_kl=kl.mean().item(),max_token_kl=kl.max().item(),argmax_match=(x.argmax(-1)==y.argmax(-1)).float().mean().item())
r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),checkpoint=cp,optimizer_writes=0,slurm_job_id=os.environ['SLURM_JOB_ID'],different_shape=stats(p,single),equal_shape_masked=stats(p,same),sequence2_isolation=torch.equal(p[:,7:],q[:,7:]),sequence2_max_difference=(p[:,7:]-q[:,7:]).abs().max().item(),max_allocated_bytes=torch.cuda.max_memory_allocated(),scope='Actual saved CP5 BF16 parameter / FP32 rotary forward diagnostic, no optimizer or training; equal-shaped masked reference separates numerical shape effects from packed boundaries')
with out.open('x') as file:json.dump(r,file,indent=2)
print(json.dumps(r),flush=True)
