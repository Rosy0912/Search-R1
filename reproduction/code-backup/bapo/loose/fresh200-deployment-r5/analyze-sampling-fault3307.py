from pathlib import Path
import os, hashlib, json
root=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-safety-r5-20261006')
assert os.uname().nodename=='h100-5'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.chdir(root/'repo')
import sys
sys.path.insert(0,str(root/'repo'))
import torch
torch.set_num_threads(8)
p=root/'runs/formal200-6a29642035ff40b2b7787de206f886c0-3307/checkpoints/diagnostics/formal-sampling-attempt_1/formal-first-sampling-batch.pt'
h=hashlib.sha256()
with p.open('rb') as f:
 for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
assert h.hexdigest()=='2544e34b622b59e5fadd802cb4d950e7d122073dac2fd1e2c51ff2d525e6c77c'
a=torch.load(p,map_location='cpu',weights_only=False); t=a['tensors'];m=t['sampling_mask'].bool(); old=t['old_log_probs']; sample=t['sampling_log_probs'];d=old.double()-sample.double();v=d[m];av=v.abs()
from verl import DataProto
from tensordict import TensorDict
from search_r1.llm_agent.sampling_tokens import audit_batch
batch=DataProto(batch=TensorDict(t,batch_size=[len(m)]),non_tensor_batch=a['non_tensor_batch'],meta_info=a['meta_info'])
audit=audit_batch(batch)
top=torch.topk(torch.where(m,d.abs(),torch.full_like(d,-1)).flatten(),20).indices
items=[]
for idx in top.tolist():
 row,pos=divmod(idx,m.shape[1]); cursor=0;turn_info=None
 for turn in a['non_tensor_batch']['fresh200_turns'][row]:
  n=turn['kept_token_count'];e=len(turn['environment_token_ids'])
  if cursor<=pos<cursor+n:turn_info={'turn':turn['turn_index'],'offset':pos-cursor,'kept_count':n,'context_tokens':len(turn['actual_request_token_ids'])};break
  cursor+=n+e
 items.append(dict(row=row,position=pos,token=int(t['responses'][row,pos]),old=float(old[row,pos]),sample=float(sample[row,pos]),signed_delta=float(d[row,pos]),turn=turn_info))
r={'source':a['source_commit'],'raw_sha256':h.hexdigest(),'rows':len(m),'selected':int(m.sum()),'dtypes':{k:str(x.dtype) for k,x in t.items()},'shapes':{k:list(x.shape) for k,x in t.items()},'finite_old':bool(torch.isfinite(old[m]).all()),'finite_sample':bool(torch.isfinite(sample[m]).all()),'abs_quantiles':{str(q):float(torch.quantile(av,q)) for q in [0.,.5,.9,.99,.999,1.]},'fractions_above':{str(x):float((av>x).double().mean()) for x in [.001,.01,.1,1.]},'signed_mean':float(v.mean()),'old_exact_bf16_fraction':float((old[m].float()==old[m].to(torch.bfloat16).float()).float().mean()),'structural_audit':audit,'top20':items}
print(json.dumps(r,allow_nan=False))
