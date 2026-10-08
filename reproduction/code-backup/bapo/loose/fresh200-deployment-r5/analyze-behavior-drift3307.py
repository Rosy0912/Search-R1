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

w=m.double()/m.sum(-1,keepdim=True).clamp_min(1)
k3=(d.exp()-d-1)
neg=(t['advantages']<0)&m
r={'scope':'ORIGINAL_PREUPDATE_HF0_VERSUS_ACTUAL_BEHAVIOR_NUMERICAL_DRIFT_NOT_EFFECT_OR_NATIVE_PASS','raw_sha256':h.hexdigest(),'rows':len(m),'selected':int(m.sum()),'sequence_mean_k3':float((k3*w).sum()/len(m)),'token_mean_k3':float(k3[m].mean()),'negative_weight_mass':float(w[neg].sum()),'negative_ratio_above_3_weight_fraction':float(w[neg&(d>__import__('math').log(3))].sum()/w[neg].sum()) if neg.any() else 0.,'row_k3_quantiles':{str(q):float(torch.quantile((k3*w).sum(-1),q)) for q in [.5,.9,.99,1.]},'ratio_quantiles':{str(q):float(torch.quantile(d[m].exp(),q)) for q in [0.,.01,.5,.99,1.]},'hf0_self_ratio_note':'HF0/HF0 identically1 conceals actual HF0/behavior drift'}
print(json.dumps(r,allow_nan=False))
