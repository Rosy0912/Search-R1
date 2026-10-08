"""CPU precision-path diagnosis only; no native GPU acceptance or production changes."""
import sys,importlib.util,json,hashlib
from pathlib import Path
import torch
base=Path(__file__).resolve().parents[1];repo=base/'bapo-e1-300-devicefix';sys.path.insert(0,str(repo))
from verl import DataProto
spec=importlib.util.spec_from_file_location('fixture',base/'e1-300-devicefix-r2/tools/native-actor-fixture.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.DataProto=DataProto
b=m.make_data().batch;expected,_=m.reference_gradient(m.make_data())
def calculate(micro,dtype=torch.bfloat16,offset_before_float=True):
 grads=[];currents=[]
 for rank in range(8):
  total=torch.zeros(16,dtype=torch.float32)
  for start in range(rank*16,(rank+1)*16,micro):
   sl=slice(start,start+micro);mask=b['loss_mask'][sl].bool();w=torch.linspace(-.04,.05,16,dtype=torch.float64).float().to(dtype).requires_grad_()
   linear=b['input_ids'][sl].to(dtype).matmul(w)
   current=-3+linear if offset_before_float else -3+linear.double()
   safe=torch.where(mask,current,0.);old=torch.where(mask,b['old_log_probs'][sl],0.);adv=torch.where(mask,b['advantages'][sl],0.)
   ratio=(safe-old).exp();pg=torch.maximum(-adv*ratio,-adv*ratio.clamp(.8,1.2));x=torch.where(mask,b['ref_log_prob'][sl],0.)-safe;kl=(x.exp()-x-1).clamp(-10,10)
   loss=((torch.where(mask,pg+.001*kl,0.).sum(-1)/mask.sum(-1).clamp_min(1)).sum()/16);loss.backward();total+=w.grad.float();currents.append(current.double().detach())
  grads.append(total)
 actual=torch.stack(grads).mean(0).double();error=(actual-expected).abs();rel=error/expected.abs()
 return dict(micro=micro,dtype=str(dtype),offset_before_float=offset_before_float,gradient=actual.tolist(),max_abs=float(error.max()),max_rel=float(rel.max()),mismatch_count=int((error>2e-4+.03*expected.abs()).sum()))
def independent_quantized_gradient(micro):
 # Differentiate the sequence objective explicitly in FP64 at BF16 forward values.
 # Cast upstream derivative and linear VJP exactly where BF16 autograd casts them.
 grads=[];branch_changes=0;forward_error=0.
 full64=-3+b['input_ids'].matmul(torch.linspace(-.04,.05,16,dtype=torch.float64))
 for rank in range(8):
  total=torch.zeros(16,dtype=torch.float32)
  for start in range(rank*16,(rank+1)*16,micro):
   sl=slice(start,start+micro);mask=b['loss_mask'][sl].bool();features=b['input_ids'][sl].bfloat16();weight=torch.linspace(-.04,.05,16,dtype=torch.float64).float().bfloat16()
   current=(-3+features.matmul(weight)).double();ratio=(current-b['old_log_probs'][sl]).exp();adv=b['advantages'][sl]
   # PPO surrogate derivative with both clipping branches independently selected.
   loss1=-adv*ratio;loss2=-adv*ratio.clamp(.8,1.2);g1=-adv*ratio;g2=g1*((ratio>=.8)&(ratio<=1.2))
   derivative=torch.where(loss1>loss2,g1,torch.where(loss1<loss2,g2,(g1+g2)/2))
   x=b['ref_log_prob'][sl]-current;kl_derivative=.001*(1-x.exp())
   scale=16*mask.sum(-1).clamp_min(1)[:,None]
   pg_upstream=torch.where(mask,derivative/scale,0.).bfloat16()
   kl_upstream=torch.where(mask,kl_derivative/scale,0.).bfloat16()
   upstream=pg_upstream+kl_upstream;total+=(features.reshape(-1,16).T.matmul(upstream.reshape(-1))).float()
   r64=(full64[sl]-b['old_log_probs'][sl]).exp();branch_changes+=int((((ratio<.8)!=(r64<.8))|((ratio>1.2)!=(r64>1.2)))[mask].sum())
   forward_error=max(forward_error,float((current-full64[sl]).abs().max()))
  grads.append(total)
 grad=torch.stack(grads).mean(0).double()
 return dict(micro=micro,gradient=grad.tolist(),forward_error=forward_error,ppo_threshold_crossings=branch_changes)
records=[calculate(micro)for micro in (1,2,4,16)]+[calculate(1,torch.float64),calculate(1,torch.float32),calculate(1,offset_before_float=False)]
oracles=[]
for micro in (1,2,4,16):
 q=independent_quantized_gradient(micro);auto=next(r for r in records if r['micro']==micro and r['dtype']=='torch.bfloat16' and r['offset_before_float']);q['autograd_max_error']=float((torch.tensor(q['gradient'],dtype=torch.float64)-torch.tensor(auto['gradient'],dtype=torch.float64)).abs().max());oracles.append(q)
print(json.dumps(dict(quantization_aware_oracles=oracles,scope='CPU_BF16_PRECISION_PATH_DIAGNOSIS_NOT_GPU_ACCEPTANCE',torch=torch.__version__,expected_fp64=expected.tolist(),records=records),indent=2))
