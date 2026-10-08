"""Actual same-operand fused arithmetic observations; original outputs preserved."""
import os,json
from pathlib import Path
import torch
from torch.nn import functional as F
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.linear import QKVParallelLinear,MergedColumnParallelLinear,UnquantizedLinearMethod
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.activation import SiluAndMul
import reproduction.common8.model_smoke as smoke
from verl.third_party.vllm.vllm_v_0_6_3.llm import LLM
records=[];active=False;sync_count=0
original_sync=LLM.sync_model_weights


def sync(self,*args,**kwargs):
 global active,sync_count
 result=original_sync(self,*args,**kwargs)
 assert kwargs.get('load_format',args[1] if len(args)>1 else None)=='dtensor'
 sync_count+=1;active=True
 return result
LLM.sync_model_weights=sync


def compare(a,b):
 d=(a.float()-b.float()).abs();return dict(max_abs=float(d.max()),mean_abs=float(d.mean()),different_values=int((d!=0).sum()),value_count=d.numel())


def linear_wrapper(original,kind):
 def forward(self,x):
  if not active:return original(self,x)
  output=original(self,x);count=getattr(self,'_diagnostic_count',0);self._diagnostic_count=count+1
  if count<2:
   assert get_tensor_model_parallel_world_size()==1 and type(self.quant_method) is UnquantizedLinearMethod
   sizes=self.output_partition_sizes;assert sum(sizes)==self.weight.shape[0] and len(sizes)==(3 if kind=='qkv' else 2)
   bias=self.bias if not self.skip_bias_add else None
   def separate():
    offset=0;chunks=[]
    for size in sizes:
     chunks.append(F.linear(x,self.weight.narrow(0,offset,size),None if bias is None else bias.narrow(0,offset,size)));offset+=size
    return torch.cat(chunks,-1)
   divided=separate();flag=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
   try:
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    fused_full=F.linear(x,self.weight,bias);divided_full=separate()
   finally:torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=flag
   records.append(dict(kind=kind,call=count,input_shape=list(x.shape),input_dtype=str(x.dtype),sizes=list(sizes),
     original_vs_split_same_operand=compare(output[0],divided),original_vs_no_reduced_fused=compare(output[0],fused_full),
     original_vs_no_reduced_split=compare(output[0],divided_full),default_reduced_bf16=flag,matmul_precision=torch.get_float32_matmul_precision()))
  return output
 return forward
QKVParallelLinear.forward=linear_wrapper(QKVParallelLinear.forward,'qkv')
MergedColumnParallelLinear.forward=linear_wrapper(MergedColumnParallelLinear.forward,'gate_up')
original_norm=RMSNorm.forward_cuda

def norm(self,x,residual=None):
 if not active:return original_norm(self,x,residual)
 count=getattr(self,'_diagnostic_count',0);self._diagnostic_count=count+1
 if count<2:
  combined=x.clone() if residual is None else x+residual
  v=combined.float().pow(2).mean(-1,keepdim=True)
  expected=(combined.float()*torch.rsqrt(v+self.variance_epsilon)).to(combined.dtype)*self.weight
 out=original_norm(self,x,residual)
 if count<2:
  normalized=out if residual is None else out[0]
  row=dict(kind='rmsnorm',call=count,residual=residual is not None,input_shape=list(x.shape),original_vs_HF_same_operand=compare(normalized,expected))
  if residual is not None:row['residual_vs_HF']=compare(out[1],combined)
  records.append(row)
 return out
RMSNorm.forward_cuda=norm
original_silu=SiluAndMul.forward_cuda

def silu(self,x):
 if not active:return original_silu(self,x)
 count=getattr(self,'_diagnostic_count',0);self._diagnostic_count=count+1
 if count<2:
  gate,up=x.chunk(2,-1);expected=F.silu(gate)*up
 out=original_silu(self,x)
 if count<2:records.append(dict(kind='silu',call=count,input_shape=list(x.shape),original_vs_HF_same_operand=compare(out,expected)))
 return out
SiluAndMul.forward_cuda=silu
original_compare=smoke.compare

def gate(*args,**kwargs):
 assert active and sync_count==1
 p=Path(os.environ['SEARCH_G1_BASE'])/'logs'/('precision-rank-'+os.environ['RANK']+'.jsonl')
 with p.open('x') as f:
  for row in records:f.write(json.dumps(dict(rank=int(os.environ['RANK']),phase='after_actual_dtensor_sync',sync_count=sync_count,**row))+'\n')
  f.flush();os.fsync(f.fileno())
 return original_compare(*args,**kwargs)
smoke.compare=gate
smoke.main(os.environ['SEARCH_G1_BASE'],'initial')
