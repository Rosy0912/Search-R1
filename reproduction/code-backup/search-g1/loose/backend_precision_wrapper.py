"""Native-only frozen-source precision observation, no training writes."""
import os,json,hashlib
import torch
from verl.workers.fsdp_workers import ActorRolloutRefWorker
import reproduction.common8.model_smoke as smoke
instances={}
import verl.utils.torch_functional as functional
original_ce=functional.logprobs_from_logits
ce_calls=0
def observe_ce(logits,labels):
 global ce_calls
 result=original_ce(logits,labels);ce_calls+=1
 flat=logits.reshape(-1,logits.shape[-1]);ys=labels.reshape(-1)
 # Compare actual selected-label CE on the same logits; bounded 64 rows only.
 indices=torch.linspace(0,flat.shape[0]-1,min(64,flat.shape[0]),device=flat.device).long()
 selected=flat.index_select(0,indices).float();target=ys.index_select(0,indices)
 fp32=torch.log_softmax(selected,dim=-1).gather(1,target[:,None]).squeeze(1)
 delta=(result.reshape(-1).index_select(0,indices).float()-fp32).abs()
 emit('same_logits_ce_fp32',call=ce_calls,rows=len(indices),logits_dtype=str(logits.dtype),max_abs=float(delta.max()),mean_abs=float(delta.mean()),finite=bool(torch.isfinite(delta).all()))
 return result
functional.logprobs_from_logits=observe_ce
def emit(kind,**data):
 record=dict(kind=kind,rank=int(os.environ['RANK']),**data)
 from pathlib import Path
 root=Path(os.environ['SEARCH_G1_BASE'])/'logs'
 with (root/('precision-rank-'+os.environ['RANK']+'.jsonl')).open('a') as f:
  f.write(json.dumps(record)+'\n');f.flush();os.fsync(f.fileno())
def signature(module):
 result={}
 for n,b in module.named_buffers():
  if n.endswith('inv_freq'):
   assert b.dtype==torch.float32 and torch.isfinite(b).all()
   t=b.detach().cpu().contiguous()
   result[n]=dict(dtype=str(b.dtype),sha256=hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest())
 assert result
 return result
init=ActorRolloutRefWorker.init_model
def initialize(self):
 result=init(self)
 if self._is_actor:instances['actor']=self;emit('actor_initialized',buffers=signature(self.actor_module_fsdp),optimizer_state_entries=len(self.actor_optimizer.state))
 if self._is_ref:instances['reference']=self;emit('reference_initialized',buffers=signature(self.ref_module_fsdp))
 if self._is_actor:
  vm=self.rollout.inference_engine.llm_engine.model_executor.worker.model_runner.model
  caches=[]
  for name,child in vm.named_modules():
   cache=getattr(child,'cos_sin_cache',None)
   if cache is None:continue
   positions=torch.tensor([x for x in (0,1,16,128,1024,4095) if x<cache.shape[0]],dtype=torch.long)
   dim=child.rotary_dim;base=child.base
   inv=1.0/(base**(torch.arange(0,dim,2,dtype=torch.float32)/dim))
   freqs=torch.outer(positions.float(),inv)
   cpu=torch.cat((freqs.cos(),freqs.sin()),dim=-1).to(cache.dtype)
   observed=cache.detach().index_select(0,positions.to(cache.device)).cpu()
   delta=(observed.float()-cpu.float()).abs()
   caches.append(dict(name=name,base=base,rotary_dim=dim,dtype=str(cache.dtype),positions=positions.tolist(),max_abs_vs_cpu=float(delta.max()),mean_abs_vs_cpu=float(delta.mean())))
  emit('vllm_rotary_cache_vs_cpu',caches=caches)
 return result
ActorRolloutRefWorker.init_model=initialize
compare=smoke.compare
def observed(old,reference,mask,tokens,logs,responses):
 valid=mask.bool().cpu();o=old.cpu().float();r=reference.cpu().float();raw=logs.cpu().float();v=valid[:,:raw.shape[1]]
 ar=(o[valid]-r[valid]).abs();av=(o[:,:raw.shape[1]][v]-raw[v]).abs()
 emit('actual_parity',actor_ref_max=float(ar.max()),actor_vllm_max=float(av.max()),actor_vllm_mean=float(av.mean()),actor_vllm_q99=float(torch.quantile(av,.99)),valid_tokens=int(v.sum()),actor_buffers=signature(instances['actor'].actor_module_fsdp),reference_buffers=signature(instances['reference'].ref_module_fsdp),optimizer_state_entries=len(instances['actor'].actor_optimizer.state),optimizer_writes=getattr(instances['actor'].actor,'_optimizer_writes',0))
 return compare(old,reference,mask,tokens,logs,responses)
smoke.compare=observed
smoke.main(os.environ['SEARCH_G1_BASE'],'initial')
