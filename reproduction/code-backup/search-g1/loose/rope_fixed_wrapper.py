"""Native-only frozen-source precision observation, no training writes."""
import os,json,hashlib
import torch
from verl.workers.fsdp_workers import ActorRolloutRefWorker
import reproduction.common8.model_smoke as smoke
instances={}
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
