"""Disposable native-only, same-token reference buffer intervention; no trainer."""
import os,json,hashlib
from pathlib import Path
import torch
from verl.workers.fsdp_workers import ActorRolloutRefWorker
instances={}
init=ActorRolloutRefWorker.init_model

def record(name,**data):
 print('COMMON8_ROPE_CAUSAL='+json.dumps(dict(kind=name,rank=int(os.environ['RANK']),**data)),flush=True)

def hash_tensor(t):
 t=t.detach().cpu().contiguous()
 return hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()

def initialize(self):
 result=init(self)
 if self._is_actor:instances['actor']=self
 if self._is_ref:instances['reference']=self
 return result
ActorRolloutRefWorker.init_model=initialize
reference_call=ActorRolloutRefWorker.compute_ref_log_prob

def summarize(old,value,mask):
 delta=(old[mask].float()-value[mask].float()).abs()
 assert torch.isfinite(delta).all()
 return dict(max_abs=float(delta.max()),mean_abs=float(delta.mean()),q99_abs=float(torch.quantile(delta,.99)),valid_tokens=int(mask.sum()))

def intervene(self,data):
 actor=instances['actor'];am=actor.actor_module_fsdp;rm=self.ref_module_fsdp
 old=data.batch['old_log_probs'].detach().cpu().clone();mask=data.batch['attention_mask'][:,-old.shape[-1]:].detach().cpu().bool()
 ap=dict(am.named_parameters());rp=dict(rm.named_parameters());assert set(ap)==set(rp)
 parameter_equal={}
 for name,a in ap.items():
  r=rp[name];assert a.shape==r.shape
  parameter_equal[name]=bool(torch.equal(a.detach().cpu().to(torch.bfloat16),r.detach().cpu().to(torch.bfloat16)))
 assert all(parameter_equal.values()), 'Fresh actor/reference BF16 parameter shards differ'
 record('canonical_bf16_shard_comparison',equal=True,shards=len(parameter_equal),source='actual FSDP parameter shards at each rank, dtype-normalized only for comparison')
 ab=dict(am.named_buffers());rb=dict(rm.named_buffers());names=sorted(n for n in ab if n.endswith('inv_freq'))
 assert names and all(n in rb for n in names)
 original_buffers={};buffers=[]
 for n in names:
  a,r=ab[n],rb[n];assert a.shape==r.shape
  assert a.dtype==torch.float32 and torch.isfinite(a).all() and torch.isfinite(r).all()
  parent,key=n.rsplit('.',1);module=rm.get_submodule(parent)
  assert module._buffers[key] is r
  original_buffers[n]=(module,key,r)
  buffers.append(dict(name=n,actor_dtype=str(a.dtype),reference_dtype=str(r.dtype),shape=list(a.shape),actor_sha256=hash_tensor(a),reference_sha256=hash_tensor(r),max_abs=float((a.detach().cpu().float()-r.detach().cpu().float()).abs().max())))
 record('rope_buffers_before',buffers=buffers)
 # Hash all teacher-forced IDs/masks/positions before either reference call.
 keys=('input_ids','responses','attention_mask','position_ids');input_hash={k:hash_tensor(data.batch[k]) for k in keys}
 before=reference_call(self,data)
 assert {k:hash_tensor(data.batch[k]) for k in keys}==input_hash
 record('reference_original',delta=summarize(old,before.batch['ref_log_prob'].cpu(),mask),input_sha256=input_hash)
 try:
  for n,(module,key,r) in original_buffers.items():module._buffers[key]=ab[n].detach().clone().to(device=r.device)
  after=reference_call(self,data)
  assert {k:hash_tensor(data.batch[k]) for k in keys}==input_hash
  record('reference_fp32_actor_rope_intervention',delta=summarize(old,after.batch['ref_log_prob'].cpu(),mask),input_sha256=input_hash,buffers=[dict(name=n,dtype=str(module._buffers[key].dtype),sha256=hash_tensor(module._buffers[key])) for n,(module,key,r) in original_buffers.items()],semantics='Only reference inv_freq replaced by actual unrounded actor FP32 buffer; official parameters/tokens/masks/positions unchanged')
 finally:
  for n,(module,key,r) in original_buffers.items():module._buffers[key]=r
  assert all(hash_tensor(rb[n])==hash_tensor(module._buffers[key]) for n,(module,key,r) in original_buffers.items())
  record('reference_restored',restored=True,optimizer_writes=getattr(actor.actor,'_optimizer_writes',0),optimizer_state_entries=len(actor.actor_optimizer.state))
 # Return the original reference result to the original smoke gate; do not make
 # the causal diagnostic impersonate a production parity acceptance.
 return before
ActorRolloutRefWorker.compute_ref_log_prob=intervene
from reproduction.common8.model_smoke import main
main(os.environ['SEARCH_G1_BASE'],'initial')
