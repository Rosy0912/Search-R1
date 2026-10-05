"""Disposable same-prefix diagnostic; unchanged original parity gate, zero writes."""
import os,json,hashlib
from pathlib import Path
from contextlib import nullcontext
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from verl.third_party.vllm.vllm_v_0_6_3.llm import LLM
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.utils.fsdp_utils import load_fsdp_param_and_grad,offload_fsdp_param_and_grad
from verl.utils.torch_functional import logprobs_from_logits
import reproduction.common8.model_smoke as smoke


def emit(kind,**kw):
 p=Path(os.environ['SEARCH_G1_BASE'])/'logs'/('precision-rank-'+os.environ['RANK']+'.jsonl')
 with p.open('a') as f:
  f.write(json.dumps(dict(kind=kind,rank=int(os.environ['RANK']),**kw))+'\n');f.flush();os.fsync(f.fileno())

capture={};original_generate=LLM.generate

def generate(self,*args,**kwargs):
 p=kwargs['sampling_params']
 emit('actual_sampling_parameters',temperature=p.temperature,top_p=p.top_p,top_k=p.top_k,min_p=p.min_p,
      presence_penalty=p.presence_penalty,frequency_penalty=p.frequency_penalty,repetition_penalty=p.repetition_penalty)
 out=original_generate(self,*args,**kwargs);capture['raw']=out[1].detach().cpu().clone();return out
LLM.generate=generate
original_ref=ActorRolloutRefWorker.compute_ref_log_prob


def summary(values):
 x=values.detach().cpu().float().abs();return dict(max_abs=float(x.max()),mean_abs=float(x.mean()),p99_abs=float(x.quantile(.99)),count=x.numel())


def reference(self,data):
 result=original_ref(self,data)
 old=data.batch['old_log_probs'].detach().cpu();raw=capture['raw'];responses=data.batch['responses']
 ids=data.batch['input_ids'];mask=data.batch['attention_mask'].bool();pos=data.batch['position_ids']
 width=responses.shape[1];offset=ids.shape[1]-width
 module=self.ref_module_fsdp;module.eval()
 if self._is_offload_param:load_fsdp_param_and_grad(module,device_id=torch.cuda.current_device(),load_grad=self._is_offload_grad)
 rows=[]
 try:
  with torch.no_grad(),torch.autocast(device_type='cuda',dtype=torch.bfloat16):
   for i in range(8):
    n=int(mask[i,offset:].sum());prompt_count=int(mask[i,:offset].sum())
    actual=ids[i][mask[i]][None];positions=pos[i][mask[i]][None]
    labels=responses[i,:n];stats={};full_logs={}
    for name in ('single_padded_auto','single_unpadded_auto','single_unpadded_math','single_unpadded_flash'):
     padded=name=='single_padded_auto'
     seq=ids[i:i+1] if padded else actual
     am=mask[i:i+1].long() if padded else torch.ones_like(actual)
     pp=pos[i:i+1] if padded else positions
     backend=SDPBackend.MATH if name.endswith('math') else SDPBackend.FLASH_ATTENTION if name.endswith('flash') else None
     context=sdpa_kernel(backend) if backend is not None else nullcontext()
     with context:out=module(input_ids=seq,attention_mask=am,position_ids=pp,use_cache=False)
     start=offset-1 if padded else prompt_count-1
     logits=out.logits[:,start:start+n,:];logits.div_(float(self.config.rollout.temperature))
     logs=logprobs_from_logits(logits,labels[None])[0].detach().cpu()
     full_logs[name]=logs
     stats[name]=dict(vs_original_actor=summary(logs-old[i,:n]),vs_actual_vllm=summary(logs-raw[i,:n]))
     del out,logits
    # Fixed first16 actual labels; identical collective counts even after EOS.
    out=module(input_ids=actual[:,:prompt_count],attention_mask=torch.ones_like(actual[:,:prompt_count]),
               position_ids=positions[:,:prompt_count],use_cache=True)
    past=out.past_key_values;incremental=[]
    for t in range(16):
     if t<n:
      last=out.logits[:,-1:,:]/float(self.config.rollout.temperature)
      incremental.append(logprobs_from_logits(last,labels[t:t+1][None])[0,0].detach().cpu())
     del out
     token=responses[i,t:t+1][None] if t<n else torch.full((1,1),self.tokenizer.eos_token_id,device=ids.device,dtype=ids.dtype)
     out=module(input_ids=token,attention_mask=torch.ones((1,prompt_count+t+1),device=ids.device,dtype=ids.dtype),
                position_ids=torch.tensor([[prompt_count+t]],device=ids.device),past_key_values=past,use_cache=True)
     past=out.past_key_values
    del out,past
    incremental=torch.stack(incremental);k=len(incremental)
    stats['incremental_auto_first16']=dict(vs_original_actor=summary(incremental-old[i,:k]),vs_actual_vllm=summary(incremental-raw[i,:k]),
      vs_unpadded_full=summary(incremental-full_logs['single_unpadded_auto'][:k]),scope='first min(16,n) real labels; inactive extra forwards excluded')
    rows.append(dict(prompt_count=prompt_count,response_count=n,input_sha256=hashlib.sha256(actual.detach().cpu().numpy().tobytes()).hexdigest(),stats=stats))
 except Exception as exc:
  emit('diagnostic_exception',exception=type(exc).__name__,message=str(exc));raise
 finally:
  if self._is_offload_param:offload_fsdp_param_and_grad(module,offload_grad=self._is_offload_grad)
  torch.cuda.empty_cache()
 emit('padding_backend_same_prefix',rows=rows,optimizer_writes=0,scope='Actual reference HF same actor/ref policy; fixed original rollout tokens; original smoke gate unchanged')
 return result
ActorRolloutRefWorker.compute_ref_log_prob=reference
smoke.main(os.environ['SEARCH_G1_BASE'],'initial')
