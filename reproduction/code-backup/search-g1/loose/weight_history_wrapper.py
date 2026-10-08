"""Disposable native audit at actual sync; original computation unchanged."""
import os,json,hashlib
from pathlib import Path
import torch
from verl.third_party.vllm.vllm_v_0_6_3.llm import LLM
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.third_party.vllm.vllm_v_0_6_3.dtensor_weight_loaders import redistribute_dtensor
import reproduction.common8.model_smoke as smoke

def emit(kind,**kw):
 p=Path(os.environ['SEARCH_G1_BASE'])/'logs'/('precision-rank-'+os.environ['RANK']+'.jsonl')
 with p.open('a') as f:
  f.write(json.dumps(dict(kind=kind,rank=int(os.environ['RANK']),**kw))+'\n');f.flush();os.fsync(f.fileno())
original_sync=LLM.sync_model_weights

def sync(self,actor_weights,load_format):
 original_sync(self,actor_weights,load_format)
 assert load_format=='dtensor'
 model=self.llm_engine.model_executor.worker.model_runner.model
 params=dict(model.named_parameters(remove_duplicate=False));rows=[]
 cfg=model.config;hd=cfg.hidden_size//cfg.num_attention_heads
 q=cfg.num_attention_heads*hd;k=cfg.num_key_value_heads*hd;inter=cfg.intermediate_size
 for source in sorted(actor_weights):
  if source.endswith('inv_freq'):continue
  if cfg.tie_word_embeddings and source=='lm_head.weight':continue
  target=source;offset=0;length=None
  for src,dst,start,size in (('q_proj','qkv_proj',0,q),('k_proj','qkv_proj',q,k),('v_proj','qkv_proj',q+k,k),('gate_proj','gate_up_proj',0,inter),('up_proj','gate_up_proj',inter,inter)):
   if '.'+src+'.' in source:
    target=source.replace('.'+src+'.','.'+dst+'.');offset=start;length=size;break
  assert target in params,(source,target)
  loaded=redistribute_dtensor(target,actor_weights[source])
  actual=params[target].detach()
  if length is not None:actual=actual.narrow(0,offset,length)
  expected=loaded.to(device=actual.device,dtype=actual.dtype)
  assert actual.shape==expected.shape,(source,actual.shape,expected.shape)
  equal=bool(torch.equal(actual,expected))
  rows.append(dict(source=source,target=target,shape=list(actual.shape),equal=equal,max_abs=0.0 if equal else float((actual.float()-expected.float()).abs().max())))
  del loaded,expected
 tied=dict(enabled=bool(cfg.tie_word_embeddings))
 if cfg.tie_word_embeddings:
  a=params['model.embed_tokens.weight'];b=params['lm_head.weight'];tied.update(equal=bool(torch.equal(a,b)),same_storage=a.data_ptr()==b.data_ptr())
 emit('actual_sync_weight_audit',rows=rows,tied=tied,scope='actual DTensor sync inputs after original sync before any offload; TP1 fused QKV/gateup mapped explicitly')
LLM.sync_model_weights=sync
original_generate=LLM.generate
captured={}
def generate(self,*args,**kw):
 captured['prompts']=kw['prompt_token_ids']
 out=original_generate(self,*args,**kw);captured['tokens']=out[0].detach().cpu().clone();captured['logs']=out[1].detach().cpu().clone();return out
LLM.generate=generate
original_ref=ActorRolloutRefWorker.compute_ref_log_prob

def reference(self,data):
 result=original_ref(self,data)
 ids=data.batch['input_ids'].detach().cpu();mask=data.batch['attention_mask'].detach().cpu().bool();pos=data.batch['position_ids'].detach().cpu();responses=data.batch['responses'].detach().cpu();old=data.batch['old_log_probs'].detach().cpu()
 width=responses.shape[1];start=ids.shape[1]-width;raw=captured['logs'];rows=[]
 for i,prompt in enumerate(captured['prompts']):
  pm=mask[i,:start];actual=ids[i,:start][pm];positions=pos[i,:start][pm];count=int(mask[i,start:].sum())
  match=actual.tolist()==prompt;position_match=positions.tolist()==list(range(len(prompt))) and pos[i,start:start+count].tolist()==list(range(len(prompt),len(prompt)+count))
  delta=(old[i,:count].float()-raw[i,:count].float()).abs();maximum=int(delta.argmax())
  rows.append(dict(prompt_tokens=len(prompt),response_tokens=count,prompt_history_equal=match,positions_contiguous=position_match,max_abs=float(delta.max()),mean_abs=float(delta.mean()),max_at_response_position=maximum,last_token_abs=float(delta[-1]),last_token_is_eos=bool(responses[i,count-1]==self.tokenizer.eos_token_id),prompt_sha256=hashlib.sha256(actual.numpy().tobytes()).hexdigest()))
 emit('actual_history_audit',rows=rows)
 return result
ActorRolloutRefWorker.compute_ref_log_prob=reference
smoke.main(os.environ['SEARCH_G1_BASE'],'initial')
