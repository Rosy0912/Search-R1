"""Read-only exact response NLL/tag accounting; no backward, optimizer or generation."""
import os,json,hashlib,time,gc,sys,re
from pathlib import Path
root=Path('/dataset/shaoshitong/search-r1-rl/cosearch/author-native')
assert os.environ['SLURM_JOB_NODELIST']=='h100-16' and os.environ['SLURM_NNODES']=='1'
assert os.environ['SEARCH_R1_GPU_PREFLIGHT']=='1'
leases=json.loads(os.environ['SEARCH_R1_GPU_LEASES']);assert len({v['uuid'] for v in leases})==8
for k in ['ROCR_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL']:os.environ.pop(k,None)
import torch,inspect,transformers
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM
sys.path.insert(0,str(root.parent/'repo'))
from search_r1.baselines.warm_start import SupervisedSequenceLoss
run=root/'runs/author-native-token-loss-probe-m';run.mkdir(exist_ok=False)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
corpus_path=root/'native-grammar-warm-start-fulltext-v2/corpus.json'
corpus=json.loads(corpus_path.read_text());assert sha(corpus_path)==json.loads((corpus_path.parent/'manifest.json').read_text())['corpus_sha256']
k_run=root/'runs/author-native-format-t1-probe-k'
k=json.loads((k_run/'result.json').read_text());assert k['valid_calls']==0 and k['total_samples']==128
assert json.loads((k_run/'driver-exit.json').read_text())['returncode']==0
assert sha(k_run/'outputs.jsonl')==k['outputs_sha256']
models=[root/'runs/author-native-grammar-warm-start-v2/warm-start/actor',root/'runs/author-native-grammar-actor-continue-v3/warm-start/actor']
tok=AutoTokenizer.from_pretrained(models[-1],local_files_only=True)
cases=[]
for index in [0,2,1]:
 r=corpus['actor'][index];n=r['response_length']
 cases.append(dict(label='SFT_search' if index%2==0 else 'SFT_final',index=index,prefix=r['input_ids'][:-n],target=r['input_ids'][-n:]))
sample=json.loads((k_run/'outputs.jsonl').read_text().splitlines()[0]);match=re.search(r'(\{"name".*?\}\})',sample['text']);assert match
call=json.loads(match[1]);assert call['name']=='search' and isinstance(call['arguments']['query'],str)
target=tok.encode('<reason>I need evidence to answer this question.</reason>\n<tool_call>'+json.dumps(call,ensure_ascii=False)+'</tool_call>',add_special_tokens=False)+[tok.eos_token_id]
cases.append(dict(label='K_unseen_TRAIN_search_teacher_target_no_gold',index=0,prefix=k['question_prefixes'][0]['prefix_ids'],target=target))
result=dict(run=run.name,job=os.environ['SLURM_JOB_ID'],real_rl_updates=0,supervised_steps=0,leases=leases,corpus_sha256=sha(corpus_path),K_result_sha256=sha(k_run/'result.json'),cases=[],source_sha256={})
source=root.parent/'repo/search_r1/baselines/warm_start.py'
expected=json.loads((root/'native-actor-continue-tools-v3/engine-source-manifest.json').read_text())
result['objective_source_sha256']=sha(source)
result['engine_manifest']=expected
assert result['objective_source_sha256']=='ce84c6b6917ac4c769b72af4d2adc446bd54f8dd39c1be3cf39435aaadf7dc9f'
result['HF_version']=transformers.__version__
result['historical_SFT_HF_version']='4.46.3'
result['diagnostic_compatibility']='Read-only num_logits_to_keep (historical HF4.46.3) -> logits_to_keep (native HF4.57.6); not a training change'
class NativeKeepAdapter(torch.nn.Module):
 def __init__(self,model):super().__init__();self.model=model
 def forward(self,*args,**kwargs):
  keep=kwargs.pop('num_logits_to_keep');assert isinstance(keep,int) and keep>0
  assert 'logits_to_keep' in inspect.signature(self.model.forward).parameters
  out=self.model(*args,logits_to_keep=keep,**kwargs)
  assert out.logits.shape[1]==keep,(out.logits.shape,keep)
  return out
started=time.monotonic();torch.cuda.set_device(0)
uuid=str(torch.cuda.get_device_properties(0).uuid);assert uuid.removeprefix('GPU-') in {v['uuid'].removeprefix('GPU-') for v in leases}
result['actual_cuda0_uuid']=uuid
saved_rows={}
for path in models:
 manifest=json.loads((path.parent/'manifest.json').read_text())
 assert manifest['completed']
 result['source_sha256'][str(path)]={}
 for name,wanted in manifest['roles']['actor']['files_sha256'].items():
  if name=='optimizer.pt':continue
  actual=sha(path/name);assert actual==wanted
  result['source_sha256'][str(path)][name]=actual
 model=AutoModelForCausalLM.from_pretrained(path,torch_dtype=torch.float32,attn_implementation='sdpa',local_files_only=True).to('cuda:0');model.eval()
 saved_rows[str(path)]={name:p.detach()[[151657,151658]].cpu().clone() for name,p in [('input_embeddings',model.get_input_embeddings().weight),('output_embeddings',model.get_output_embeddings().weight)]}
 result.setdefault('embedding_tie_identity',{})[str(path)]=model.get_input_embeddings().weight is model.get_output_embeddings().weight
 with torch.inference_mode():
  for case in cases:
   prefix,target=case['prefix'],case['target'];n=len(target);p=len(prefix)
   ids=torch.tensor([prefix+target],device='cuda:0');mask=torch.ones_like(ids)
   positions=mask.cumsum(-1)-1
   full=model(ids,attention_mask=mask,position_ids=positions,use_cache=False).logits[0,p-1:p+n-1].float()
   labels=torch.tensor(target,device='cuda:0')
   independent=F.cross_entropy(full,labels,reduction='none')
   original=SupervisedSequenceLoss(NativeKeepAdapter(model))(ids,mask,torch.tensor([n],device='cuda:0'),n)[0]
   torch.testing.assert_close(original,independent.mean(),rtol=1e-5,atol=1e-5)
   tokens=[]
   for i,t in enumerate(target):
    lp=F.log_softmax(full[i],dim=-1)
    tokens.append(dict(position=i,id=t,text=tok.decode([t],skip_special_tokens=False),nll=float(independent[i]),probability=float(lp[t].exp()),rank=int((full[i]>full[i,t]).sum())+1,response_mask=1))
   tags=[v for v in tokens if v['id'] in [151657,151658]]
   result['cases'].append(dict(model_path=str(path),label=case['label'],index=case['index'],prefix_ids=prefix,target_ids=target,target_text=tok.decode(target,skip_special_tokens=False),tokens=tokens,original_sequence_mean_nll=float(original),independent_sequence_mean_nll=float(independent.mean()),tag_nll_sum=sum(v['nll'] for v in tags),tag_contribution_to_sequence_mean=sum(v['nll'] for v in tags)/n,tag_probes=tags,EOS_nll=sum(v['nll'] for v in tokens if v['id']==tok.eos_token_id)))
   del full,independent,original,ids,mask,lp
   (run/'result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
  # Reproduce production mixed response lengths with explicit left-pad attention,
  # including EOS IDs even when EOS equals PAD. Compare every row to full CE.
  selected=cases[:3];width=max(len(c['prefix'])+len(c['target']) for c in selected)
  batch_ids=[];batch_masks=[];lengths=[]
  for c in selected:
   sequence=c['prefix']+c['target'];pad=width-len(sequence)
   batch_ids.append([tok.pad_token_id]*pad+sequence);batch_masks.append([0]*pad+[1]*len(sequence));lengths.append(len(c['target']))
  ids=torch.tensor(batch_ids,device='cuda:0');mask=torch.tensor(batch_masks,device='cuda:0');positions=mask.cumsum(-1)-1;positions.masked_fill_(mask==0,1)
  full=model(ids,attention_mask=mask,position_ids=positions,use_cache=False).logits.float()
  original=SupervisedSequenceLoss(NativeKeepAdapter(model))(ids,mask,torch.tensor(lengths,device='cuda:0'),max(lengths))
  checks=[]
  for i,n in enumerate(lengths):
   independent=F.cross_entropy(full[i,width-n-1:width-1],ids[i,-n:],reduction='none')
   torch.testing.assert_close(original[i],independent.mean(),rtol=1e-5,atol=1e-5)
   singleton=[v for v in result['cases'] if v['model_path']==str(path) and v['label']==selected[i]['label'] and v['index']==selected[i]['index']][0]['independent_sequence_mean_nll']
   torch.testing.assert_close(independent.mean(),torch.tensor(singleton,device='cuda:0'),rtol=1e-5,atol=1e-5)
   checks.append(dict(case_index=i,response_length=n,original_nll=float(original[i]),independent_nll=float(independent.mean()),response_mask=[int(j>=max(lengths)-n) for j in range(max(lengths))],pad_count=width-len(selected[i]['prefix'])-n,EOS_target_id=selected[i]['target'][-1],token_nll=independent.tolist()))
   checks[-1]['singleton_unpadded_nll']=singleton
   checks[-1]['padding_nll_difference']=float(independent.mean())-singleton
  result.setdefault('mixed_left_pad_checks',[]).append(dict(model_path=str(path),width=width,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id,attention_mask=mask.tolist(),rows=checks))
  del full,independent,original,ids,mask,positions
 assert all(p.grad is None for p in model.parameters())
 del model;gc.collect();torch.cuda.empty_cache()
result['actual_saved_tag_row_deltas']={name:dict(l2_per_tag=(saved_rows[str(models[-1])][name]-value).double().norm(dim=1).tolist(),incoming_norm=value.double().norm(dim=1).tolist(),outgoing_norm=saved_rows[str(models[-1])][name].double().norm(dim=1).tolist()) for name,value in saved_rows[str(models[0])].items()}
result.update(wall_seconds=time.monotonic()-started,peak_bytes=torch.cuda.max_memory_allocated(),not_vLLM_parity=True,no_Adam_or_backward_executed=True)
(run/'result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
(run/'driver-exit.json').write_text(json.dumps(dict(returncode=0,job=os.environ['SLURM_JOB_ID'],real_rl_updates=0,supervised_steps=0)))
print(json.dumps(dict(completed=True,run=run.name,real_rl_updates=0,supervised_steps=0,wall_seconds=result['wall_seconds'])),flush=True)
