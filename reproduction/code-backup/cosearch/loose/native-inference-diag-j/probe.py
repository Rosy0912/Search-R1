"""Read-only HF warm initialization/format diagnostic; no training or RL updates."""
import os,json
assert os.environ['SLURM_JOB_NODELIST']=='h100-16' and os.environ['SLURM_NNODES']=='1'
assert os.environ['SEARCH_R1_GPU_PREFLIGHT']=='1'
leases=json.loads(os.environ['SEARCH_R1_GPU_LEASES']);assert len(leases)==8 and len({v['uuid'] for v in leases})==8
for k in ['ROCR_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL']:os.environ.pop(k,None)
assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))==8
import torch,time,hashlib,gc
from pathlib import Path
from transformers import AutoTokenizer,AutoModelForCausalLM
root=Path('/dataset/shaoshitong/search-r1-rl/cosearch/author-native')
run=root/'runs/author-native-warm-hf-format-diagnostic-j';run.mkdir(exist_ok=False)
import sys,traceback
def failed(type_,value,tb):
 (run/'driver-exit.json').write_text(json.dumps(dict(returncode=1,job=os.environ['SLURM_JOB_ID'],real_rl_updates=0,error=str(value))))
 traceback.print_exception(type_,value,tb)
sys.excepthook=failed
model_path=root/'runs/author-native-grammar-warm-start-v2/warm-start/actor'
tok=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
corpus_file=root/'native-grammar-warm-start-fulltext-v2/corpus.json'
corpus=json.loads(corpus_file.read_text())
row=corpus['actor'][0];train_prefix=row['input_ids'][:-row['response_length']];train_target=row['input_ids'][-row['response_length']:]
result=dict(run=run.name,job=os.environ['SLURM_JOB_ID'],real_rl_updates=0,training_steps=0,source='warmv2 actor HF weights',leases=leases,cases=[],diagnostic_greedy_temperature=0,not_native_sampling_acceptance=True)
def file_hash(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
result['corpus_sha256']=file_hash(corpus_file)
manifest=json.loads((model_path.parent/'manifest.json').read_text())
result['hf_files_sha256']={}
for name,expected in manifest['roles']['actor']['files_sha256'].items():
 if name=='optimizer.pt':continue
 actual=file_hash(model_path/name);assert actual==expected
 result['hf_files_sha256'][name]=actual
raw_file=sorted((root/'runs/author-native-warm-five-i/evidence/raw-rollouts').glob('pre-postprocess-696092-*.pt'))[0]
import sys,re
sys.path[:0]=[str(root/'repo'),str(root/'repo/verl')]
raw=torch.load(raw_file,map_location='cpu',weights_only=False)[0]
actual_prefix=raw.prompt_ids.reshape(-1).tolist()
while actual_prefix and actual_prefix[0]==tok.pad_token_id:actual_prefix.pop(0)
actual_response=tok.decode(raw.response_ids.reshape(-1)[raw.response_mask.reshape(-1).bool()].tolist(),skip_special_tokens=False)
match=re.search(r'(\{"name".*?\}\})',actual_response)
assert match,actual_response
call=json.loads(match[1]);assert call['name']=='search'
actual_target=tok.encode('<reason>I need evidence to answer this question.</reason>\n<tool_call>'+json.dumps(call,ensure_ascii=False)+'</tool_call>',add_special_tokens=False)+[tok.eos_token_id]
result['actual_I_raw_file']=dict(path=str(raw_file),sha256=file_hash(raw_file),prefix_ids=actual_prefix,prefix_text=tok.decode(actual_prefix,skip_special_tokens=False),recorded_response=actual_response)
result['diagnostic_script_sha256']=file_hash(Path(__file__))
started=time.monotonic();torch.cuda.set_device(0)
uuid=str(torch.cuda.get_device_properties(0).uuid)
assert uuid.removeprefix('GPU-') in {v['uuid'].removeprefix('GPU-') for v in leases},uuid
result['actual_cuda0_uuid']=uuid;result['cuda_visible_devices']=os.environ['CUDA_VISIBLE_DEVICES']
base=root.parent/'assets-root/assets/qwen2.5-3b'
result['base_hf_sha256']={p.name:file_hash(p) for p in base.iterdir() if p.is_file() and (p.suffix in ['.safetensors','.json'] or p.name=='merges.txt')}
for path,dtype,attention in [(base,torch.float32,'sdpa'),(model_path,torch.float32,'sdpa'),(model_path,torch.bfloat16,'sdpa'),(model_path,torch.bfloat16,'flash_attention_2')]:
 model=AutoModelForCausalLM.from_pretrained(path,torch_dtype=dtype,attn_implementation=attention,local_files_only=True).to('cuda:0');model.eval()
 torch.cuda.reset_peak_memory_stats()
 for prompt_label,prefix,target in [('training_first_search',train_prefix,train_target),('actual_I_first_prompt',actual_prefix,actual_target)]:
  ids=torch.tensor([prefix+target],device='cuda:0');mask=torch.ones_like(ids);p=len(prefix)
  with torch.inference_mode():
   logits=model(ids,attention_mask=mask,use_cache=False).logits
   probes=[]
   for index,token in enumerate(target):
    if token not in [151657,151658]:continue
    scores=logits[0,p+index-1].float();lp=torch.log_softmax(scores,dim=-1)
    values,indices=torch.topk(scores,10)
    probes.append(dict(token_id=token,position=index,token_rank=int((scores>scores[token]).sum().item())+1,log_probability=float(lp[token]),top10=[dict(id=int(i),text=tok.decode([int(i)],skip_special_tokens=False),logit=float(v)) for i,v in zip(indices,values)]))
   del logits,ids,mask,scores,lp,values,indices
   prompt=torch.tensor([prefix],device='cuda:0')
   out=model.generate(prompt,attention_mask=torch.ones_like(prompt),do_sample=False,max_new_tokens=256,use_cache=True,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
   generated=out[0,len(prefix):].tolist()
  result['cases'].append(dict(model_path=str(path),prompt_label=prompt_label,dtype=str(dtype),attention=attention,teacher_forced_tag_probes=probes,greedy_ids=generated,greedy_text=tok.decode(generated,skip_special_tokens=False),peak_allocated=torch.cuda.max_memory_allocated(),cuda_uuid=str(torch.cuda.get_device_properties(0).uuid)))
  (run/'result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
 del model,prompt,out;gc.collect();torch.cuda.empty_cache()
result['wall_seconds']=time.monotonic()-started;(run/'result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n');print(json.dumps(result,ensure_ascii=False),flush=True)

(run/"driver-exit.json").write_text(json.dumps(dict(returncode=0,job=os.environ["SLURM_JOB_ID"],real_rl_updates=0,training_steps=0)))
