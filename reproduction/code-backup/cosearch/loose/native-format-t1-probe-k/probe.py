"""Independent native HF T1 format check on 16 retained-train prompts;0RL."""
import os,json,hashlib,time,gc
from pathlib import Path
root=Path('/dataset/shaoshitong/search-r1-rl/cosearch/author-native')
assert os.environ['SLURM_JOB_NODELIST']=='h100-16' and os.environ['SEARCH_R1_GPU_PREFLIGHT']=='1'
leases=json.loads(os.environ['SEARCH_R1_GPU_LEASES']);assert len({x['uuid'] for x in leases})==8
for k in ['ROCR_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL']:os.environ.pop(k,None)
import sys,torch,asyncio
from transformers import AutoTokenizer,AutoModelForCausalLM
sys.path[:0]=[str(root/'repo'),str(root/'repo/verl')]
from verl.experimental.agent_loop.tool_parser import SearchR1ToolParser
from verl.experimental.agent_loop.co_search_agent_loop import CoSearchAgentLoop
from types import SimpleNamespace
run=root/'runs/author-native-format-t1-probe-k';run.mkdir(exist_ok=False)
def file_hash(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
source=root/'runs/author-native-grammar-actor-continue-v3'
assert json.loads((source/'driver-exit.json').read_text())['returncode']==0
manifest=json.loads((source/'warm-start/manifest.json').read_text());assert manifest['completed'] and manifest['roles']['actor']['supervised_optimizer_steps']==32 and manifest['ranker_additional_supervised_updates']==0
model_path=source/'warm-start/actor'
for n,expected in manifest['roles']['actor']['files_sha256'].items():
 if n!='optimizer.pt':assert file_hash(model_path/n)==expected
raw_file=sorted((root/'runs/author-native-warm-five-i/evidence/raw-rollouts').glob('pre-postprocess-696092-*.pt'))[0]
raw=torch.load(raw_file,map_location='cpu',weights_only=False)
tok=AutoTokenizer.from_pretrained(model_path);parser=SearchR1ToolParser(tok)
import pandas as pd
train_file=root/'data/train.parquet';val_file=root/'data/validation-64.parquet'
train_prompts={list(p)[-1]['content'] for p in pd.read_parquet(train_file,columns=['prompt']).prompt}
val_prompts={list(p)[-1]['content'] for p in pd.read_parquet(val_file,columns=['prompt']).prompt}
corpus_file=root/'native-grammar-warm-start-fulltext-v2/corpus.json'
corpus=json.loads(corpus_file.read_text())
assert file_hash(corpus_file)==json.loads((corpus_file.parent/'manifest.json').read_text())['corpus_sha256']
warm_prefixes={tuple(r['input_ids'][:-r['response_length']]) for r in corpus['actor'][::2]}
questions=[];seen=set()
for row in raw:
 ids=row.prompt_ids.reshape(-1).tolist()
 while ids and ids[0]==tok.pad_token_id:ids.pop(0)
 digest=hashlib.sha256(json.dumps(ids).encode()).hexdigest()
 if digest in seen or tuple(ids) in warm_prefixes:continue
 text=tok.decode(ids,skip_special_tokens=False);content=text.split('<|im_start|>user\n',1)[1].split('<|im_end|>',1)[0]
 assert content in train_prompts and content not in val_prompts,'Format diagnostic must use retained TRAIN only, excluding val64'
 seen.add(digest);questions.append(dict(prefix_ids=ids,prefix_text=tok.decode(ids,skip_special_tokens=False),sha256=digest))
 if len(questions)==16:break
assert len(questions)==16
# These exact prefixes come from the original retained TRAIN rollout; no gold or scores are used.
model=AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16,attn_implementation='flash_attention_2',local_files_only=True).to('cuda:0');model.eval()
uuid=str(torch.cuda.get_device_properties(0).uuid);assert uuid.removeprefix('GPU-') in {v['uuid'].removeprefix('GPU-') for v in leases}
result=dict(run=run.name,job=os.environ['SLURM_JOB_ID'],leases=leases,actual_cuda0_uuid=uuid,source_manifest_sha256=file_hash(source/'warm-start/manifest.json'),raw_file_sha256=file_hash(raw_file),question_prefixes=questions,train_parquet_sha256=file_hash(train_file),val64_sha256=file_hash(val_file),format_questions_excluded_from_warm128=True,real_rl_updates=0,supervised_steps=0,HF_not_vLLM_parity=True,sampling=dict(temperature=1.,top_p=1.,top_k=0,top_k_native_equivalent=-1,repetition_penalty=1.,max_new_tokens=4096,n=8,seed=0),rows=[],cost_excludes_RL=True)
torch.manual_seed(0);started=time.monotonic()
with torch.inference_mode(),(run/'outputs.jsonl').open('w') as stream:
 for qi,q in enumerate(questions):
  prompt=torch.tensor([q['prefix_ids']],device='cuda:0')
  output=model.generate(prompt,attention_mask=torch.ones_like(prompt),do_sample=True,temperature=1.,top_p=1.,top_k=0,repetition_penalty=1.,max_new_tokens=4096,num_return_sequences=8,use_cache=True,eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id)
  for si,out in enumerate(output):
   full_ids=out[len(q['prefix_ids']):].tolist()
   eos_index=full_ids.index(tok.eos_token_id) if tok.eos_token_id in full_ids else None
   ids=full_ids[:eos_index+1] if eos_index is not None else full_ids
   text=tok.decode(ids,skip_special_tokens=False)
   _,calls=asyncio.run(parser.extract_tool_calls(ids));query=None;error=None
   try:
    if len(calls)==1:query=json.loads(calls[0].arguments).get('query')
   except Exception as e:error=str(e)
   answer=CoSearchAgentLoop.detect_answer(SimpleNamespace(tokenizer=tok),ids)
   budget_exhausted=len(ids)>=4096
   process_tool=not budget_exhausted and not answer and len(calls)==1
   query_is_string=isinstance(query,str);query_nonempty=query_is_string and bool(query.strip())
   valid=process_tool and calls[0].name=='search' and query_nonempty
   row=dict(question_index=qi,sample=si,response_ids=ids,full_generated_ids_with_padding=full_ids,text=text,actual_original_parser_calls=len(calls),query_is_string=query_is_string,query_nonempty=query_nonempty,actual_detect_answer=answer,eos_inclusive_generated_length=len(ids),budget_exhausted=budget_exhausted,first_assistant_turn=1,max_assistant_turns=6,native_initial_would_process_tool=process_tool,valid_search_tool_call=valid,parser_error=error)
   stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush();result['rows'].append({k:v for k,v in row.items() if k not in ['response_ids','full_generated_ids_with_padding','text']})
  (run/'result.json').write_text(json.dumps(result,indent=2)+'\n');del prompt,output
result.update(valid_calls=sum(v['valid_search_tool_call'] for v in result['rows']),native_initial_process_tool_count=sum(v['native_initial_would_process_tool'] for v in result['rows']),total_samples=len(result['rows']),wall_seconds=time.monotonic()-started,peak_bytes=torch.cuda.max_memory_allocated(),outputs_sha256=file_hash(run/'outputs.jsonl'))
assert len(result['rows'])==128
assert all(p.grad is None for p in model.parameters())
(run/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
(run/'driver-exit.json').write_text(json.dumps(dict(returncode=0,job=os.environ['SLURM_JOB_ID'],real_rl_updates=0,supervised_steps=0)))
