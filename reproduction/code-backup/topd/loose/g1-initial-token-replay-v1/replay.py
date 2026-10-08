"""Read-only tokenizer/original-grader replay; no model/GPU or calibration approval."""
import argparse,ast,hashlib,importlib.metadata,json,os,pathlib,re,time,types,__future__
DUMP_SHA='816052e763ff52f4322932372a58ba0b438710af50db5f21a82631241f246719'
DECODE_SHA='2630ec7effb6cac44837ab914c91196b6c8bc68970cd374eceb02aaca1f1c27b'
STOP_SHA='03ec8586e92de0fff3eaa6066fff53885dea3084e22ba830c9ad56859c9b8e7b'
TOKENIZER_SHAS={'merges.txt':'599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3','tokenizer.json':'c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539','tokenizer_config.json':'c91efca15ceff6e9ee9424db58a6f59cd41294e550a86cbd07e3c1fb500b34f9','vocab.json':'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910'}
def need(ok,text):
 if not ok:raise ValueError(text)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(pathlib.Path(p).read_text())
def grader_functions(dump):
 need(sha(dump)==DUMP_SHA,'Original grader source drift')
 ns={'re':re};tree=ast.parse(pathlib.Path(dump).read_bytes())
 nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('normalize','extract_answer','em_score','contains_gold')]
 nodes += [n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ANSWER_RE' for t in n.targets)]
 need(len(nodes)==5,'Original grader AST schema')
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(dump),'exec',flags=__future__.annotations.compiler_flag),ns)
 return ns

def original_functions(dump,decode,stop):
 need(sha(decode)==DECODE_SHA and sha(stop)==STOP_SHA,'Original detokenizer/stop source drift')
 ns=grader_functions(dump)
 dt=ast.parse(pathlib.Path(decode).read_bytes());dn=[n for n in dt.body if isinstance(n,ast.FunctionDef)]
 dn += [n for n in dt.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET' for t in n.targets)]
 exec(compile(ast.Module(body=dn,type_ignores=[]),str(decode),'exec',flags=__future__.annotations.compiler_flag),ns)
 st=ast.parse(pathlib.Path(stop).read_bytes());cl=next(n for n in st.body if isinstance(n,ast.ClassDef) and n.name=='StopChecker');fn=next(n for n in cl.body if isinstance(n,ast.FunctionDef) and n.name=='_check_stop_strings');fn.decorator_list=[]
 exec(compile(ast.Module(body=[fn],type_ignores=[]),str(stop),'exec',flags=__future__.annotations.compiler_flag),ns)
 return ns

def decode_original(tok,row,sampling,functions):
 prompt=row['prompt_token_ids'];ids=row['token_ids'];allids=list(prompt);tokens,prefix,read=functions['convert_prompt_ids_to_tokens'](tok,prompt,skip_special_tokens=True);seq=types.SimpleNamespace(output_text='');stopped=None
 params=types.SimpleNamespace(stop=sampling['stop'],include_stop_str_in_output=sampling['include_stop_str_in_output'])
 for index,token in enumerate(ids):
  allids.append(token);new,text,prefix,read=functions['detokenize_incrementally'](tok,allids,tokens,prefix,read,skip_special_tokens=True,spaces_between_special_tokens=True);tokens.extend(new);seq.output_text+=text
  if token==tok.eos_token_id:stopped='EOS'
  else:stopped=functions['_check_stop_strings'](seq,len(text),params)
  if stopped is not None:
   need(index==len(ids)-1,'Actual output contains tokens after original stop');break
 return seq.output_text,stopped

def labels(trace,calls,functions):
 need(trace['nq']==300 and len(trace['gens'])==1500,'actual300/G5 shape')
 normalize=functions['normalize'];extract=functions['extract_answer'];em=functions['em_score'];contains=functions['contains_gold']
 need([r['text'] for r in calls[0]['outputs']]==trace['cb_ans'] and calls[0]['prompts']==trace['cb_prompt'],'closedbook original call alignment')
 cb=[contains(t,g) for t,g in zip(trace['cb_ans'],trace['golds_all'])];need(cb==trace['q_cb'],'original closedbook labels differ')
 valid=[j for j,g in enumerate(trace['gens']) if '<answer>' in g];need(valid==trace['valid'],'original valid filter differs')
 answers=[];ems=[];commit=[];ansctx=[];flipctx=[];flip2=[]
 for j in valid:
  g=trace['gens'][j];qi=trace['flat_qi'][j];mi=g.find('<answer>');ae=g.find('</answer>',mi);ae=len(g) if ae<0 else ae;answer=extract(g)
  answers.append(answer);ems.append(em(answer,trace['golds_all'][qi]));commit.append(trace['prompts'][qi]+g[:mi+8]);ansctx.append(trace['prompts'][qi]+g[:ae]);empty=re.sub(r'<information>.*?</information>','<information></information>',g[:mi+8],flags=re.DOTALL);flipctx.append(trace['prompts'][qi]+empty);flip2.append(trace['prompts'][qi]+re.sub(r'<think>.*?</think>','',empty,flags=re.DOTALL))
 for key,value in [('answers',answers),('ems',ems),('commit_ctx',commit),('ans_ctx',ansctx),('flip_ctx',flipctx),('flip2_ctx',flip2)]:need(trace[key]==value,'original generation assembly differs: '+key)
 need(calls[-2]['prompts']==flipctx and calls[-1]['prompts']==flip2,'counterfactual call context differs')
 first=[r['text'] for r in calls[-2]['outputs']];second=[r['text'] for r in calls[-1]['outputs']];need(first==trace['flip_ans'],'actual flip answers trace differs')
 def flips(raw):return [int(normalize(extract(v+'</answer>') or v.split('</answer>')[0])!=normalize(answers[k])) for k,v in enumerate(raw)]
 need(flips(first)==trace['flips'] and flips(second)==trace['flips2'],'original counterfactual labels differ')
 return {'closedbook_questions':len(cb),'closedbook_correct':sum(cb),'valid_rollouts':len(valid),'em_correct':sum(ems),'flip_changed':sum(trace['flips']),'flip2_changed':sum(trace['flips2']),'all_original_contexts_and_labels_identical':True}

def run(directory,dump):
 d=pathlib.Path(directory).resolve();config=load(d/'frozen-input-config.json');model=pathlib.Path(config['dump_args']['model']).resolve()
 need(model==pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd/assets-root/assets/qwen2.5-32b'),'Wrong official tokenizer directory')
 actual={name:sha(model/name) for name in TOKENIZER_SHAS};need(actual==TOKENIZER_SHAS and all(config['model_files'][k]==v for k,v in actual.items()),'Official complete tokenizer identity drift')
 need(importlib.metadata.version('vllm')=='0.6.3','Wrong native vLLM version')
 base=pathlib.Path(importlib.metadata.distribution('vllm').locate_file('vllm'));decode=base/'transformers_utils/detokenizer.py';stop=base/'engine/output_processor/stop_checker.py';functions=original_functions(dump,decode,stop)
 # Inspect the real frozen defaults; never import vLLM or initialize CUDA.
 params=base/'sampling_params.py';tree=ast.parse(params.read_bytes());cl=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SamplingParams');defaults={n.target.id:ast.literal_eval(n.value) for n in cl.body if isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name) and n.target.id in ('skip_special_tokens','spaces_between_special_tokens','ignore_eos')}
 need(defaults=={'skip_special_tokens':True,'spaces_between_special_tokens':True,'ignore_eos':False},'Native original sampling defaults drift')
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(model,local_files_only=True,trust_remote_code=False);need(tok.is_fast and tok.eos_token_id==151643,'Official tokenizer runtime type/EOS differs')
 calls=load(d/'generation-token-trace.json');trace=load(d/'generation-trace.json');mismatches=[];rows=0;tokens=0;prompt_mismatches=[]
 for ci,call in enumerate(calls):
  need(len(call['outputs'])==len(call['prompts']),'generation call rows differ')
  for ri,(prompt,row) in enumerate(zip(call['prompts'],call['outputs'])):
   rows+=1;tokens+=len(row['token_ids']);expected_prompt=tok.encode(prompt,add_special_tokens=True)
   if expected_prompt!=row['prompt_token_ids']:prompt_mismatches.append({'call':ci,'row':ri,'expected_sha256':hashlib.sha256(json.dumps(expected_prompt).encode()).hexdigest(),'actual_sha256':hashlib.sha256(json.dumps(row['prompt_token_ids']).encode()).hexdigest()})
   text,reason=decode_original(tok,row,call['sampling'],functions)
   if text!=row['text']:mismatches.append({'call':ci,'row':ri,'actual_text_sha256':hashlib.sha256(row['text'].encode()).hexdigest(),'replay_text_sha256':hashlib.sha256(text.encode()).hexdigest(),'actual_chars':len(row['text']),'replayed_chars':len(text),'stop':reason})
 label_result=labels(trace,calls,functions)
 return {'scope':'ORIGINAL_TOKENIZER_GRADER_REPLAY_ONLY_NOT_CALIBRATION_ACCEPTANCE','status':'REPLAY_MATCH' if not mismatches and not prompt_mismatches else 'REPLAY_CONTRADICTION','generated_output_rows':rows,'output_tokens':tokens,'mismatch_count':len(mismatches),'mismatch_examples':mismatches[:12],'prompt_mismatch_count':len(prompt_mismatches),'prompt_mismatch_examples':prompt_mismatches[:12],'labels':label_result,'tokenizer_files':actual,'native_defaults':defaults,'native_versions':{n:importlib.metadata.version(n) for n in ('vllm','transformers','tokenizers')},'raw_refs':{n:sha(d/n) for n in ('frozen-input-config.json','generation-token-trace.json','generation-trace.json')},'source_refs':{str(p):sha(p) for p in (pathlib.Path(dump),decode,stop,params)},'missing_B_not_accepted':not (d/'phase-B-receipt.json').exists(),'cuda_initialized':False,'vllm_imported':False}

def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--original-dump',required=True);p.add_argument('--output',required=True);a=p.parse_args();need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Explicit CPU-only CUDA_VISIBLE_DEVICES empty required');start=time.monotonic()
 try:result=run(a.directory,a.original_dump)
 except Exception as exc:result={'scope':'ORIGINAL_TOKENIZER_GRADER_REPLAY_ONLY_NOT_CALIBRATION_ACCEPTANCE','status':'REPLAY_ERROR','error':repr(exc),'raw_refs':{n:sha(pathlib.Path(a.directory)/n) for n in ('frozen-input-config.json','generation-token-trace.json','generation-trace.json') if (pathlib.Path(a.directory)/n).is_file()}}
 result.update(elapsed_seconds=time.monotonic()-start,reader_sha256=sha(__file__))
 import sys
 result['vllm_imported']='vllm' in sys.modules;result['cuda_initialized']=bool(sys.modules['torch'].cuda.is_initialized()) if 'torch' in sys.modules else False;need(not result['cuda_initialized'] and not result['vllm_imported'],'Reader unexpectedly initialized GPU/imported vLLM')
 fd=os.open(a.output,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
 with os.fdopen(fd,'w') as f:json.dump(result,f,indent=2,sort_keys=True);f.flush();os.fsync(f.fileno())
 print_record={k:result.get(k) for k in ('status','generated_output_rows','mismatch_count','prompt_mismatch_count','labels','error')}
 print(json.dumps(print_record))
 if result['status']!='REPLAY_MATCH':raise SystemExit(1)
if __name__=='__main__':main()
