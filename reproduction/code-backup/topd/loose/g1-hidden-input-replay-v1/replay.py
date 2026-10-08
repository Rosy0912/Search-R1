"""Readonly original hidden-input tokenizer replay, without model or GPU."""
import argparse,ast,hashlib,json,os,pathlib,time
DUMP_SHA='816052e763ff52f4322932372a58ba0b438710af50db5f21a82631241f246719'
TOKENIZER_SHAS={'merges.txt':'599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3','tokenizer.json':'c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539','tokenizer_config.json':'c91efca15ceff6e9ee9424db58a6f59cd41294e550a86cbd07e3c1fb500b34f9','vocab.json':'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910'}
def need(v,m):
 if not v:raise ValueError(m)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def expression(dump):
 need(sha(dump)==DUMP_SHA,'Original dump source drift')
 tree=ast.parse(pathlib.Path(dump).read_bytes())
 calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='tok' and {k.arg for k in n.keywords}=={'return_tensors','add_special_tokens','padding','truncation','max_length'}]
 need(len(calls)==1,'Original tokenizer call ambiguous')
 # Execute the actual original expression, without the following .to(device).
 return compile(ast.Expression(body=calls[0]),str(dump)+':original-hidden-tokenizer','eval')
def batches(tok,expr,texts,recorded,progress=None):
 need(set(recorded)=={'commit_ctx','ans_ctx','prompts'},'Hidden groups differ original')
 stats={}
 for group in ('commit_ctx','ans_ctx','prompts'):
  need(group in texts,'Missing original group text');offset=0;tokens=0
  for bi,batch in enumerate(recorded[group]):
   ids=batch['input_ids'];mask=batch['attention_mask'];count=len(ids)
   need(1<=count<=8 and count==len(mask),'Actual original batch count invalid')
   need(all(len(r)==len(ids[0]) for r in ids+mask) and 1<=len(ids[0])<=4096,'Actual batch shape')
   chunk=texts[group][offset:offset+count];need(len(chunk)==count,'Recorded batch exceeds actual texts')
   enc=eval(expr,{'tok':tok,'chunk':chunk,'max_len':4096})
   actual={k:enc[k].tolist() for k in ('input_ids','attention_mask')}
   for key in actual:
    need(actual[key]==batch[key],f'Original hidden tokenizer mismatch group={group} batch={bi} field={key}')
   offset+=count;tokens+=sum(sum(r) for r in mask)
   if progress:progress({'group':group,'batch_completed':bi,'rows_checked':offset,'active_tokens_checked':tokens})
  need(offset==len(texts[group]),'Incomplete original hidden text coverage '+group)
  stats[group]={'rows':offset,'batches':len(recorded[group]),'active_tokens':tokens}
 return stats

def run(directory,dump,progress):
 d=pathlib.Path(directory);config=json.loads((d/'frozen-input-config.json').read_text());model=pathlib.Path(config['dump_args']['model']).resolve()
 need(model==pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd/assets-root/assets/qwen2.5-32b'),'Wrong official tokenizer directory')
 identity={n:sha(model/n) for n in TOKENIZER_SHAS}
 need(identity==TOKENIZER_SHAS and all(config['model_files'][n]==h for n,h in identity.items()),'Official tokenizer content identity changed')
 expr=expression(dump)
 import torch
 torch.set_num_threads(2);torch.set_num_interop_threads(2)
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(model,padding_side='left',local_files_only=True,trust_remote_code=False)
 if tok.pad_token is None:tok.pad_token=tok.eos_token
 need(tok.is_fast and tok.eos_token_id==151643 and tok.padding_side=='left','Actual tokenizer EOS/padding differs')
 trace=json.loads((d/'generation-trace.json').read_text());recorded=json.loads((d/'hidden-input-token-trace.json').read_text())
 need(trace['nq']==300 and len(trace['questions'])==300 and len(set(trace['questions']))==300 and len(trace['gens'])==1500,'Original300/G5 trace shape')
 need(trace['valid']==[i for i,g in enumerate(trace['gens']) if '<answer>' in g],'Original valid filter mismatch')
 need(len(trace['commit_ctx'])==len(trace['ans_ctx'])==len(trace['valid']) and len(trace['prompts'])==300,'Original hidden group coverage')
 stats=batches(tok,expr,trace,recorded,progress)
 need(not torch.cuda.is_initialized(),'Unexpected CUDA initialization')
 return {'status':'HIDDEN_INPUT_EXACT_MATCH','groups':stats,'valid_rollouts':len(trace['valid']),'tokenizer_files':identity,'original_dump_sha256':sha(dump),'raw_refs':{n:sha(d/n) for n in ('frozen-input-config.json','generation-trace.json','hidden-input-token-trace.json')},'actual_torch_threads':torch.get_num_threads(),'actual_torch_interop_threads':torch.get_num_interop_threads(),'cuda_initialized':False,'model_loaded':False,'missing_B_not_accepted':not (d/'phase-B-receipt.json').exists()}
def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--original-dump',required=True);p.add_argument('--output',required=True);p.add_argument('--progress',required=True);a=p.parse_args();need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Explicit CVD empty CPU-only required')
 for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','RAYON_NUM_THREADS'):os.environ[name]='2'
 os.environ['TOKENIZERS_PARALLELISM']='false';os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
 began=time.monotonic();fd=os.open(a.progress,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
 def progress(v):
  v.update(scope='ACTUAL_PARTIAL_HIDDEN_INPUT_CHECK_NOT_ACCEPTANCE',elapsed_seconds=time.monotonic()-began);raw=(json.dumps(v)+'\n').encode();view=memoryview(raw)
  while view:view=view[os.write(fd,view):]
  os.fsync(fd)
 try:r=run(a.directory,a.original_dump,progress)
 except Exception as exc:r={'status':'HIDDEN_INPUT_REPLAY_ERROR','error_type':type(exc).__name__,'error':str(exc)}
 finally:os.close(fd)
 r.update(scope='READ_ONLY_HIDDEN_INPUT_REPLAY_NOT_HIDDEN_VALUE_OR_CALIBRATION_APPROVAL',elapsed_seconds=time.monotonic()-began,reader_sha256=sha(__file__),progress_sha256=sha(a.progress))
 out=os.open(a.output,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
 with os.fdopen(out,'w') as f:json.dump(r,f,indent=2);f.flush();os.fsync(f.fileno())
 print(json.dumps(r))
 if r['status']!='HIDDEN_INPUT_EXACT_MATCH':raise SystemExit(1)
if __name__=='__main__':main()
