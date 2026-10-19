"""Eight independent vLLM inference workers, isolated from every live trainer.

One checkpoint, exactly 51,713 fixed questions, greedy at every call, four
interactions plus one final generation. No optimizer or training module exists.
"""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import time
import urllib.request
from .protocol import PROFILE, SCIENCE, TOTAL, answer, action, em, strict_format, postprocess, passages, INVALID, digest, sha, validate_rows, metrics

def atomic(path,data):
 path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(path)

def once(path,data):
 with Path(path).open('x') as f:
  json.dump(data,f,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())

def verify(plan):
 assert plan['profile']==PROFILE and plan['science']==SCIENCE
 assert plan['kind'] in ('base','searchr1') and plan['repeat'] in (1,2,3)
 assert Path(plan['root']).name==plan['kind']+'-r'+str(plan['repeat'])
 assert plan['job']=='zhangjunming-7b-v2-'+plan['kind']+'-r'+str(plan['repeat'])
 assert plan['job'].startswith('zhangjunming-7b-v2-')
 for n,h in plan['source_files'].items():assert sha(Path(plan['source'])/n)==h,n
 for n,h in plan['model_files'].items():assert sha(Path(plan['model'])/n)==h,n
 assert sha(plan['data'])==plan['data_sha256']
 with Path(plan['data']).open() as f:rows=[json.loads(s) for s in f]
 validate_rows(rows)
 assert digest([dict(final_index=r['final_index'],question_id=r['question_id'],source=r['source'],gold_answers=r['gold_answers']) for r in rows])==plan['id_gold_identity']
 assert sha(plan['retriever_receipt'])==plan['retriever_receipt_sha256']
 return rows

def search(url,queries):
 payload=json.dumps(dict(queries=queries,topk=3,return_scores=True)).encode()
 req=urllib.request.Request(url,data=payload,headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=180) as response:raw=response.read()
 data=json.loads(raw)['result'];assert len(data)==len(queries)
 return data,digest(json.loads(raw))

def worker(plan,rank,gpu_uuid,memory_mib):
 os.environ['CUDA_VISIBLE_DEVICES']=gpu_uuid
 for name in ['ROCR_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','RAY_ADDRESS']:
  os.environ.pop(name,None)
 import torch
 from transformers import AutoTokenizer
 from vllm import LLM,SamplingParams
 import importlib.metadata as metadata
 assert torch.cuda.device_count()==1
 assert metadata.version('vllm')=='0.6.3'
 torch.set_grad_enabled(False)
 out=Path(plan['root'])/'output';path=out/f'rank-{rank}.predictions.jsonl'
 tokenizer=AutoTokenizer.from_pretrained(plan['model'],local_files_only=True)
 assert tokenizer.eos_token_id==151645 and tokenizer.pad_token_id==151643
 memory_fraction=.85 if memory_mib<44000 else .65
 llm=LLM(model=plan['model'],tokenizer=plan['model'],tensor_parallel_size=1,dtype='bfloat16',
  enforce_eager=True,gpu_memory_utilization=memory_fraction,max_model_len=4596,
  max_num_batched_tokens=8192,max_num_seqs=128,seed=0,disable_log_stats=True)
 native_model=llm.llm_engine.model_executor.driver_worker.model_runner.model
 parameters=list(native_model.parameters())
 for parameter in parameters:parameter.requires_grad_(False)
 assert not any(v.requires_grad for v in parameters)
 params=SamplingParams(n=1,temperature=0,top_p=1,top_k=-1,max_tokens=500,
  stop=['</search>'],stop_token_ids=[151645,151643],repetition_penalty=1.0,
  presence_penalty=0.0,frequency_penalty=0.0,include_stop_str_in_output=True,skip_special_tokens=True)
 sampling={k:getattr(params,k) for k in ['n','temperature','top_p','top_k','max_tokens','stop','stop_token_ids','repetition_penalty','presence_penalty','frequency_penalty','include_stop_str_in_output']}
 assert sampling['temperature']==0 and sampling['max_tokens']==500
 with Path(plan['data']).open() as f:rows=[json.loads(s) for s in f]
 rows=[r for r in rows if r['final_index']%8==rank]
 once(out/f'rank-{rank}.initialized.json',dict(at=time.time(),rank=rank,pid=os.getpid(),gpu_uuid=gpu_uuid,
  model=plan['model'],model_identity=plan['model_identity'],checkpoint_weight_manifest=plan['model_files'],
  parameters=sum(v.numel() for v in parameters),trainable_parameters=0,optimizer_constructed=False,
  autograd_enabled=torch.is_grad_enabled(),visible_devices=torch.cuda.device_count(),sampling=sampling,
  runtime=dict(vllm=metadata.version('vllm'),torch=metadata.version('torch'),transformers=metadata.version('transformers'),
   gpu_memory_fraction=memory_fraction,max_model_len=4596,max_num_seqs=128,max_num_batched_tokens=8192),
  source_commit=plan['source_commit'],no_training_import_or_update=True))
 completed=raw_total=retained_total=search_total=0
 started=time.time()
 with path.open('x') as output:
  for start in range(0,len(rows),128):
   batch=rows[start:start+128];states=[]
   for r in batch:
    ids=tokenizer.apply_chat_template([dict(role='user',content=r['prompt'])],add_generation_prompt=True,tokenize=True)
    assert len(ids)<=2048
    states.append(dict(row=r,rolling=ids,spans=[],events=[],done=False))
   for turn in range(5):
    active=[s for s in states if not s['done']]
    if not active:break
    inputs=[s['rolling'][-4096:] for s in active]
    generated=llm.generate(prompt_token_ids=inputs,sampling_params=params,use_tqdm=False)
    assert len(generated)==len(active)
    queries=[];query_states=[]
    for s,req,context in zip(active,generated,inputs):
     assert list(req.prompt_token_ids)==context and len(req.outputs)==1
     g=req.outputs[0];text=postprocess(tokenizer.decode(g.token_ids,skip_special_tokens=True))
     token_ids=tokenizer.encode(text,add_special_tokens=False);kind,payload=action(text)
     event=dict(turn=turn+1,final_only=turn==4,model_text=text,action_token_ids=token_ids,
      raw_emitted_tokens=len(g.token_ids),raw_token_ids_sha256=digest(list(g.token_ids)),
      context_token_ids_sha256=digest(context),kind=kind,query=payload if kind=='search' else None,
      actual_retrieval=False,finish_reason=g.finish_reason)
     s['spans'].append(text);s['events'].append(event)
     raw_total+=len(g.token_ids);retained_total+=len(token_ids)
     if kind=='answer':s['done']=True
     elif kind=='search' and turn<4:queries.append(payload);query_states.append(s)
     elif kind=='search':s['done']=True
     else:
      observation=tokenizer.encode(INVALID,add_special_tokens=False)[:500]
      s['rolling']=(s['rolling']+token_ids+observation)[-4096:]
     if kind=='search':s['rolling']=(s['rolling']+token_ids)[-4096:]
    if queries:
     docs,body_sha=search(plan['retriever_url'],queries)
     for s,result in zip(query_states,docs):
      text='\n\n<information>'+passages(result).strip()+'</information>\n\n'
      obs=tokenizer.encode(text,add_special_tokens=False)[:500]
      s['rolling']=(s['rolling']+obs)[-4096:]
      s['events'][-1].update(actual_retrieval=True,retrieval_batch_sha256=body_sha,
       retrieved_documents=len(result),observation_token_ids_sha256=digest(obs))
      search_total+=1
   for s in states:
    r=s['row'];pred=answer(s['spans']);events=s['events']
    result=dict(final_index=r['final_index'],question_id=r['question_id'],source=r['source'],
     gold_answers=r['gold_answers'],prediction=pred,em=em(pred,r['gold_answers']),
     format_valid=int(strict_format(s['spans'])),
     generated_tokens=sum(len(e['action_token_ids']) for e in events),
     raw_emitted_tokens=sum(e['raw_emitted_tokens'] for e in events),
     search_calls=sum(e['actual_retrieval'] for e in events),model_response_spans=s['spans'],
     events=events,rank=rank,gpu_uuid=gpu_uuid,model_identity=plan['model_identity'],protocol=PROFILE)
    output.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n')
   output.flush();os.fsync(output.fileno());completed+=len(states)
   atomic(out/f'rank-{rank}.progress.json',dict(at=time.time(),rank=rank,completed=completed,total=len(rows),
    raw_emitted_tokens=raw_total,generated_tokens=retained_total,search_calls=search_total))
   print('V2_PROGRESS='+json.dumps(dict(rank=rank,completed=completed,total=len(rows))),flush=True)
 assert completed==len(rows)
 assert not any(v.requires_grad for v in parameters)
 once(out/f'rank-{rank}.complete.json',dict(at=time.time(),rank=rank,gpu_uuid=gpu_uuid,completed=completed,
  predictions_sha256=sha(path),raw_emitted_tokens=raw_total,generated_tokens=retained_total,search_calls=search_total,
  optimizer_constructed=False,trainable_parameters=0,seconds=time.time()-started))

def run(planfile):
 planfile=Path(planfile);plan=json.loads(planfile.read_text());rows=verify(plan);root=Path(plan['root'])
 jid=os.environ['SLURM_JOB_ID'];sub=json.loads((root/'submission.json').read_text())
 assert sub['job_id']==jid and sub['plan_sha256']==sha(planfile)
 actual=subprocess.check_output(['scontrol','show','job',jid,'-o'],text=True)
 assert 'JobName='+plan['job']+' ' in actual and 'gres/gpu=8' in actual and 'UserId=jmzhang(1066)' in actual
 gpu_lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total','--format=csv,noheader,nounits'],text=True).splitlines()
 gpu=[v.split(',') for v in gpu_lines];assert len(gpu)==8 and len({v[1].strip() for v in gpu})==8
 assert all(float(v[-1])>=23000 for v in gpu)
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
 once(root/'entry-once.json',dict(at=time.time(),job_id=jid,plan_sha256=sha(planfile),gpu_inventory=gpu_lines))
 out=root/'output';out.mkdir(exist_ok=False);(out/'plan.json').write_bytes(planfile.read_bytes())
 once(out/'evaluation-config.json',dict(science=SCIENCE,profile=PROFILE,model=plan['model'],data_sha256=plan['data_sha256'],
  eight_gpu_data_parallel=True,runtime_batch_size_per_gpu=128,maximum_model_context=4596,
  hardware_memory_fraction_policy={'below44000MiB':.85,'otherwise':.65},
  official_legacy_metadata_sampling_loss_not_reproduced='Greedy is explicitly enforced on all five calls.'))
 ctx=mp.get_context('spawn');processes=[]
 for rank,record in enumerate(gpu):
  proc=ctx.Process(target=worker,args=(plan,rank,record[1].strip(),float(record[-1])));proc.start();processes.append(proc)
 try:
  while any(q.is_alive() for q in processes):
   if any(q.exitcode not in (None,0) for q in processes):raise RuntimeError('A frozen inference worker failed; preserve partial outputs, no automatic retry')
   progress=[json.loads(q.read_text()) for q in out.glob('rank-*.progress.json')]
   atomic(out/'progress.json',dict(at=time.time(),completed=sum(q['completed'] for q in progress),total=TOTAL,job=jid))
   time.sleep(5)
  for q in processes:q.join()
  assert all(q.exitcode==0 for q in processes)
 except BaseException:
  for q in processes:
   if q.is_alive():q.terminate()
  for q in processes:q.join(timeout=30)
  raise
 final=[];workers=[]
 for rank in range(8):
  f=out/f'rank-{rank}.predictions.jsonl';w=json.loads((out/f'rank-{rank}.complete.json').read_text())
  assert sha(f)==w['predictions_sha256'];workers.append(w);final.extend(json.loads(v) for v in f.open())
 final.sort(key=lambda q:q['final_index']);validate_rows(final)
 for expected,observed in zip(rows,final):
  assert all(observed[k]==expected[k] for k in ['final_index','question_id','source','gold_answers'])
  assert observed['em']==em(answer(observed['model_response_spans']),observed['gold_answers'])
 with (out/'predictions.jsonl').open('x') as f:
  for r in final:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
 assert len({w['gpu_uuid'] for w in workers})==8
 # Model payloads are immutable: recheck all file SHA after all workers exit.
 for n,h in plan['model_files'].items():assert sha(Path(plan['model'])/n)==h,n
 summary=dict(at=time.time(),profile=PROFILE,job_id=jid,count=TOTAL,metrics=metrics(final),workers=workers,
  predictions_sha256=sha(out/'predictions.jsonl'),raw_emitted_tokens=sum(w['raw_emitted_tokens'] for w in workers),
  retained_generated_tokens=sum(w['generated_tokens'] for w in workers),retrieval_calls=sum(w['search_calls'] for w in workers),
  model_files_post_sha_verified=True,model_identity=plan['model_identity'],data_sha256=plan['data_sha256'])
 once(out/'summary.json',summary);atomic(out/'progress.json',dict(at=time.time(),completed=TOTAL,total=TOTAL,job=jid))
 once(out/'entry-exit.json',dict(at=time.time(),job_id=jid,exit_code=0,summary_sha256=sha(out/'summary.json')))

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('plan');args=parser.parse_args();run(args.plan)
