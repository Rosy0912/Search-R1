"""Standalone four-replica fixed512 evaluation of the preserved outer200 HF model."""
import os,sys,pathlib,json,time,hashlib,subprocess,fcntl
BASE=pathlib.Path('/data/user/jxu957/experiments/topd-20261008');ROOT=BASE/'runtime'
SOURCE=ROOT/'tools/topd-instruct300-source-v1';sys.path.insert(0,str(SOURCE))
MODEL=ROOT/'logs/topd-hpc3-707690/evidence/checkpoints/actor/global_step_200'
OUT=ROOT/('logs/topd-stage512-four-gpu-'+os.environ['SLURM_JOB_ID'])
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def atomic(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,sort_keys=True));t.replace(p)
def worker(rank):
 import torch
 from transformers import AutoTokenizer
 from vllm import LLM,SamplingParams
 from verl import DataProto
 from verl.utils.dataset.rl_dataset import RLHFDataset,collate_fn
 from torch.utils.data import DataLoader,Subset
 from search_r1.llm_agent.generation import LLMGenerationManager,GenerationConfig
 from search_r1.baselines.evaluation import batch_answer_metrics,aggregate_answer_metrics
 from verl.utils.torch_functional import get_eos_mask
 torch.manual_seed(42)
 tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True);tok.padding_side='left'
 engine=LLM(model=str(MODEL),tokenizer=str(MODEL),dtype='bfloat16',tensor_parallel_size=1,gpu_memory_utilization=.4,enforce_eager=True,max_model_len=4596,max_num_batched_tokens=8192,max_num_seqs=64,enable_prefix_caching=False,seed=42)
 params=SamplingParams(n=1,best_of=1,top_p=1.,top_k=-1,min_p=0.,temperature=0.,max_tokens=500,ignore_eos=False,detokenize=False)
 class Adapter:
  def generate_sequences(self,p):
   ids=p.batch['input_ids'];att=p.batch['attention_mask'];pos=p.batch['position_ids']
   prompts=[x[m.bool()].tolist() for x,m in zip(ids,att)]
   requests=engine.generate(prompt_token_ids=prompts,sampling_params=params,use_tqdm=False)
   tokens=[list(x.outputs[0].token_ids) for x in requests];assert len(tokens)==len(prompts)
   response=torch.full((len(tokens),500),tok.pad_token_id,dtype=torch.long)
   for i,t in enumerate(tokens):response[i,:len(t)]=torch.tensor(t,dtype=torch.long)
   seq=torch.cat([ids,response],-1);rp=pos[:,-1:]+torch.arange(1,501).unsqueeze(0)
   mask=get_eos_mask(response_id=response,eos_token=p.meta_info['eos_token_id'],dtype=att.dtype)
   return DataProto.from_dict({'prompts':ids,'responses':response,'input_ids':seq,'attention_mask':torch.cat([att,mask],-1),'position_ids':torch.cat([pos,rp],-1)})
 dataset=RLHFDataset(parquet_files=str(ROOT/'assets/common-instruct-0b002b/validation.parquet'),tokenizer=tok,prompt_key='prompt',max_prompt_length=4096,filter_prompts=True,return_raw_chat=False,truncation='error');assert len(dataset)==512
 indexes=list(range(rank*128,(rank+1)*128))
 manager=LLMGenerationManager(tok,Adapter(),GenerationConfig(max_turns=4,max_start_length=2048,max_prompt_length=4096,max_response_length=500,max_obs_length=500,num_gpus=1,search_url='http://127.0.0.1:18086/retrieve',topk=3,force_final_answer=True),is_validation=True)
 rows=[];started=time.time()
 for batch_i,bd in enumerate(DataLoader(Subset(dataset,indexes),batch_size=32,shuffle=False,drop_last=False,collate_fn=collate_fn)):
  batch=DataProto.from_single_dict(bd);gen=batch.pop(batch_keys=['input_ids','attention_mask','position_ids']);gen.meta_info={'eos_token_id':tok.eos_token_id,'pad_token_id':tok.pad_token_id,'recompute_log_prob':False,'do_sample':False,'validate':True}
  manager.timing_raw={};result=manager.run_llm_loop(gen_batch=gen,initial_input_ids=gen.batch['input_ids'][:,-2048:].clone());batch=batch.union(result)
  scored=batch_answer_metrics(batch,tok)
  for i,row in enumerate(scored): row['row_id']=indexes[batch_i*32+i]
  rows.extend(scored);atomic(OUT/('rank%d-progress.json'%rank),{'rank':rank,'rows':len(rows),'epoch':time.time(),'seconds':time.time()-started,'metrics':aggregate_answer_metrics(rows)})
  print('RANK_PROGRESS',rank,len(rows),flush=True)
 atomic(OUT/('rank%d-rows.json'%rank),rows)
 atomic(OUT/('rank%d-complete.json'%rank),{'rank':rank,'rows':len(rows),'sha256':sha(OUT/('rank%d-rows.json'%rank)),'seconds':time.time()-started})
def main():
 OUT.mkdir(parents=True,exist_ok=True);lock=open(OUT/'eval.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 receipt=json.loads((MODEL/'resume-complete.json').read_text());assert receipt['outer_step']==200
 verified=[]
 for r in receipt['ranks']+[receipt['driver']]:
  p=pathlib.Path(r['path']);assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'];verified.append({'path':str(p),'sha256':r['sha256']})
 atomic(OUT/'checkpoint200-fresh-sha.json',{'verified':verified,'epoch':time.time(),'hf_files':{p.name:sha(p) for p in MODEL.iterdir() if p.suffix in ['.safetensors','.json'] and not p.name.startswith('resume')}})
 c=json.loads((ROOT/'logs/topd-hpc3-707690/config.json').read_text());cmd=c['retriever_command'].copy();cmd.remove('--faiss_gpu');cmd[cmd.index('--port')+1]='18086';cmd[cmd.index('--threads')+1]='16'
 env=os.environ.copy();visible=env['CUDA_VISIBLE_DEVICES'].split(',');assert len(visible)==4
 env['CUDA_VISIBLE_DEVICES']=visible[3]
 retriever=subprocess.Popen(cmd,env=env,stdout=open(OUT/'retriever.log','w'),stderr=subprocess.STDOUT)
 children=[]
 try:
  import requests
  for _ in range(1200):
   if retriever.poll() is not None:raise RuntimeError('Standalone retriever exited')
   try:
    if requests.get('http://127.0.0.1:18086/docs',timeout=2).status_code==200:break
   except requests.RequestException:pass
   time.sleep(1)
  else:raise TimeoutError('Retriever startup')
  for rank in range(4):
   e=os.environ.copy();e['CUDA_VISIBLE_DEVICES']=visible[rank];e['TRITON_CACHE_DIR']=str(OUT/('cache/triton%d'%rank));e['TMPDIR']=str(OUT/('cache/tmp%d'%rank));pathlib.Path(e['TMPDIR']).mkdir(parents=True,exist_ok=True)
   children.append(subprocess.Popen([sys.executable,'-B',__file__,'--rank',str(rank)],env=e,stdout=open(OUT/('rank%d.log'%rank),'w'),stderr=subprocess.STDOUT))
  while any(p.poll() is None for p in children):
   for p in children:
    if p.poll() not in (None,0):raise RuntimeError('Evaluation worker exited code '+str(p.returncode))
   time.sleep(5)
  assert all(p.returncode==0 for p in children)
  rows=[]
  for rank in range(4): rows.extend(json.loads((OUT/('rank%d-rows.json'%rank)).read_text()))
  assert len(rows)==512 and sorted(r['row_id'] for r in rows)==list(range(512))
  from search_r1.baselines.evaluation import aggregate_answer_metrics
  atomic(OUT/'complete.json',{'status':'FIXED512_ALL_FOUR_GPU_WORKERS_VERIFIED','checkpoint_step':200,'rows':512,'metrics':aggregate_answer_metrics(rows),'rank_receipts':[json.loads((OUT/('rank%d-complete.json'%r)).read_text()) for r in range(4)],'inference':'fixed-HF-four-independent-vllm-TP1-eager','retrieval':'same-E5-fullWiki-CPU-FAISS-GPU-E5','epoch':time.time()})
 finally:
  for p in children+[retriever]:
   if p.poll() is None:p.terminate()
  for p in children+[retriever]:
   try:p.wait(timeout=30)
   except subprocess.TimeoutExpired:p.kill();p.wait()
if __name__=='__main__':
 if '--rank' in sys.argv:worker(int(sys.argv[sys.argv.index('--rank')+1]))
 else:main()
