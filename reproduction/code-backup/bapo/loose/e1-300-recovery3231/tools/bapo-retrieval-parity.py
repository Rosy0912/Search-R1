#!/usr/bin/env python3
"""Read-only real Search-R1 retrieval parity. No gold labels; no training.
Run CPU/GPU separately, then compare JSON files. GPU modes require existing
Slurm allocation and inherited common preflight leases BEFORE importing torch.
Results describe differences; this program never declares production parity PASS.
"""
import argparse, hashlib, json, math, os, resource, subprocess, sys, time
from pathlib import Path
from types import SimpleNamespace
sys.dont_write_bytecode = True

QUERIES = [
 'What is the capital of France?', 'Who wrote Pride and Prejudice?',
 'What is the largest planet in the Solar System?', 'Where is Mount Everest located?',
 'Who painted the Mona Lisa?', 'What is the chemical symbol for gold?',
 'Which country contains the ancient city of Petra?', 'When did the Apollo 11 mission land on the Moon?',
 'What language is primarily spoken in Brazil?', 'Who developed the theory of general relativity?',
 'Which river flows through Cairo?', 'What is the tallest building in the world?',
 'Who was the first person to win two Nobel Prizes?', 'Which ocean lies between Africa and Australia?']


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()


def identity_queries(q):
 return hashlib.sha256(json.dumps(q,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def peak_ram():
 # Linux KiB; macOS bytes. Includes this process only, not dataset worker peaks.
 value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
 return int(value if sys.platform=='darwin' else value*1024)


def current_ram():
 try:
  for line in Path('/proc/self/status').read_text().splitlines():
   if line.startswith('VmRSS:'):return int(line.split()[1])*1024
 except OSError:pass
 return None


def compare(a,b):
 if a['queries_sha256']!=b['queries_sha256'] or a['queries']!=b['queries']:
  raise ValueError('Different query sets/order')
 if a['recipe']!=b['recipe']:raise ValueError('Different retrieval recipe')
 source_equal=a['asset_identity']==b['asset_identity']
 result=dict(status='DIFFERENCE_REPORT_ONLY',same_asset_identity=source_equal,
  asset_identity_verified=a.get('asset_identity_verified',False) and b.get('asset_identity_verified',False),
  modes=[a['mode'],b['mode']], queries_sha256=a['queries_sha256'], per_query=[],summary={})
 for k in (3,50):
  all_rows=[]
  for n,q in enumerate(a['queries']):
   x,y=a['results'][n][str(k)],b['results'][n][str(k)]
   xi,yi=x['ids'],y['ids'];xs,ys=x['scores'],y['scores']
   if len(xi)!=k or len(yi)!=k:raise ValueError('Incomplete topk')
   if len(set(xi))!=k or len(set(yi))!=k or min(xi+yi)<0:raise ValueError('Duplicate/invalid document IDs')
   if not all(math.isfinite(v) for v in xs+ys):raise ValueError('Nonfinite result scores')
   ux,uy={v:i for i,v in enumerate(xi)},{v:i for i,v in enumerate(yi)}
   common=sorted(set(xi)&set(yi))
   shifts=[dict(id=v,rank_a=ux[v]+1,rank_b=uy[v]+1,
    score_a=xs[ux[v]],score_b=ys[uy[v]],score_abs_delta=abs(xs[ux[v]]-ys[uy[v]]),
    content_hash_equal=x['contents_sha256'][ux[v]]==y['contents_sha256'][uy[v]]) for v in common]
   row=dict(query_index=n,query=q,k=k,exact_rank=xi==yi,overlap_count=len(common),overlap_fraction=len(common)/k,
    max_matched_id_score_abs_delta=max((v['score_abs_delta'] for v in shifts),default=None),
    max_position_score_abs_delta=max(abs(v-w) for v,w in zip(xs,ys)),
    content_hash_equal_for_all_shared=all(v['content_hash_equal'] for v in shifts),
    only_a=sorted(set(xi)-set(yi)),only_b=sorted(set(yi)-set(xi)),shared_rank_score_differences=shifts,
    ids_a=xi,ids_b=yi,scores_a=xs,scores_b=ys)
   all_rows.append(row);result['per_query'].append(row)
  result['summary'][str(k)]=dict(exact_rank_queries=sum(v['exact_rank'] for v in all_rows),queries=len(all_rows),
   mean_overlap_fraction=sum(v['overlap_fraction'] for v in all_rows)/len(all_rows),
   max_matched_id_score_abs_delta=max((v['max_matched_id_score_abs_delta'] for v in all_rows if v['max_matched_id_score_abs_delta'] is not None),default=None))
 result['resources']=[a['resources'],b['resources']]
 return result


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--mode',choices=['cpu','gpu','gpu-encoder-cpu-index','compare'],required=True)
 p.add_argument('--repo',type=Path);p.add_argument('--work-root',type=Path)
 p.add_argument('--index',type=Path);p.add_argument('--corpus',type=Path);p.add_argument('--model',type=Path)
 p.add_argument('--asset-manifest',type=Path);p.add_argument('--hash-large-assets',action='store_true')
 p.add_argument('--queries-json',type=Path,help='Optional fixed list of query strings only, never gold records')
 p.add_argument('--threads',type=int,default=8);p.add_argument('--batch-size',type=int,default=512)
 p.add_argument('--repeats',type=int,default=1);p.add_argument('--embeddings-out',type=Path)
 p.add_argument('--embeddings-in',type=Path,help='Reuse saved embeddings to isolate index quantization (ablation)')
 p.add_argument('--left',type=Path);p.add_argument('--right',type=Path);p.add_argument('--output',type=Path,required=True)
 args=p.parse_args()
 if args.output.exists():p.error('Refuse to overwrite an existing result')
 if args.mode=='compare':
  if not args.left or not args.right:p.error('compare needs --left and --right')
  report=compare(json.loads(args.left.read_text()),json.loads(args.right.read_text()))
 else:
  if not all((args.repo,args.work_root,args.index,args.corpus,args.model)):p.error('run modes need repo/work-root/index/corpus/model')
  if min(args.threads,args.batch_size,args.repeats)<1:p.error('Positive threads/batch/repeats required')
  root=args.work_root.resolve()
  for dest in [args.output,args.embeddings_out]:
   if dest and (root not in dest.resolve().parents):p.error('Run outputs must be under the own work-root')
  args.repo=args.repo.resolve();sys.path.insert(0,str(args.repo))
  gpu=args.mode!='cpu'
  if gpu:
   if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('GPU parity needs own Slurm allocation')
   from search_r1.baselines.preflight import validate_gpu_lease
   validate_gpu_lease(SimpleNamespace(trainer=SimpleNamespace(nnodes=1,n_gpus_per_node=8)))
  else:
   # Ensure CPU encoder/index and any imported library cannot select a CUDA GPU.
   os.environ['CUDA_VISIBLE_DEVICES']=''
  root.mkdir(parents=True,exist_ok=True)
  (root/'tmp').mkdir(exist_ok=True)
  os.environ.update(HF_HOME=str(root/'cache/hf'),HF_DATASETS_CACHE=str(root/'cache/datasets'),
   HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
   PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(root/'tmp'))
  import torch
  import faiss
  import numpy as np
  from search_r1.search.retrieval_server import Config,DenseRetriever,load_docs
  torch.set_num_threads(args.threads);faiss.omp_set_num_threads(args.threads)
  if gpu and (torch.cuda.device_count()!=8 or faiss.get_num_gpus()!=8):raise RuntimeError('GPU overlay and visible allocation must both expose exactly eight GPUs')
  queries=json.loads(args.queries_json.read_text()) if args.queries_json else list(QUERIES)
  if not isinstance(queries,list) or not queries or not all(isinstance(q,str) and q.strip() for q in queries):raise ValueError('Queries must be a nonempty fixed string list, without gold fields')
  assets={};verified=True
  for key,path in [('index',args.index),('corpus',args.corpus)]:
   assets[key]=dict(size=path.stat().st_size)
   if args.hash_large_assets:assets[key]['sha256']=sha(path)
   else:verified=False
  assets['model']={str(f.relative_to(args.model)):sha(f) for f in sorted(args.model.rglob('*')) if f.is_file()}
  if args.asset_manifest:assets['manifest_sha256']=sha(args.asset_manifest)
  # Manifest presence is provenance; independently hashed index/corpus required for verified identity.
  def sync():
   if gpu:torch.cuda.synchronize()
  gpu_samples=[]
  def sample_gpu(phase):
   if gpu:
    leased={r['uuid'] for r in json.loads(os.environ['SEARCH_R1_GPU_LEASES'])}
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    rows=[line for line in raw.splitlines() if line.split(',')[0].strip() in leased]
    if len(rows)!=8:raise RuntimeError('Cannot record all eight allocated UUIDs')
    gpu_samples.append(dict(phase=phase,utc_epoch_seconds=time.time(),allocated_gpu_inventory=rows))
  sample_gpu('before_load')
  start_wall,start_cpu=time.perf_counter(),time.process_time()
  config=Config(retrieval_method='e5',index_path=str(args.index),corpus_path=str(args.corpus),retrieval_topk=50,
   faiss_gpu=args.mode=='gpu',retrieval_model_path=str(args.model),retrieval_pooling_method='mean',
   retrieval_query_max_length=256,retrieval_use_fp16=True,retrieval_batch_size=args.batch_size,
   encoder_device='cuda' if gpu else 'cpu')
  retriever=DenseRetriever(config);sync()
  load_wall,load_cpu=time.perf_counter()-start_wall,time.process_time()-start_cpu
  sample_gpu('after_load')
  if gpu:torch.cuda.reset_peak_memory_stats()
  timings=[];results=[];embeddings=[]
  reused=None
  if args.embeddings_in:
   payload=np.load(str(args.embeddings_in),allow_pickle=False)
   if str(payload['queries_sha256'].item())!=identity_queries(queries):raise ValueError('Embedding query identity differs')
   reused=np.asarray(payload['embeddings'],dtype=np.float32,order='C')
   if reused.shape!=(len(queries),retriever.index.d):raise ValueError('Embedding shape differs')
  for repeat in range(args.repeats):
   rows=[];vectors=[];timer=dict(encode_wall=0.,search_wall=0.,docs_wall=0.)
   t,c=time.perf_counter(),time.process_time()
   for start in range(0,len(queries),args.batch_size):
    qs=queries[start:start+args.batch_size];sync();at=time.perf_counter()
    emb=reused[start:start+len(qs)] if reused is not None else retriever.encoder.encode(qs)
    sync();timer['encode_wall']+=time.perf_counter()-at
    if not np.isfinite(emb).all():raise ValueError('Nonfinite embeddings')
    vectors.append(emb)
    local=[{} for _ in qs]
    for k in (3,50):
     sync();at=time.perf_counter();scores,ids=retriever.index.search(emb,k=k);sync()
     timer['search_wall']+=time.perf_counter()-at
     at=time.perf_counter()
     for i in range(len(qs)):
      if (ids[i]<0).any() or len(set(ids[i].tolist()))!=k or not np.isfinite(scores[i]).all():raise ValueError('Incomplete/invalid topk results')
      docs=load_docs(retriever.corpus,ids[i].tolist())
      contents=[doc['contents'] for doc in docs]
      local[i][str(k)]=dict(ids=[int(v) for v in ids[i]],scores=[float(v) for v in scores[i]],
       corpus_ids=[str(doc.get('id','')) for doc in docs],
       contents_sha256=[hashlib.sha256(v.encode()).hexdigest() for v in contents])
     timer['docs_wall']+=time.perf_counter()-at
    rows.extend(local)
   timer.update(wall=time.perf_counter()-t,process_cpu=time.process_time()-c,query_count=len(queries),rss_bytes=current_ram())
   timings.append(timer)
   sample_gpu('after_repeat_'+str(repeat))
   if repeat==0:results=rows;embeddings=np.concatenate(vectors)
   elif rows!=results:raise RuntimeError('Repeated search changed results; inspect before parity comparison')
  if args.embeddings_out:
   if args.embeddings_out.exists():raise ValueError('Embedding output already exists')
   args.embeddings_out.parent.mkdir(parents=True,exist_ok=True)
   with args.embeddings_out.open('xb') as f:np.savez(f,embeddings=embeddings,queries_sha256=np.array(identity_queries(queries)))
  try:commit=subprocess.check_output(['git','-C',str(args.repo),'rev-parse','HEAD'],text=True).strip()
  except subprocess.CalledProcessError:commit=None
  report=dict(status='MEASUREMENT_ONLY',mode=args.mode,queries=queries,queries_sha256=identity_queries(queries),
   recipe=dict(model='e5',pooling='mean',max_query_length=256,encoder_fp16=True,topk=[3,50],batch_size=args.batch_size),
   asset_identity=assets,asset_identity_verified=verified,results=results,
   embeddings_sha256=hashlib.sha256(embeddings.tobytes()).hexdigest(),
   reused_embeddings=bool(args.embeddings_in),reused_embeddings_file_sha256=sha(args.embeddings_in) if args.embeddings_in else None,
   index_metadata=dict(type=type(retriever.index).__name__,d=int(retriever.index.d),ntotal=int(retriever.index.ntotal),
    gpu_shard=args.mode=='gpu',gpu_use_float16=args.mode=='gpu'),
   source=dict(code_commit=commit,retrieval_server_sha256=sha(args.repo/'search_r1/search/retrieval_server.py'),
    index_path=str(args.index.resolve()),corpus_path=str(args.corpus.resolve()),model_path=str(args.model.resolve()),
    slurm_job_id=os.environ.get('SLURM_JOB_ID'),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
    gpu_leases=json.loads(os.environ.get('SEARCH_R1_GPU_LEASES','[]')) if gpu else [],
    torch_version=torch.__version__,faiss_version=getattr(faiss,'__version__',None),faiss_import_path=faiss.__file__),
   resources=dict(gpu_inventory_samples=gpu_samples,threads=args.threads,load_wall_seconds=load_wall,load_process_cpu_seconds=load_cpu,
    rss_bytes=current_ram(),peak_self_rss_bytes=peak_ram(),measurements=timings,
    gpu_torch_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()) if gpu else 0,
    note='Self RSS/CPU excludes dataset subprocesses; torch allocator excludes FAISS GPU allocations; collect cgroup/nvidia-smi externally. Corpus load and searches preserve original API.'))
 args.output.parent.mkdir(parents=True,exist_ok=True)
 with args.output.open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
 print(str(args.output))

if __name__=='__main__':main()
