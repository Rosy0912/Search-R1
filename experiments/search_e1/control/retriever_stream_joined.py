"""Shared full FP32 GPU Flat oracle; unchanged CPU FP32 E5 encoder."""
import os
os.environ['NVIDIA_TF32_OVERRIDE']='0';os.environ['E1_ENCODER_DEVICE']='cpu'
import json,time,threading,hashlib
from pathlib import Path
import numpy as np,torch,faiss,uvicorn
from fastapi import FastAPI,HTTPException
from search_r1.search import retrieval_server as server
from torch_streaming_flat import TorchStreamingFlatIP
base=Path(os.environ['E1_BASE']);torch.set_num_threads(16);faiss.omp_set_num_threads(32)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.set_float32_matmul_precision('highest')
assert torch.cuda.device_count()==1 and not torch.backends.cuda.matmul.allow_tf32
config=server.Config(retrieval_method='e5',retrieval_topk=3,index_path=str(Path('/weka_3/shaoshitong/search-e1-retriever-stream-20261004')/'assets/index/e5_Flat.index'),corpus_path=str(Path('/weka_3/shaoshitong/search-e1-retriever-stream-20261004')/'assets/corpus/wiki-18.jsonl'),faiss_gpu=False,retrieval_model_path=str(Path('/weka_3/shaoshitong/search-e1-retriever-stream-20261004')/'assets/encoder'),retrieval_use_fp16=False,retrieval_batch_size=128)
retriever=server.get_retriever(config);cpu= retriever.index
assert cpu.ntotal==21015324 and cpu.d==768 and len(retriever.corpus)==cpu.ntotal
# Full original FP32 vectors remain on host; bounded tiles use the owned training GPU.
gpu=TorchStreamingFlatIP(cpu,'cuda:0',query_micro=64)
queries=json.loads((base/'receipts/retriever-probe-queries.json').read_text())
emb=retriever.encoder.encode(queries);t=time.monotonic();cs,ci=cpu.search(emb,4);cpu_sec=time.monotonic()-t
t=time.monotonic();gs,gi=gpu.search(emb,4);gpu_sec=time.monotonic()-t
tol=1e-5
checks=[]
for q,a,b,sa,sb in zip(emb,ci,gi,cs,gs):
 # Candidate scores must agree with the original source vectors; different IDs
 # are permitted only inside a numerical tie at their CPU rank boundary.
 exact=np.asarray(cpu.reconstruct_batch(b),np.float32)@q
 if not np.allclose(exact,sb,atol=tol,rtol=0):raise RuntimeError('GPU candidate score drift')
 for rank in range(4):
  if a[rank]!=b[rank] and abs(float(exact[rank])-float(sa[rank]))>tol:
   raise RuntimeError('Non-tie CPU/GPU top4 ID mismatch')
 if not np.allclose(sa,sb,atol=tol,rtol=0):raise RuntimeError('Top4 score drift')
 checks.append(dict(same_ids=bool((a==b).all()),max_score_delta=float(np.max(np.abs(sa-sb)))))
retriever.index=gpu
from joined_scan import install_joined_scan
install_joined_scan(retriever,server.load_docs)
# Exercise 512-query requests concurrently. Lock covers encoding, search and
# document assembly; the second client waits rather than multiplying VRAM use.
lock=threading.Lock();app=FastAPI();timings=base/'receipts/retriever-requests.jsonl'
@app.get('/health')
def health():return dict(ready=True,ntotal=gpu.ntotal,d=gpu.d,encoder='CPU FP32 E5',index='full FP32 streaming FlatIP joined scan',query_micro=64,serialized=True,tf32=False,source_manifest_sha256=receipt['source_manifest_sha256'],acceptance_sha256=hashlib.sha256((base/'receipts/gpu-oracle-acceptance.json').read_bytes()).hexdigest())
@app.post('/retrieve')
def retrieve(request:server.QueryRequest):
 if request.topk is not None and not 1<=request.topk<=4:raise HTTPException(400,'topk must be 1..4')
 if len(request.queries)>2560:raise HTTPException(400,'At most 2560 queries per request')
 if not request.queries:return {'result':[]}
 start=time.monotonic()
 with lock:
  entered=time.monotonic();results,scores=retriever.batch_search(request.queries,num=request.topk or 3,return_score=True)
  out=[[{'document':doc,'score':score} for doc,score in zip(docs,ss)] if request.return_scores else docs for docs,ss in zip(results,scores)]
  with timings.open('a') as f:f.write(json.dumps(dict(queries=len(request.queries),wait_seconds=entered-start,service_seconds=time.monotonic()-entered))+'\n')
 return {'result':out}
from concurrent.futures import ThreadPoolExecutor
for i in range(1):torch.cuda.reset_peak_memory_stats(i)
request=server.QueryRequest(queries=(queries*32)[:512],topk=3,return_scores=True)
with ThreadPoolExecutor(2) as pool:outputs=list(pool.map(retrieve,[request,request]))
assert outputs[0]==outputs[1] and len(outputs[0]['result'])==512
peaks=[dict(device=i,allocated=torch.cuda.max_memory_allocated(i),reserved=torch.cuda.max_memory_reserved(i),total=torch.cuda.get_device_properties(i).total_memory) for i in range(1)]
if any(p['reserved']>512*1024**2 for p in peaks):raise RuntimeError('Streaming retriever allocator exceeds 512MiB')
receipt=dict(cpu_seconds=cpu_sec,gpu_seconds=gpu_sec,queries=queries,cpu_ids=ci.tolist(),gpu_ids=gi.tolist(),cpu_scores=cs.tolist(),gpu_scores=gs.tolist(),checks=checks,score_atol=tol,tf32=False,encoder='unchanged CPU FP32 E5',shard_bounds=gpu.bounds,ntotal=gpu.ntotal,d=gpu.d,concurrent_512_requests=2,peaks=peaks,source_manifest_sha256=hashlib.sha256((base/'assets/manifest.json').read_bytes()).hexdigest())
(base/'receipts/gpu-oracle-acceptance.json').write_text(json.dumps(receipt,indent=2))
print('FULL_FP32_GPU_ORACLE_ACCEPTED',json.dumps(peaks),flush=True)
uvicorn.run(app,host='10.123.4.11',port=18003)
