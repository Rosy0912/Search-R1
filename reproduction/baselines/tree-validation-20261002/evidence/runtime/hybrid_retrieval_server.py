"""Original H100 CPU encoder/corpus/API, only index.search delegated as FP32."""
import argparse,pathlib,json,io,hashlib,os,sys,time,socket
p=argparse.ArgumentParser();p.add_argument('--instance-json',required=True,type=pathlib.Path);p.add_argument('--run',required=True,type=pathlib.Path);p.add_argument('--index-url',default='http://127.0.0.1:18083');p.add_argument('--baseline-json',type=pathlib.Path,default=pathlib.Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/retrieval-2854/cpu.json'));p.add_argument('--port',type=int,default=18082);a=p.parse_args()
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/tree-grpo');assert root in a.run.resolve().parents
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
a.run.mkdir(parents=True,exist_ok=False)
with socket.socket() as s:s.bind(('127.0.0.1',a.port))
import torch
import numpy as np,faiss,requests
import search_r1.search.retrieval_server as rs
assert sys.prefix==str(root/'env') and str(root/'env') in torch.__file__ and str(root/'env') in faiss.__file__ and str(root/'repo') in rs.__file__
assert 'retriever-gpu-lib' not in faiss.__file__
poison=[]
def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
baseline=json.loads(a.baseline_json.read_text())
assert baseline['mode']=='cpu' and baseline['asset_identity_verified']
for key,name in [('index','e5_Flat.index'),('corpus','wiki-18.jsonl')]:assert sha(root/'assets-root/assets/wiki'/name)==baseline['asset_identity'][key]['sha256']
ignored={'model.safetensors.partial','model.safetensors.source.partial','model.safetensors.source.partial.aria2'}
model=root/'assets-root/assets/e5-base-v2'
model_hash={str(f.relative_to(model)):sha(f) for f in model.rglob('*') if f.is_file() and str(f.relative_to(model)) not in ignored}
assert model_hash=={k:v for k,v in baseline['asset_identity']['model'].items() if k not in ignored}
expected=json.loads(a.instance_json.read_text());assert expected['gpu_uuid']=='GPU-c63282c6-1d05-224e-234b-2a8c794b1f04' and expected['mode']=='fp32-index-only' and expected['index_float16'] is False
assert expected['ntotal']==21015324 and expected['dimension']==768 and expected['index_sha256']=='69c98463fdb41fc08737d88513c597725f311c44f7ba6dca4b05d8c7c658d166'
assert a.index_url=='http://127.0.0.1:18083'
response=requests.get(a.index_url+'/provenance',timeout=30);response.raise_for_status();assert response.json()==expected
class IndexProxy:
 d=expected['dimension'];ntotal=expected['ntotal']
 def search(self,q,k):
  if poison:raise RuntimeError('Hybrid index instance is permanently failed')
  try:return self._search(q,k)
  except Exception as error:
   poison.append(type(error).__name__+': '+str(error))
   (a.run/'index-failure.json').write_text(json.dumps({'epoch':time.time(),'failure':poison[0],'scope':'own hybrid retrieval permanently fail closed'})+'\n')
   raise
 def _search(self,q,k):
  if not isinstance(q,np.ndarray) or q.dtype!=np.dtype('float32') or q.ndim!=2 or q.shape[1]!=self.d or not 1<=len(q)<=512 or not np.isfinite(q).all():raise ValueError('Expected production float32 query matrix')
  q=np.ascontiguousarray(q);frame=io.BytesIO();np.save(frame,q,allow_pickle=False);raw=frame.getvalue();t=time.perf_counter()
  reply=requests.post(a.index_url+'/index_search',params={'k':int(k)},data=raw,headers={'Content-Type':'application/x-npy'},timeout=120);reply.raise_for_status();payload=reply.json()
  if payload['instance']!=expected or payload['k']!=k or payload['vectors_sha256']!=hashlib.sha256(q.tobytes()).hexdigest() or payload['request_sha256']!=hashlib.sha256(raw).hexdigest():raise RuntimeError('Index instance or query frame changed')
  raw_ids=payload['ids'];raw_scores=payload['scores']
  if not isinstance(raw_ids,list) or not isinstance(raw_scores,list) or not all(isinstance(row,list) and all(type(v) is int for v in row) for row in raw_ids) or not all(isinstance(row,list) and all(type(v) in (int,float) for v in row) for row in raw_scores):raise RuntimeError('Index response must contain numeric FP32 scores and integer IDs')
  ids=np.asarray(raw_ids);original_scores=np.asarray(raw_scores,dtype=np.float64);scores=np.asarray(raw_scores,dtype=np.float32)
  if not np.array_equal(scores.astype(np.float64),original_scores):raise RuntimeError('Index response silently loses score precision')
  if ids.shape!=(len(q),k) or scores.shape!=ids.shape or ids.dtype.kind not in 'iu' or not np.isfinite(scores).all() or (ids<0).any() or (ids>=self.ntotal).any() or any(len(set(row.tolist()))!=k for row in ids):raise RuntimeError('Invalid exact index result shape/range/finite')
  with (a.run/'index-requests.jsonl').open('a') as f:f.write(json.dumps({'epoch':time.time(),'queries':len(q),'k':k,'vectors_sha256':payload['vectors_sha256'],'request_sha256':payload['request_sha256'],'roundtrip_seconds':time.perf_counter()-t})+'\n')
  return scores,ids.astype(np.int64,copy=False)
torch.set_num_threads(8);faiss.omp_set_num_threads(8)
rs.config=rs.Config(retrieval_method='e5',retrieval_topk=3,index_path=str(root/'assets-root/assets/wiki/e5_Flat.index'),corpus_path=str(root/'assets-root/assets/wiki/wiki-18.jsonl'),faiss_gpu=False,retrieval_model_path=str(root/'assets-root/assets/e5-base-v2'),retrieval_pooling_method='mean',retrieval_query_max_length=256,retrieval_use_fp16=True,retrieval_batch_size=512,encoder_device='cpu')
rs.retriever=rs.DenseRetriever(rs.config);assert rs.retriever.index.d==expected['dimension'] and rs.retriever.index.ntotal==expected['ntotal']
rs.retriever.index=IndexProxy()
original=rs.retriever.encoder.encode
# Hashes bind the exact CPU vectors sent; no tensor/precision/normalization mutation.
def measured_encode(queries,is_query=True):
 t=time.perf_counter();cpu=time.process_time();out=original(queries,is_query=is_query)
 with (a.run/'encoder-requests.jsonl').open('a') as f:f.write(json.dumps({'epoch':time.time(),'queries':len(queries) if not isinstance(queries,str) else 1,'vectors_sha256':hashlib.sha256(out.tobytes()).hexdigest(),'wall_seconds':time.perf_counter()-t,'process_cpu_seconds':time.process_time()-cpu})+'\n')
 return out
rs.retriever.encoder.encode=measured_encode
metadata={'mode':'h100-cpu-encoder-remote-fp32-index','pid':os.getpid(),'run':str(a.run),'source_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'index_instance':expected,'encoder_device':'cpu','encoder_fp16':True,'index_float16':False,'baseline_sha256':sha(a.baseline_json),'model_sha256':model_hash,'asset_identity':baseline['asset_identity'],'retrieval_module_sha256':hashlib.sha256(pathlib.Path(rs.__file__).read_bytes()).hexdigest()}
(a.run/'provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
@rs.app.get('/provenance')
async def provenance():
 if poison:
  from fastapi import HTTPException
  raise HTTPException(503,'Own hybrid index instance has failed')
 return metadata
print('HYBRID_RETRIEVER_READY_CANDIDATE',json.dumps(metadata),flush=True)
import uvicorn
uvicorn.run(rs.app,host='127.0.0.1',port=a.port)
