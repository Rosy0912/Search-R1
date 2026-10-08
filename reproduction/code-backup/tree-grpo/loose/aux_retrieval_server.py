"""Own auxiliary retriever: unchanged CPU E5 recipe, one FP32 GPU flat index."""
import argparse,pathlib,os,sys,json,time,hashlib,resource,threading,signal,socket
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--port',type=int,default=18082);p.add_argument('--max-seconds',type=int,default=14400);a=p.parse_args()
root=pathlib.Path('/data/shaoshitong/search-r1-rl/tree-grpo-retrieval');run=root/'runs'/a.run
assert run.resolve().parent==(root/'runs').resolve() and a.max_seconds>0
run.mkdir(parents=True,exist_ok=False)
with socket.socket() as check:check.bind(('127.0.0.1',a.port))
expected='GPU-c63282c6-1d05-224e-234b-2a8c794b1f04'
assert os.environ['CUDA_VISIBLE_DEVICES']==expected
from search_r1.baselines.preflight import validate_gpu_lease
from types import SimpleNamespace
validate_gpu_lease(SimpleNamespace(trainer=SimpleNamespace(nnodes=1,n_gpus_per_node=1)))
allocation=json.loads((root/'logs/fp32-index-20261003-0754-dispatched.json').read_text())
assert pathlib.Path('/proc/self/cgroup').read_text()==allocation['platform_cgroup'] and 'cri-containerd-' in allocation['platform_cgroup']
start=time.time();samples=[];stop=threading.Event();ready=threading.Event();violations=[]
def watch():
 try:
  import pynvml as nv
  nv.nvmlInit();handle=nv.nvmlDeviceGetHandleByUUID(expected.encode())
  while not stop.is_set():
   mem=nv.nvmlDeviceGetMemoryInfo(handle);util=nv.nvmlDeviceGetUtilizationRates(handle)
   procs={v.pid:v for v in nv.nvmlDeviceGetComputeRunningProcesses(handle)+nv.nvmlDeviceGetGraphicsRunningProcesses(handle)}
   foreign=[{'pid':int(v.pid),'memory_bytes':int(v.usedGpuMemory)} for v in procs.values() if v.pid!=os.getpid()]
   sample={'epoch':time.time(),'gpu_uuid':expected,'used_bytes':mem.used,'total_bytes':mem.total,'gpu_util':util.gpu,'foreign':foreign};samples.append(sample)
   with (run/'gpu-samples.jsonl').open('a') as out:out.write(json.dumps(sample)+'\n')
   if foreign:raise RuntimeError('Foreign process appeared on selected auxiliary GPU; end own retriever only')
   if time.time()-start>a.max_seconds:raise RuntimeError('Own bounded service lifetime reached')
   ready.set();stop.wait(1)
  nv.nvmlShutdown()
 except BaseException as err:
  violations.append(repr(err));ready.set();os.kill(os.getpid(),signal.SIGTERM)
def terminate(signum,frame):raise SystemExit(128+signum)
signal.signal(signal.SIGTERM,terminate)
thread=threading.Thread(target=watch,daemon=True)
def healthy():
 if violations or not thread.is_alive() or not samples or time.time()-samples[-1]['epoch']>5:raise RuntimeError('Aux GPU guard unavailable or stale')
def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
try:
 thread.start();assert ready.wait(10);healthy()
 import torch
 import numpy as np,faiss
 import search_r1.search.retrieval_server as rs
 assert sys.prefix==str(root/'env') and str(root/'env') in torch.__file__ and str(root/'retriever-gpu-lib') in faiss.__file__
 assert torch.cuda.device_count()==faiss.get_num_gpus()==1
 torch.set_num_threads(8);faiss.omp_set_num_threads(8)
 baseline=json.loads((root/'inputs/retrieval-cpu-2854.json').read_text());verified=json.loads((root/'logs/assets-verified.json').read_text())
 assert verified['e5_Flat.index']['sha256']==baseline['asset_identity']['index']['sha256']
 assert verified['wiki-18.jsonl']['sha256']==baseline['asset_identity']['corpus']['sha256']
 for rec in verified.values():
  st=pathlib.Path(rec['path']).stat();assert (st.st_size,st.st_mtime_ns,st.st_ino)==(rec['size'],rec['mtime_ns'],rec['inode'])
 model=pathlib.Path('/data/search-r1/assets/e5-base-v2')
 model_hash={str(f.relative_to(model)):sha(f) for f in sorted(model.rglob('*')) if f.is_file()}
 excluded_download_artifacts={'model.safetensors.partial','model.safetensors.source.partial','model.safetensors.source.partial.aria2'}
 expected_model_hash={name:value for name,value in baseline['asset_identity']['model'].items() if name not in excluded_download_artifacts}
 assert model_hash==expected_model_hash
 rs.config=rs.Config(retrieval_method='e5',retrieval_topk=3,index_path=verified['e5_Flat.index']['path'],corpus_path=verified['wiki-18.jsonl']['path'],faiss_gpu=False,retrieval_model_path=str(model),retrieval_pooling_method='mean',retrieval_query_max_length=256,retrieval_use_fp16=True,retrieval_batch_size=512,encoder_device='cpu')
 rs.retriever=rs.DenseRetriever(rs.config)
 healthy();resources=faiss.StandardGpuResources();resources.setTempMemory(512*1024*1024)
 opts=faiss.GpuClonerOptions();opts.useFloat16=False
 torch.cuda.synchronize();rs.retriever.index=faiss.index_cpu_to_gpu(resources,0,rs.retriever.index,opts);torch.cuda.synchronize()
 # Same CPU encoder and same fixed real queries, independently evaluated on A800 host CPU.
 cpu_payload=np.load(root/'inputs/cpu-embeddings-2854.npz',allow_pickle=False);original=np.ascontiguousarray(cpu_payload['embeddings'],dtype=np.float32)
 assert str(cpu_payload['queries_sha256'].item())==baseline['queries_sha256'] and hashlib.sha256(original.tobytes()).hexdigest()==baseline['embeddings_sha256']
 t=time.perf_counter();cpu=time.process_time();encoded=rs.retriever.encoder.encode(baseline['queries']);encoding={'wall_seconds':time.perf_counter()-t,'process_cpu_seconds':time.process_time()-cpu}
 assert encoded.shape==original.shape and np.isfinite(encoded).all()
 rows=[]
 for k in [3,50]:
  torch.cuda.synchronize();scores,ids=rs.retriever.index.search(encoded,k);torch.cuda.synchronize()
  assert np.isfinite(scores).all() and (ids>=0).all()
  details=[]
  for i in range(len(encoded)):
   wanted=baseline['results'][i][str(k)];documents=rs.load_docs(rs.retriever.corpus,ids[i].tolist())
   contents=[hashlib.sha256(d['contents'].encode()).hexdigest() for d in documents]
   by_id=dict(zip(ids[i].tolist(),scores[i].tolist()));common=set(by_id)&set(wanted['ids'])
   details.append({'query_index':i,'exact_rank':ids[i].tolist()==wanted['ids'],'overlap':len(common),'ids':ids[i].tolist(),'scores':scores[i].tolist(),'contents_sha256':contents,'content_rank_equal':contents==wanted['contents_sha256'],'max_shared_score_delta':max((abs(by_id[v]-wanted['scores'][wanted['ids'].index(v)]) for v in common),default=None)})
  rows.append({'k':k,'details':details,'exact_rank_queries':sum(x['exact_rank'] for x in details),'content_rank_equal_queries':sum(x['content_rank_equal'] for x in details)})
 healthy()
 for rec in verified.values():
  st=pathlib.Path(rec['path']).stat();assert (st.st_size,st.st_mtime_ns,st.st_ino)==(rec['size'],rec['mtime_ns'],rec['inode'])
 assert {str(f.relative_to(model)):sha(f) for f in sorted(model.rglob('*')) if f.is_file()}==model_hash
 report={'status':'DIFFERENCE_REPORT_ONLY','scope':'14 fixed real queries, actual A800 CPU FP16 E5 plus single GPU FP32 index; not global query equivalence','gpu_uuid':expected,'encoder_device':'cpu','encoder_fp16':True,'index_float16':False,'scratch_bytes':512*1024*1024,'baseline_sha256':sha(root/'inputs/retrieval-cpu-2854.json'),'queries':baseline['queries'],'queries_sha256':baseline['queries_sha256'],'model_sha256':model_hash,'excluded_non_model_download_artifacts':{name:baseline['asset_identity']['model'][name] for name in sorted(excluded_download_artifacts) if name in baseline['asset_identity']['model']},'asset_identity':verified,'encoder_embeddings_sha256':hashlib.sha256(encoded.tobytes()).hexdigest(),'baseline_embeddings_sha256':baseline['embeddings_sha256'],'max_encoder_embedding_delta':float(np.max(np.abs(encoded-original))),'encoding_cost':encoding,'results':rows,'source_sha256':sha(__file__),'retrieval_module_sha256':sha(rs.__file__),'pid':os.getpid(),'gpu_leases':json.loads(os.environ['SEARCH_R1_GPU_LEASES']),'environment_freeze_sha256':sha(root/'logs/environment-freeze.txt'),'platform_cgroup':pathlib.Path('/proc/self/cgroup').read_text()}
 (run/'real-encoder-difference.json').write_text(json.dumps(report,indent=2)+'\n')
 np.savez(run/'a800-cpu-embeddings.npz',embeddings=encoded,queries_sha256=np.array(baseline['queries_sha256']))
 # Keep author/common response code unchanged; add read-only provenance and own cost records.
 @rs.app.get('/provenance')
 async def provenance():
  healthy();return {'gpu_uuid':expected,'pid':os.getpid(),'run':str(run),'encoder_device':'cpu','encoder_fp16':True,'index_float16':False,'difference_report_sha256':sha(run/'real-encoder-difference.json'),'source_sha256':sha(__file__)}
 original_search=rs.retriever.batch_search
 def measured_search(query_list,num=None,return_score=False):
  healthy();t=time.perf_counter();cpu=time.process_time()
  out=original_search(query_list,num,return_score)
  healthy()
  with (run/'requests.jsonl').open('a') as f:f.write(json.dumps({'epoch':time.time(),'queries':len(query_list) if not isinstance(query_list,str) else 1,'topk':num,'wall_seconds':time.perf_counter()-t,'process_cpu_seconds':time.process_time()-cpu,'main_rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024})+'\n')
  return out
 rs.retriever.batch_search=measured_search
 import uvicorn
 print('AUX_RETRIEVER_READY_CANDIDATE',json.dumps({'run':str(run),'gpu_uuid':expected,'real_difference_sha256':sha(run/'real-encoder-difference.json')}),flush=True)
 uvicorn.run(rs.app,host='127.0.0.1',port=a.port)
finally:
 stop.set()
 if thread.ident is not None:thread.join(timeout=3)
 cpu=resource.getrusage(resource.RUSAGE_SELF);children=resource.getrusage(resource.RUSAGE_CHILDREN)
 (run/'cost.json').write_text(json.dumps({'start_epoch':start,'end_epoch':time.time(),'allocated_auxiliary_gpus':1,'gpu_uuid':expected,'scope':'complete own auxiliary server process lifetime including loading, gate and idle; excludes source env preparation and platform idle allocation before launch','gpu_peak_used_bytes':max((s['used_bytes'] for s in samples),default=None),'self_cpu_seconds':cpu.ru_utime+cpu.ru_stime,'reaped_child_cpu_seconds':children.ru_utime+children.ru_stime,'self_peak_rss_bytes':cpu.ru_maxrss*1024,'unmeasured':'combined simultaneous dataset worker RSS','violations':violations})+'\n')
