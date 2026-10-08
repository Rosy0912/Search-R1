"""Own auxiliary index only: binary FP32 vectors, one FP32 GPU flat index."""
import argparse,pathlib,os,sys,json,time,hashlib,resource,threading,signal,socket,io
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
 assert sys.prefix==str(root/'env') and str(root/'env') in torch.__file__ and str(root/'retriever-gpu-lib') in faiss.__file__
 assert torch.cuda.device_count()==faiss.get_num_gpus()==1
 faiss.omp_set_num_threads(8)
 baseline=json.loads((root/'inputs/retrieval-cpu-2854.json').read_text());verified=json.loads((root/'logs/assets-verified.json').read_text())
 assert verified['e5_Flat.index']['sha256']==baseline['asset_identity']['index']['sha256']
 def check_assets():
  for rec in verified.values():
   st=pathlib.Path(rec['path']).stat();assert (st.st_size,st.st_mtime_ns,st.st_ino)==(rec['size'],rec['mtime_ns'],rec['inode'])
 check_assets();t=time.perf_counter();cpu_index=faiss.read_index(verified['e5_Flat.index']['path']);load_seconds=time.perf_counter()-t
 assert type(cpu_index).__name__=='IndexFlatIP' and cpu_index.ntotal==21015324 and cpu_index.d==768
 healthy();resources=faiss.StandardGpuResources();resources.setTempMemory(512*1024*1024)
 opts=faiss.GpuClonerOptions();opts.useFloat16=False
 torch.cuda.synchronize();index=faiss.index_cpu_to_gpu(resources,0,cpu_index,opts);torch.cuda.synchronize();del cpu_index
 check_assets();instance={'gpu_uuid':expected,'pid':os.getpid(),'run':str(run),'mode':'fp32-index-only','index_float16':False,'index_sha256':verified['e5_Flat.index']['sha256'],'ntotal':int(index.ntotal),'dimension':int(index.d),'scratch_bytes':512*1024*1024,'source_sha256':sha(__file__)}
 (run/'provenance.json').write_text(json.dumps(dict(instance,platform_cgroup=pathlib.Path('/proc/self/cgroup').read_text(),gpu_leases=json.loads(os.environ['SEARCH_R1_GPU_LEASES']),load_seconds=load_seconds),indent=2)+'\n')
 from fastapi import FastAPI,Request,HTTPException
 import uvicorn
 app=FastAPI()
 @app.get('/provenance')
 async def provenance():
  healthy();check_assets();return instance
 @app.post('/index_search')
 async def index_search(request:Request,k:int=3):
  healthy();check_assets()
  if k not in (3,50):raise HTTPException(400,'Only reviewed k3/k50 allowed')
  chunks=[];size=0
  async for chunk in request.stream():
   size+=len(chunk)
   if size>2*1024*1024:raise HTTPException(413,'Query frame exceeds own512 batch budget')
   chunks.append(chunk)
  raw=b''.join(chunks)
  try:
   header=io.BytesIO(raw);version=np.lib.format.read_magic(header)
   if version==(1,0):shape,fortran,dtype=np.lib.format.read_array_header_1_0(header)
   elif version==(2,0):shape,fortran,dtype=np.lib.format.read_array_header_2_0(header)
   else:raise ValueError('Unsupported numeric frame version')
   if dtype!=np.dtype('float32') or len(shape)!=2 or shape[1]!=768 or not 1<=shape[0]<=512 or len(raw)!=header.tell()+shape[0]*768*4:raise ValueError('Header shape/dtype/length outside fixed budget')
   q=np.load(io.BytesIO(raw),allow_pickle=False)
   if not isinstance(q,np.ndarray) or q.dtype!=np.dtype('float32') or q.ndim!=2 or q.shape[1]!=768 or not 1<=len(q)<=512 or not np.isfinite(q).all():raise ValueError('Expected finite1..512 x768 float32 array')
   q=np.ascontiguousarray(q);vector_sha=hashlib.sha256(q.tobytes()).hexdigest()
  except Exception as err:raise HTTPException(400,'Invalid numeric vector frame') from err
  torch.cuda.synchronize();t=time.perf_counter();cpu=time.process_time();scores,ids=index.search(q,k);torch.cuda.synchronize()
  if not np.isfinite(scores).all() or (ids<0).any() or (ids>=index.ntotal).any():raise RuntimeError('Invalid exact index result')
  healthy()
  with (run/'requests.jsonl').open('a') as f:f.write(json.dumps({'epoch':time.time(),'queries':len(q),'k':k,'vectors_sha256':vector_sha,'search_wall_seconds':time.perf_counter()-t,'process_cpu_seconds':time.process_time()-cpu})+'\n')
  return {'ids':ids.tolist(),'scores':scores.tolist(),'vectors_sha256':vector_sha,'request_sha256':hashlib.sha256(raw).hexdigest(),'k':k,'instance':instance}
 print('AUX_INDEX_READY_CANDIDATE',json.dumps(instance),flush=True)
 uvicorn.run(app,host='127.0.0.1',port=a.port)
finally:
 stop.set()
 if thread.ident is not None:thread.join(timeout=3)
 cpu=resource.getrusage(resource.RUSAGE_SELF);children=resource.getrusage(resource.RUSAGE_CHILDREN)
 (run/'cost.json').write_text(json.dumps({'start_epoch':start,'end_epoch':time.time(),'allocated_auxiliary_gpus':1,'gpu_uuid':expected,'scope':'complete own auxiliary server process lifetime including loading, gate and idle; excludes source env preparation and platform idle allocation before launch','gpu_peak_used_bytes':max((s['used_bytes'] for s in samples),default=None),'self_cpu_seconds':cpu.ru_utime+cpu.ru_stime,'reaped_child_cpu_seconds':children.ru_utime+children.ru_stime,'self_peak_rss_bytes':cpu.ru_maxrss*1024,'unmeasured':'combined simultaneous dataset worker RSS','violations':violations})+'\n')
