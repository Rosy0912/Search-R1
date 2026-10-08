"""Single leased auxiliary A800 FP32 index gate; fixed CPU embeddings only."""
import os,sys,json,pathlib,time,hashlib,subprocess,resource,threading,signal
root=pathlib.Path('/data/shaoshitong/search-r1-rl/tree-grpo-retrieval')
run=root/'runs'/sys.argv[1];run.mkdir(parents=True,exist_ok=False)
expected='GPU-c63282c6-1d05-224e-234b-2a8c794b1f04'
assert os.environ['CUDA_VISIBLE_DEVICES']==expected
assert 'cri-containerd-' in pathlib.Path('/proc/self/cgroup').read_text(), 'Expected existing platform job container allocation'
from search_r1.baselines.preflight import validate_gpu_lease
from types import SimpleNamespace
validate_gpu_lease(SimpleNamespace(trainer=SimpleNamespace(nnodes=1,n_gpus_per_node=1)))
start=time.time();samples=[];stop=threading.Event();ready=threading.Event();violation=[]
def sample():
 try:
  import pynvml as nv
  nv.nvmlInit();h=nv.nvmlDeviceGetHandleByUUID(expected.encode())
  while not stop.is_set():
   m=nv.nvmlDeviceGetMemoryInfo(h);u=nv.nvmlDeviceGetUtilizationRates(h)
   processes={p.pid:p for p in nv.nvmlDeviceGetComputeRunningProcesses(h)+nv.nvmlDeviceGetGraphicsRunningProcesses(h)}
   foreign=[{'pid':int(p.pid),'memory_bytes':int(p.usedGpuMemory)} for p in processes.values() if p.pid!=os.getpid()]
   row={'epoch':time.time(),'uuid':expected,'used_bytes':m.used,'total_bytes':m.total,'gpu_util':u.gpu,'foreign':foreign};samples.append(row)
   with (run/'gpu-samples.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   if foreign:raise RuntimeError('Foreign process appeared on selected auxiliary GPU')
   ready.set();stop.wait(1)
  nv.nvmlShutdown()
 except BaseException as e:
  violation.append(repr(e));ready.set();os.kill(os.getpid(),15)
def terminate(signum,frame):
 raise SystemExit(128+signum)
signal.signal(signal.SIGTERM,terminate)
thread=threading.Thread(target=sample,daemon=True)
try:
 thread.start()
 assert ready.wait(10) and not violation and samples and thread.is_alive(), 'GPU monitor not ready'
 import torch
 import numpy as np,faiss
 assert torch.cuda.device_count()==faiss.get_num_gpus()==1
 assert str(root/'env')==sys.prefix
 assert str(root/'env') in torch.__file__ and str(root/'retriever-gpu-lib') in faiss.__file__
 verified=json.loads((root/'logs/assets-verified.json').read_text())
 for rec in verified.values():
  p=pathlib.Path(rec['path']);s=p.stat();assert (s.st_size,s.st_mtime_ns,s.st_ino)==(rec['size'],rec['mtime_ns'],rec['inode'])
 a=np.load(root/'inputs/cpu-embeddings-2854.npz');q=a['embeddings'];base=json.loads((root/'inputs/retrieval-cpu-2854.json').read_text())
 assert str(a['queries_sha256'])==base['queries_sha256']
 q=np.ascontiguousarray(q,dtype=np.float32)
 assert q.shape==(14,768) and np.isfinite(q).all()
 assert hashlib.sha256(q.tobytes()).hexdigest()==base['embeddings_sha256']
 assert verified['e5_Flat.index']['sha256']==base['asset_identity']['index']['sha256']
 assert verified['wiki-18.jsonl']['sha256']==base['asset_identity']['corpus']['sha256']
 faiss.omp_set_num_threads(8)
 t=time.perf_counter();cpu=faiss.read_index(verified['e5_Flat.index']['path']);load=time.perf_counter()-t
 assert type(cpu).__name__=='IndexFlatIP' and cpu.ntotal==21015324 and cpu.d==768
 resources=faiss.StandardGpuResources();resources.setTempMemory(512*1024*1024)
 co=faiss.GpuClonerOptions();co.useFloat16=False
 torch.cuda.synchronize();t=time.perf_counter();index=faiss.index_cpu_to_gpu(resources,0,cpu,co);torch.cuda.synchronize();transfer=time.perf_counter()-t
 assert index.ntotal==cpu.ntotal and index.d==cpu.d
 del cpu
 rows=[]
 for n in [14,512]:
  batch=np.ascontiguousarray(q[np.arange(n)%14])
  for k in [3,50]:
   assert thread.is_alive() and samples and time.time()-samples[-1]['epoch']<5 and not violation
   torch.cuda.synchronize();t=time.perf_counter();c=time.process_time()
   scores,ids=index.search(batch,k);torch.cuda.synchronize();wall=time.perf_counter()-t;cpu_seconds=time.process_time()-c
   assert np.isfinite(scores).all() and (ids>=0).all()
   differences=[]
   for i in range(n):
    b=base['results'][i%14][str(k)];wanted=b['ids'];got=ids[i].tolist()
    assert len(set(got))==k
    common=set(wanted)&set(got);by_id=dict(zip(got,scores[i].tolist()))
    differences.append({'row':i,'query_index':i%14,'exact_rank':got==wanted,'overlap':len(common),'max_shared_score_delta':max((abs(by_id[x]-b['scores'][wanted.index(x)]) for x in common),default=None),'ids':got,'scores':scores[i].tolist()})
   rows.append({'queries':n,'k':k,'wall_seconds':wall,'process_cpu_seconds':cpu_seconds,'exact_rank_rows':sum(r['exact_rank'] for r in differences),'details':differences})
 assert thread.is_alive() and samples and time.time()-samples[-1]['epoch']<5 and not violation
 for rec in verified.values():
  stat=pathlib.Path(rec['path']).stat();assert (stat.st_size,stat.st_mtime_ns,stat.st_ino)==(rec['size'],rec['mtime_ns'],rec['inode'])
 report={'status':'DIFFERENCE_REPORT_ONLY','scope':'14 fixed CPU embeddings repeated to 512; not unique real query throughput nor global equivalence','gpu_uuid':expected,'platform_cgroup':pathlib.Path('/proc/self/cgroup').read_text(),'sys_prefix':sys.prefix,'torch_path':torch.__file__,'faiss_path':faiss.__file__,'faiss_compile_options':faiss.get_compile_options(),'index_float16':False,'scratch_bytes':512*1024*1024,'asset_identity':verified,'corpus_content_identity':'Matching document IDs imply same content from independently identical corpus SHA; no new encoder used in this index-only ablation','embeddings_file_sha256':hashlib.sha256((root/'inputs/cpu-embeddings-2854.npz').read_bytes()).hexdigest(),'load_cpu_index_seconds':load,'transfer_gpu_index_seconds':transfer,'results':rows,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
 (run/'index-difference.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({k:v for k,v in report.items() if k not in ['results','platform_cgroup','asset_identity']}),flush=True)
finally:
 stop.set()
 if thread.ident is not None:thread.join(timeout=3)
 (run/'cost.json').write_text(json.dumps({'start_epoch':start,'end_epoch':time.time(),'allocated_gpus':1,'scope':'auxiliary index probe body; setup and platform idle allocation excluded','gpu_uuid':expected,'gpu_peak_used_bytes':max((s['used_bytes'] for s in samples),default=None),'foreign_violations':violation})+'\n')
