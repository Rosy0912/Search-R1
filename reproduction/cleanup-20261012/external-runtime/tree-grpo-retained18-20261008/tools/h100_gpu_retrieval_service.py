"""One independently allocated H100: original GPU E5 recipe, FP32 flat index.
All CUDA initialization and queries run on one worker thread. No training changes.
"""
import argparse, asyncio, concurrent.futures, hashlib, json, os, pathlib
import resource, signal, socket, subprocess, sys, time, threading

ROOT = pathlib.Path('/dataset/shaoshitong/search-r1-rl/tree-grpo-retrieval-h100')
p = argparse.ArgumentParser()
p.add_argument('--run', required=True)
p.add_argument('--baseline', type=pathlib.Path, required=True)
p.add_argument('--host', default='10.32.0.11', choices=['10.32.0.11'])
p.add_argument('--port', type=int, default=18082)
p.add_argument('--max-seconds', type=int, default=14400)
a = p.parse_args()
assert os.environ['SLURM_JOB_NODELIST'] == 'h100-11'
job = os.environ['SLURM_JOB_ID']; assert job.isdigit()
assert any('job_'+job in line.split(':',2)[-1].strip().split('/') for line in pathlib.Path('/proc/self/cgroup').read_text().splitlines())
assert sys.prefix == str(ROOT/'env') and a.max_seconds > 0
run = ROOT/'runs'/a.run
assert run.resolve().parent == (ROOT/'runs').resolve()
run.mkdir(parents=True, exist_ok=False)
with socket.socket() as s: s.bind((a.host, a.port))
from search_r1.baselines.preflight import validate_gpu_lease
from types import SimpleNamespace
validate_gpu_lease(SimpleNamespace(trainer=SimpleNamespace(nnodes=1,n_gpus_per_node=1)))
lease = json.loads(os.environ['SEARCH_R1_GPU_LEASES'])
assert len(lease) == 1
start = time.time(); poison = []; metadata = None; identity = {}
guard = subprocess.Popen([sys.executable, str(ROOT/'tools/h100_retrieval_resource_guard.py'), '--output', str(run/'gpu-samples.jsonl')])
executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix='own-h100-retrieval')
def sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2), b''): h.update(b)
    return h.hexdigest()
def healthy():
    assert not poison and guard.poll() is None, 'Own retrieval guard/service failed'
    assert time.time()-start < a.max_seconds, 'Own bounded retrieval lifetime reached'
    latest = (run/'gpu-samples.jsonl').read_text().splitlines()[-1]
    assert time.time()-json.loads(latest)['epoch'] < 15, 'GPU guard stale'
    for record in identity.values():
        st=pathlib.Path(record['path']).stat()
        assert (st.st_size,st.st_mtime_ns,st.st_ino)==(record['size'],record['mtime_ns'],record['inode']), 'Immutable retrieval asset changed'
def terminate(signum, frame): raise SystemExit(128+signum)
signal.signal(signal.SIGTERM, terminate)
deadline_timer=threading.Timer(a.max_seconds,lambda:os.kill(os.getpid(),signal.SIGTERM))
deadline_timer.daemon=True; deadline_timer.start()
def initialize():
    global metadata, identity
    import torch, numpy as np, faiss
    import search_r1.search.retrieval_server as rs
    assert str(ROOT/'retriever-gpu-lib') in faiss.__file__
    assert str(ROOT/'env') in torch.__file__ and str(ROOT/'repo') in rs.__file__
    assert torch.cuda.device_count() == faiss.get_num_gpus() == 1
    torch.set_num_threads(8); faiss.omp_set_num_threads(8)
    baseline = json.loads(a.baseline.read_text())
    assert baseline['mode']=='cpu' and baseline['asset_identity_verified']
    assets = ROOT/'assets-root/assets'
    for kind, name in [('index','e5_Flat.index'),('corpus','wiki-18.jsonl')]:
        f=assets/'wiki'/name
        assert sha(f)==baseline['asset_identity'][kind]['sha256']
        st=f.stat(); identity[kind]={'path':str(f),'sha256':baseline['asset_identity'][kind]['sha256'],'size':st.st_size,'mtime_ns':st.st_mtime_ns,'inode':st.st_ino}
    ignored={'model.safetensors.partial','model.safetensors.source.partial','model.safetensors.source.partial.aria2'}
    model=assets/'e5-base-v2'
    model_hash={str(f.relative_to(model)):sha(f) for f in model.rglob('*') if f.is_file() and str(f.relative_to(model)) not in ignored}
    assert model_hash=={k:v for k,v in baseline['asset_identity']['model'].items() if k not in ignored}
    rs.config=rs.Config(retrieval_method='e5',retrieval_topk=3,index_path=str(assets/'wiki/e5_Flat.index'),corpus_path=str(assets/'wiki/wiki-18.jsonl'),faiss_gpu=False,retrieval_model_path=str(model),retrieval_pooling_method='mean',retrieval_query_max_length=256,retrieval_use_fp16=True,retrieval_batch_size=512,encoder_device='cuda')
    rs.retriever=rs.DenseRetriever(rs.config)
    assert type(rs.retriever.index).__name__=='IndexFlatIP' and rs.retriever.index.d==768 and rs.retriever.index.ntotal==21015324
    resources=faiss.StandardGpuResources(); resources.setTempMemory(512*1024**2)
    options=faiss.GpuClonerOptions(); options.useFloat16=False
    rs.retriever.index=faiss.index_cpu_to_gpu(resources,0,rs.retriever.index,options)
    rs.retriever._own_gpu_resources=resources
    torch.cuda.synchronize(); healthy()
    assert model_hash=={str(f.relative_to(model)):sha(f) for f in model.rglob('*') if f.is_file() and str(f.relative_to(model)) not in ignored}, 'E5 identity changed during loading'
    original=rs.retriever.batch_search
    def measured(*args,**kwargs):
        healthy(); t=time.perf_counter(); cpu=time.process_time()
        out=original(*args,**kwargs); torch.cuda.synchronize(); healthy()
        queries=kwargs.get('query_list', args[0] if args else None)
        with (run/'requests.jsonl').open('a') as f:
            f.write(json.dumps({'epoch':time.time(),'queries':len(queries),'k':kwargs.get('num'),'wall_seconds':time.perf_counter()-t,'process_cpu_seconds':time.process_time()-cpu,'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated()})+'\n')
        return out
    rs.retriever.batch_search=measured
    metadata={'mode':'h100-same-cluster-gpu-e5-fp32-index','node':'h100-11','cluster':'wuhan-h100','training_node':'h100-9','job_id':job,'pid':os.getpid(),'run':str(run),'gpu_uuid':lease[0]['uuid'],'encoder_device':'cuda','encoder_fp16':True,'index_float16':False,'batch_size':512,'max_query_length':256,'ntotal':21015324,'dimension':768,'asset_identity':identity,'model_sha256':model_hash,'baseline_sha256':sha(a.baseline),'source_sha256':sha(__file__),'retrieval_module_sha256':sha(rs.__file__),'deadline_epoch':start+a.max_seconds,'endpoint':f'http://{a.host}:{a.port}','transport':'H100 internal IB interface, direct node9 to node11, no Mac or other cluster'}
    metadata['guard_source_sha256']=sha(ROOT/'tools/h100_retrieval_resource_guard.py')
    metadata['launcher_source_sha256']=sha(ROOT/'tools/start_h100_gpu_retrieval.sh')
    (run/'provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return rs
try:
    for _ in range(100):
        if (run/'gpu-samples.jsonl').exists(): break
        assert guard.poll() is None; time.sleep(.1)
    healthy()
    rs=executor.submit(initialize).result()
    from fastapi import FastAPI, Request, HTTPException
    import uvicorn
    app=FastAPI()
    @app.middleware('http')
    async def internal_only(request: Request, call_next):
        if request.client.host not in ('10.32.0.13','10.32.0.11'):
            from starlette.responses import JSONResponse
            return JSONResponse({'detail':'Only own H100 training node and retrieval node allowed'},status_code=403)
        return await call_next(request)
    @app.get('/provenance')
    async def provenance():
        healthy(); return metadata
    @app.post('/retrieve')
    async def retrieve(request: rs.QueryRequest):
        healthy()
        if request.topk not in (None,3,50):
            raise HTTPException(400,'Expected k3/k50')
        # The original dense retriever chunks arbitrary request sizes at 512;
        # Tree can send 1024 roots/2048 expansions, including an empty search batch.
        # Preserve the advertised document-only schema by stripping scores from
        # the exact original scored endpoint result; Tree always requests scores.
        scored = request.copy(update={'return_scores':True})
        try:
            out = await asyncio.wait_for(asyncio.get_running_loop().run_in_executor(executor,lambda:asyncio.run(rs.retrieve_endpoint(scored))),timeout=600)
            if not request.return_scores:
                return {'result':[[v['document'] for v in row] for row in out['result']]}
            return out
        except Exception as err:
            poison.append(repr(err)); (run/'service-failure.json').write_text(json.dumps({'epoch':time.time(),'error':poison[0]})+'\n'); raise
    print('H100_GPU_RETRIEVER_READY_CANDIDATE', json.dumps(metadata), flush=True)
    uvicorn.run(app,host=a.host,port=a.port)
finally:
    deadline_timer.cancel()
    guard.terminate()
    try: guard.wait(timeout=10)
    except subprocess.TimeoutExpired: guard.kill(); guard.wait()
    executor.shutdown(wait=False,cancel_futures=True)
    usage=resource.getrusage(resource.RUSAGE_SELF)
    (run/'cost.json').write_text(json.dumps({'start_epoch':start,'end_epoch':time.time(),'job_id':job,'allocated_gpus':1,'gpu_uuid':lease[0]['uuid'],'self_cpu_seconds':usage.ru_utime+usage.ru_stime,'self_peak_rss_bytes':usage.ru_maxrss*1024,'poison':poison,'scope':'Own service body, including load, gate, idle; scheduler allocation cost separate'})+'\n')
