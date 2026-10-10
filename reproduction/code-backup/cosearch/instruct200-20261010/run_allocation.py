"""Supervise native retrieval, short acceptance, recovery continuation and final evaluation."""
import hashlib,json,math,os,signal,socket,subprocess,sys,time,urllib.request
from pathlib import Path
from allocation import acquire
from retriever_spec import specification,BASE
from verify_updates import verify as verify_updates
from verify_checkpoint import verify as verify_checkpoint
from verify_evaluation import verify as verify_evaluation

ROOT=Path(os.environ.get('COSEARCH_RUN_ROOT',str(BASE/'runtime/cosearch-instruct200-20261010')))

def status(phase,**values):
    record=dict(phase=phase,epoch=time.time(),pid=os.getpid(),job=os.environ.get('SLURM_JOB_ID'),**values)
    path=ROOT/'controller-state.json';temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record,allow_nan=False,indent=2)+'\n');os.replace(temporary,path)

def recovery(step):
    records=[]
    for p in (ROOT/'evidence').glob('*.jsonl'):
        for line in p.read_text().splitlines():
            e=json.loads(line)
            if e.get('kind')=='recovery_state_equal' and str(e.get('job'))==os.environ['SLURM_JOB_ID'] and '/global_step_'+str(step)+'/' in e.get('path',''):records.append(e)
    if {(e['role'],e['rank']) for e in records}!={(role,rank) for role in ('main','reranker') for rank in range(4)} or len(records)!=8:raise RuntimeError('Missing complete fresh both-role recovery evidence')
    data=json.loads((ROOT/('data-recovery-'+str(step)+'.json')).read_text())
    if not data['equal'] or data['step']!=step:raise RuntimeError('Data recovery differs')
    return {'step':step,'role_rank_records':len(records),'data_state_sha256':data['data_state_sha256'],'actual_state_equal':True}

def main():
    approval=json.loads((ROOT/'storage-authorization.json').read_text())
    if approval.get('human_authorized') is not True or approval.get('budget_bytes',0)<1_000_000_000_000 or approval.get('archive_strategy')!='hkust3_verified_then_delete':raise RuntimeError('Storage authorization is required before launch')
    heartbeat=json.loads((ROOT/'archive-worker-heartbeat.json').read_text())
    if time.time()-heartbeat['epoch']>120:raise RuntimeError('HKUST3 archive worker not freshly reachable')
    manifest=json.loads((ROOT/'deployment-manifest.json').read_text())
    if not manifest.get('published_remote_sha_verified'):raise RuntimeError('Unverified publication')
    for name,expected in manifest['sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:raise RuntimeError('Source changed before execution: '+name)
    checked=subprocess.run([sys.executable,str(ROOT/'tools/verify_inputs.py')],capture_output=True,text=True,check=True)
    inputs=json.loads(checked.stdout)
    if inputs['status']!='FRESH_MODEL_AND_FIXED_DATA_SHA_PASS':raise RuntimeError('Initial inputs not verified')
    (ROOT/'inputs-verified-at-launch.json').write_text(json.dumps(inputs,indent=2)+'\n')
    if not str(ROOT.resolve()).startswith('/data/'):raise RuntimeError('Outputs must resolve to HPC shared storage')
    probe={'job':os.environ['SLURM_JOB_ID'],'host':socket.gethostname(),'root':str(ROOT.resolve()),'epoch':time.time()}
    payload=(json.dumps(probe)+'\n').encode();path=ROOT/'shared-storage-probe.json'
    with path.open('wb') as stream:stream.write(payload);stream.flush();os.fsync(stream.fileno())
    if path.read_bytes()!=payload:raise RuntimeError('Shared storage readback differs')
    leases=acquire(ROOT)
    for name in ('logs','tmp','cache','evidence'):(ROOT/name).mkdir(exist_ok=True)
    env=dict(os.environ,COSEARCH_RUN_ROOT=str(ROOT),COSEARCH_AUDIT_ROOT=str(ROOT/'evidence'),TMPDIR=str(ROOT/'tmp'),HF_HOME=str(ROOT/'cache/hf-training'),HF_DATASETS_CACHE=str(ROOT/'cache/datasets-training'),TRITON_CACHE_DIR=str(ROOT/'cache/triton'),TORCH_EXTENSIONS_DIR=str(ROOT/'cache/torch-extensions'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    for key in ('RAY_ADDRESS','RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES','RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL'):
        env.pop(key,None)
        os.environ.pop(key,None)
    # Keep the verified four-device mask stable before any worker CUDA imports.
    env['RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES']='1'
    rpc=Path('/tmp')/('csrpc-'+env['SLURM_JOB_ID'])
    rpc.mkdir(mode=0o700,exist_ok=True)
    env['VLLM_RPC_BASE_PATH']=str(rpc)
    scorer=BASE/'runtime/pytrec-eval-05-hpc-build'
    scorer_receipt=json.loads((scorer/'build-complete.json').read_text())
    extension=Path(scorer_receipt['extension'])
    if scorer_receipt['status']!='PYTREC_05_NATIVE_IMPORT_METRICS_PASS' or hashlib.sha256(extension.read_bytes()).hexdigest()!=scorer_receipt['sha256']:
        raise RuntimeError('Native scorer compatibility build not verified')
    env['PYTHONPATH']=':'.join([str(ROOT/'tools'),str(ROOT/'repo'),str(ROOT/'repo/verl'),str(extension.parent),str(BASE/'runtime/flash-attn-281-hpc-build/lib'),str(BASE/'source/author-native/env/lib/python3.11/site-packages')])
    python=str(BASE/'source/author-native/python/cpython-3.11.17-linux-x86_64-gnu/bin/python3.11')
    children=[];logs=[]
    def launch(command,name,child_env):
        log=(ROOT/'logs'/name).open('a');logs.append(log)
        p=subprocess.Popen(command,env=child_env,cwd=ROOT/'repo',stdout=log,stderr=log,start_new_session=True)
        children.append(p);return p
    def wait(child,phase,service=None):
        status(phase,child_pid=child.pid)
        while child.poll() is None:
            if service is not None and service.poll() is not None:raise RuntimeError('Owned retriever exited')
            inventory=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,memory.total','--format=csv,noheader,nounits'],text=True)
            rows=[]
            for line in inventory.splitlines():
                uuid,used,total=[x.strip() for x in line.split(',')]
                if uuid in env['COSEARCH_LEASED_UUIDS'].split(','):rows.append({'uuid':uuid,'used_mib':int(used),'total_mib':int(total)})
            if len(rows)!=4:raise RuntimeError('Missing own GPU memory samples')
            with (ROOT/'gpu-memory.jsonl').open('a') as f:f.write(json.dumps({'phase':phase,'epoch':time.time(),'devices':rows})+'\n')
            time.sleep(10)
        if child.returncode!=0:raise RuntimeError(phase+' exited '+str(child.returncode))
    try:
        command,retriever_env=specification(ROOT)
        retriever_env.update(COSEARCH_RUN_ROOT=str(ROOT),COSEARCH_LEASED_UUIDS=env['COSEARCH_LEASED_UUIDS'])
        probe=launch([command[0],str(ROOT/'tools/probe_faiss_gpu.py')],'faiss-probe.log',retriever_env);wait(probe,'faiss_gpu_probe')
        with socket.socket() as sock:
            if sock.connect_ex(('127.0.0.1',18086))==0:raise RuntimeError('Retrieval port already occupied; do not interfere')
        service=launch(command,'retriever.log',retriever_env);deadline=time.monotonic()+1200
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        while True:
            if service.poll() is not None:raise RuntimeError('Retriever startup failed')
            try:
                with opener.open('http://127.0.0.1:18086/openapi.json',timeout=5) as response:
                    if response.status==200:break
            except OSError:pass
            if time.monotonic()>deadline:raise RuntimeError('Retriever startup timeout')
            status('retriever_startup',child_pid=service.pid);time.sleep(5)
        request=urllib.request.Request('http://127.0.0.1:18086/retrieve',data=json.dumps({'queries':['What is the capital of France?'],'topk':50,'return_scores':True}).encode(),headers={'Content-Type':'application/json'})
        with opener.open(request,timeout=120) as response:raw=response.read()
        result=json.loads(raw)['result'];assert len(result)==1 and len(result[0])==50
        assert all(math.isfinite(row['score']) for row in result[0])
        (ROOT/'full-retriever-query.json').write_text(json.dumps({'status':'FULL_INDEX_OWN_GPU_QUERY_PASS','response_sha256':hashlib.sha256(raw).hexdigest(),'results':50})+'\n')
        short=launch([python,str(ROOT/'tools/launch_training.py'),'short5'],'short5.log',env);wait(short,'short5',service)
        update_receipt=verify_updates(ROOT/'evidence',5,env['SLURM_JOB_ID'])
        samples=[json.loads(line) for line in (ROOT/'gpu-memory.jsonl').read_text().splitlines() if json.loads(line)['phase']=='short5']
        if not samples or any(d['used_mib']>0.95*d['total_mib'] for sample in samples for d in sample['devices']):raise RuntimeError('Short-run GPU memory headroom failed')
        events=[]
        for path in (ROOT/'evidence').glob('main-*.jsonl'):
            events.extend(json.loads(line) for line in path.read_text().splitlines())
        identities={e['rank']:e for e in events if e['kind']=='initialized' and str(e['job'])==env['SLURM_JOB_ID']}
        updates={(e['outer'],e['rank']):e for e in events if e['kind']=='update' and str(e['job'])==env['SLURM_JOB_ID']}
        for rank in range(4):
            previous=updates[(4,rank)]['metrics']['perf/max_memory_reserved_gb']
            final_peak=updates[(5,rank)]['metrics']['perf/max_memory_reserved_gb']
            if final_peak-previous>0.05*identities[rank]['total_memory_bytes']/(1024**3):raise RuntimeError('Last short-run outer shows excessive memory growth')
        update_receipt['gpu_memory_headroom_pass']=True
        update_receipt['gpu_memory_stability_still_requires_review']=False
        (ROOT/'short-acceptance.json').write_text(json.dumps(update_receipt,indent=2)+'\n')
        verify_checkpoint(ROOT,5);verify_evaluation(ROOT/'validation/5.jsonl',512,5)
        continuation=launch([python,str(ROOT/'tools/launch_training.py'),'continue200'],'continue200.log',env);wait(continuation,'continue200',service)
        verify_updates(ROOT/'evidence',200,env['SLURM_JOB_ID']);recovery(5)
        from archive_checkpoint import allow_cleanup
        allow_cleanup(ROOT,5)
        for step in (5,50,100,150,200):
            saved=json.loads((ROOT/('checkpoint-receipt-'+str(step)+'.json')).read_text())
            archived=json.loads((ROOT/('archive-verified-'+str(step)+'.json')).read_text())
            if saved['step']!=step or archived['status']!='HKUST3_ALL_CHECKPOINT_SHA_PASS':raise RuntimeError('Missing full archived checkpoint')
            receipt=verify_evaluation(ROOT/'validation'/ (str(step)+'.jsonl'),512,step)
            (ROOT/('validation-receipt-'+str(step)+'.json')).write_text(json.dumps(receipt,indent=2)+'\n')
        final=launch([python,str(ROOT/'tools/launch_training.py'),'final_eval'],'final-eval.log',env);wait(final,'final_eval',service)
        receipt=verify_evaluation(ROOT/'final-evaluation/200.jsonl',51713,200);recovery(200)
        verify_updates(ROOT/'evidence',200,env['SLURM_JOB_ID'])
        (ROOT/'final-evaluation-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        allow_cleanup(ROOT,200)
        status('COMPLETE_200_FINAL_EVALUATION_AND_RECOVERY_VERIFIED')
    except BaseException as error:
        status('FAILED',error_type=type(error).__name__,error=str(error));raise
    finally:
        for child in reversed(children):
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        for log in logs:log.close()
        for lease in leases:lease.close()
if __name__=='__main__':main()
