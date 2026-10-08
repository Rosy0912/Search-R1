"""Native8 AF_UNIX FP32 scorer + full GPU retrieval +8vLLM co-residency; hardware surrogate only."""
import importlib.util,concurrent.futures,hashlib,json,math,os,signal,socket,subprocess,sys,threading,time
from pathlib import Path
import pandas as pd
import requests
from probe_data import resolve_train,sha
from probe_gpu_mapping import numeric_worker_mapping
from probe_round_checks import resource_envelope
ROOT=Path(os.environ['AGENT_ROOT']).resolve();JOB=os.environ['SLURM_JOB_ID']
assert ROOT==Path('/dataset/shaoshitong/search-r1-rl/topd'), 'Unexpected own root'
assert socket.gethostname().split('.')[0]=='h100-11' and os.environ.get('SLURM_JOB_NODELIST')=='h100-11', 'Require own h100-11 allocation'
assert int(os.environ.get('SLURM_CPUS_PER_TASK','0'))>=128 and int(os.environ.get('SLURM_MEM_PER_NODE','0'))>=384*1024, 'Require allocated128CPU/384GiB'
assert int(os.environ.get('SLURM_GPUS_ON_NODE','0'))==8 and len(os.environ.get('SLURM_JOB_GPUS','').split(','))==8, 'Require real Slurm GRES8'
assert len(os.environ.get('CUDA_VISIBLE_DEVICES','').split(','))==8, 'Require exactly8 allocated visible devices'
reviewed=json.loads((Path(__file__).parent/'reviewed-sources.json').read_text())
assert subprocess.check_output(['git','-C',str(ROOT/'repo'),'rev-parse','HEAD'],text=True).strip()==reviewed['code_commit'], 'Actual HEAD differs from reviewed source'
for name,expected in reviewed['files'].items():
    assert sha(ROOT/name)==expected, 'Reviewed deployed bytes differ: '+name
OUT=ROOT/'logs'/f'afunix-scorer-hardware-{JOB}';OUT.mkdir(exist_ok=False)
(OUT/'teacher-ipc').mkdir(mode=0o700)
ALIAS=Path('/tmp')/f'topd-ipc-{JOB}'
assert not ALIAS.exists() and not ALIAS.is_symlink(),'Refuse existing short IPC alias'
spec=importlib.util.spec_from_file_location('owned_topd_launcher',ROOT/'tools/topd-short-run.py')
launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)
OwnedProcesses=launcher.OwnedProcesses;cgroup_scope=launcher.cgroup_scope
assert os.environ.get('SEARCH_R1_GPU_PREFLIGHT')=='1'
leases=json.loads(os.environ['SEARCH_R1_GPU_LEASES']);assert len(leases)==8
for item in leases:os.fstat(int(item['fd']))
UUIDS={x['uuid'] for x in leases};scope,relative=cgroup_scope(JOB);owned=OwnedProcesses(scope,relative)
for key in ('SEARCH_R1_TOPD_AUDIT_DIR','SEARCH_R1_TOPD_RUN_ID','SEARCH_R1_TOPD_COMMIT','SEARCH_R1_TOPD_CONFIG_SHA256','SEARCH_R1_TOPD_DATA_IDENTITY'):os.environ.pop(key,None)
for key in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):os.environ.pop(key,None)
os.environ.update(NO_PROXY='127.0.0.1,localhost,::1',no_proxy='127.0.0.1,localhost,::1')
http_session=requests.Session();http_session.trust_env=False
stop=threading.Event();phase='baseline';samples=[]
def inventory():
    text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=10)
    result=[]
    for line in text.splitlines():
        index,uid,used,total,util=[v.strip() for v in line.split(',')]
        if uid in UUIDS:result.append(dict(index=int(index),uuid=uid,memory_used_mib=float(used),memory_total_mib=float(total),utilization=float(util)))
    assert len(result)==8
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=10)
    process_samples=[launcher.process_sample(pid) for pid in sorted(launcher.cgroup_pids(scope)-owned.baseline)]
    parent_process=launcher.process_sample(os.getpid())
    resources=resource_envelope(process_samples,parent_process)
    return dict(time=time.time(),phase=phase,gpus=result,processes=apps.splitlines(),own_process_samples=process_samples,parent_process_sample=parent_process,own_resource_envelope=resources)
def sample():
    entry=inventory();samples.append(entry)
    with (OUT/'gpu-memory-timeseries.jsonl').open('a') as f:f.write(json.dumps(entry)+'\n')
    return entry
monitor_errors=[]
def monitor():
    while not stop.wait(1):
        try:sample()
        except Exception as e:monitor_errors.append(repr(e))
def spawn(cmd,env,name):
    return owned.spawn(cmd,OUT/(name+'.log'),env)
def waitfiles(children,directory,kind,timeout=1200):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        assert all(c.poll() is None for c in children),'Own child failed; inspect its exact log'
        if all((directory/f'{kind}-{i}.json').is_file() for i in range(len(children))):return
        time.sleep(1)
    raise TimeoutError(kind+' barrier exceeded')
def http(queries,k,label):
    started_epoch=time.time();tick=time.perf_counter();r=http_session.post('http://127.0.0.1:18085/retrieve',json=dict(queries=queries,topk=k,return_scores=True),timeout=(5,900));r.raise_for_status();payload=r.json()
    assert len(payload['result'])==len(queries)
    for row in payload['result']:
        assert len(row)==k
        assert all(isinstance(x['document']['contents'],str) and math.isfinite(float(x['score'])) for x in row)
    result=dict(label=label,started_epoch=started_epoch,completed_epoch=time.time(),queries=len(queries),topk=k,wall_seconds=time.perf_counter()-tick,
        response_sha256=hashlib.sha256(r.content).hexdigest(),
        content_sha256=[[hashlib.sha256(v['document']['contents'].encode()).hexdigest() for v in row] for row in payload['result']])
    (OUT/(label+'.json')).write_text(json.dumps(result,indent=2));return result
def interrupted(signum,frame):raise RuntimeError(f'Own hardware probe interrupted by signal {signum}')
signal.signal(signal.SIGTERM,interrupted)
signal.signal(signal.SIGINT,interrupted)
result=dict(job_id=JOB,NO_TRAINING=True,scope='HARDWARE_SURROGATE_NO_TRAINING; real8 AF_UNIX scorer + retriever +vLLM; no algorithm teacher/actor optimizer/FSDP training residency',leases=[{'uuid':i['uuid']} for i in leases],requests=[],script_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*') if p.is_file() and p.suffix in ('.py','.sbatch')},common_config_sha256={name:sha(ROOT/'repo'/name) for name in ['verl/trainer/config/ppo_baselines.yaml','verl/trainer/config/ppo_trainer.yaml','verl/workers/rollout/vllm_rollout/vllm_rollout.py']})
try:
    ALIAS.symlink_to(OUT,target_is_directory=True)
    baseline=sample();worker_mapping=numeric_worker_mapping(baseline['gpus'],UUIDS,os.environ['CUDA_VISIBLE_DEVICES'].split(','));result['worker_gpu_mapping']=worker_mapping;assert all(g['memory_used_mib']/g['memory_total_mib']<=.15 and g['utilization']<=15 for g in baseline['gpus'])
    import torch
    torch.set_num_threads(16)
    from search_r1.baselines.distillation import FrozenTeacher
    from search_r1.baselines.topd_local_gpu import LocalTeacherClient,implementation_fingerprint,tensor_save,digest
    from verl.utils import hf_tokenizer
    from hydra import compose,initialize_config_dir
    from omegaconf import OmegaConf
    os.environ['SEARCH_R1_ASSETS']=str(ROOT/'assets-root')
    with initialize_config_dir(config_dir=str(ROOT/'repo/verl/trainer/config'),version_base=None):
        config=compose(config_name='ppo_baselines')
    rollout=config.actor_rollout_ref.rollout
    assert float(rollout.temperature)==1., 'Hardware parity temperature must be1'
    assert rollout.gpu_memory_utilization==.6 and rollout.tensor_model_parallel_size==1 and rollout.enforce_eager
    effective=dict(max_model_len=int(rollout.prompt_length+rollout.response_length),dtype=str(rollout.dtype),
        gpu_memory_utilization=float(rollout.gpu_memory_utilization),tensor_parallel_size=int(rollout.tensor_model_parallel_size),
        enforce_eager=bool(rollout.enforce_eager),configured_max_num_seqs=int(rollout.max_num_seqs),
        configured_max_num_batched_tokens=int(rollout.max_num_batched_tokens),
        constructor_passes_max_num_seqs=False,constructor_passes_max_num_batched_tokens=False,
        implementation_scope='common patched rollout wrapper uses actor-module weights/dummy_dtensor then offloads; this probe loads normal HF standalone LLM only')
    # Reviewed wrapper passes neither configured max_num_seqs nor max_num_batched_tokens.
    wrapper=(ROOT/'repo/verl/workers/rollout/vllm_rollout/vllm_rollout.py').read_text()
    constructor=wrapper.split('self.inference_engine = LLM(',1)[1].split('\n        # Offload',1)[0]
    assert 'max_num_seqs=' not in constructor and 'max_num_batched_tokens=' not in constructor, 'Wrapper changed: review probe constructor alignment'
    (OUT/'effective-rollout-config.json').write_text(json.dumps(effective,indent=2))
    result['effective_rollout_config']=effective
    config.trainer.total_training_steps=4
    config.baselines.distillation.chunk_size=16
    base=ROOT/'assets-root/assets/qwen2.5-3b'
    hardware_config=dict(scope='HARDWARE_SURROGATE_NO_TRAINING',rounds=3,rows_per_service=2,prompt_lengths=[256,256,2048],response_lengths=[32,500,4096],max_context_scope='round3 longestrow6144 original-GPU-function parity; CPU cross-backend diagnostic first2 only',
        numeric_oracle='ORIGINAL_FROZENTEACHER_GPU_FUNCTION_ORACLE_NO_TRAINING',cpu_parity_scope='NON_BLOCKING_CROSS_BACKEND_DIAGNOSTIC',
        temperature=1.,chunk=16,atol=1e-4,rtol=1e-4,rollout=effective,
        data_identity_scope='common split deployment identity only; scoring inputs are synthetic ordinary tokens, not training examples')
    config_path=OUT/'hardware-config.json';config_path.write_text(json.dumps(hardware_config,sort_keys=True,indent=2))
    hf_files={str(path.relative_to(base)):sha(path) for path in base.rglob('*') if path.is_file()}
    hardware_hf_identity=hashlib.sha256(json.dumps(hf_files,sort_keys=True).encode()).hexdigest()
    binding=dict(code_commit=subprocess.check_output(['git','-C',str(ROOT/'repo'),'rev-parse','HEAD'],text=True).strip(),teacher_receipt_sha256=hardware_hf_identity)
    launch=dict(run_id='hardware-afunix-'+JOB,config_sha256=sha(config_path))
    result['launcher_sha256']=sha(ROOT/'tools/topd-short-run.py')
    result['actual_code_commit']=binding['code_commit'];result['actual_source_fingerprint']=implementation_fingerprint()
    result['torch_version']=torch.__version__
    assert binding['code_commit']==reviewed['code_commit'] and implementation_fingerprint()==reviewed['source_fingerprint'], 'Reviewed source identity differs'
    result['hardware_hf_identity_sha256']=hardware_hf_identity
    result['teacher_receipt_field_scope']='hardware-kind protocol field binds actual HF inventorySHA; no trained teacher receipt exists or is claimed'
    # Must start all CPU-resident scorers before any own GPU occupancy exceeds15%.
    scorer_children,scorer_manifest=launcher.start_parallel_scorers(OUT,ALIAS,owned,binding,base,launch,config,relative,
        kind='HARDWARE_SURROGATE_NO_TRAINING',rows_per_service=2)
    assert all(s['pid']!=os.getpid() for s in scorer_manifest['services'])
    result['scorer_startup']=[dict(rank=s['rank'],uuid=s['uuid'],pid=s['pid'],cgroup=relative) for s in scorer_manifest['services']]
    # Independent original-function CPU-ready oracle, before E5/vLLM residency.
    from original_gpu_oracle import atomic_json,IDENTITY
    oracle_dir=OUT/'original-gpu-oracle';oracle_dir.mkdir(mode=0o700)
    oracle_lease=next(item for item in leases if item['uuid']==scorer_manifest['services'][4]['uuid'])
    oracle_env=os.environ.copy();oracle_env.update(CUDA_VISIBLE_DEVICES=oracle_lease['uuid'],
        SEARCH_R1_GPU_LEASES=json.dumps([oracle_lease]),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',
        TORCH_ALLOW_TF32_CUBLAS_OVERRIDE='0',NVIDIA_TF32_OVERRIDE='0',PYTHONPATH=str(ROOT/'repo'))
    oracle=owned.spawn([sys.executable,str(ROOT/'tools/afunix-scorer-probe/original_gpu_oracle.py'),
        '--manifest',str(OUT/'teacher-ipc/manifest.json'),'--directory',str(oracle_dir)],
        OUT/'original-gpu-oracle.log',oracle_env,pass_fds=(int(oracle_lease['fd']),))
    deadline=time.monotonic()+1200
    while not (oracle_dir/'ready.json').exists():
        assert oracle.poll() is None,'Original oracle failed before CPU-ready'
        assert time.monotonic()<deadline,'Original oracle CPU-ready timeout';time.sleep(.2)
    oracle_ready=json.loads((oracle_dir/'ready.json').read_text())
    assert oracle_ready['pid']==oracle.pid and oracle_ready['uuid']==oracle_lease['uuid'] and oracle_ready['cgroup']==relative
    assert oracle_ready['status']=='CPU_READY_NO_CUDA_CONTEXT' and oracle_ready['cuda_initialized'] is False
    assert all(oracle_ready[key]==scorer_manifest[key] for key in IDENTITY)
    result['original_gpu_oracle_startup']=oracle_ready
    result['numeric_oracle_scope']='Prospective original one-row FrozenTeacher GPU function; not new ScorerSession/LocalTeacherClient; fixed1e-4 all3rounds'
    phase='eight-scorers-and-original-oracle-cpu-ready';sample()
    client=LocalTeacherClient(OUT/'teacher-ipc/manifest.json',base,required_kind='HARDWARE_SURROGATE_NO_TRAINING')
    tokenizer=hf_tokenizer(str(base),local_files_only=True,trust_remote_code=False)
    reference_teacher=FrozenTeacher(str(base),tokenizer,'cpu')
    ordinary=[value for value in tokenizer.encode('the answer search information Paris London science history book',add_special_tokens=False)
        if value not in {tokenizer.pad_token_id,tokenizer.bos_token_id,tokenizer.eos_token_id}]
    ordinary=list(dict.fromkeys(ordinary));assert len(ordinary)>=2


    with socket.socket() as s:s.bind(('127.0.0.1',18085))
    train_path,manifest_path=resolve_train(ROOT)
    result['split_artifact']=dict(train_path=str(train_path),manifest_path=str(manifest_path),train_sha256=sha(train_path),manifest_sha256=sha(manifest_path))
    rows=pd.read_parquet(train_path).head(512);assert len(rows)==512
    queries=[list(row['prompt'])[-1]['content'].rsplit('Question: ',1)[1].strip() for _,row in rows.iterrows()]
    (OUT/'queries.json').write_text(json.dumps(queries));result['data_sources']=rows['data_source'].value_counts().to_dict();result['query_selection']='first512 immutable train rows; not representative of all sources'
    server=ROOT/'repo/search_r1/search/retrieval_server.py';result['retrieval_source_sha256']=hashlib.sha256(server.read_bytes()).hexdigest()
    env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'retriever-gpu-lib')+':'+str(ROOT/'repo');env['OMP_NUM_THREADS']='8';env['MKL_NUM_THREADS']='8'
    phase='gpu-faiss-e5-startup';thread=threading.Thread(target=monitor,daemon=True);thread.start()
    tick=time.perf_counter();service=spawn([sys.executable,'-c',"import torch,runpy,sys;server=sys.argv.pop(1);sys.argv[0]=server;runpy.run_path(server,run_name='__main__')",str(server),'--index_path',str(ROOT/'assets-root/assets/wiki/e5_Flat.index'),'--corpus_path',str(ROOT/'assets-root/assets/wiki/wiki-18.jsonl'),'--retriever_model',str(ROOT/'assets-root/assets/e5-base-v2'),'--retriever_name','e5','--topk','3','--device','cuda','--faiss_gpu','--host','127.0.0.1','--port','18085','--threads','8'],env,'retriever')
    deadline=time.monotonic()+900
    while time.monotonic()<deadline:
        assert service.poll() is None,'Own GPU retriever failed'
        try:
            if http_session.get('http://127.0.0.1:18085/openapi.json',timeout=2).ok:break
        except requests.RequestException:pass
        time.sleep(2)
    else:raise TimeoutError('GPU retriever startup')
    result['startup_seconds']=time.perf_counter()-tick
    phase='faiss-e5-idle';time.sleep(3);sample()
    for k in (3,50):
        phase=f'gpu-http-512-top{k}';result['requests'].append(http(queries,k,phase));sample()
    def worker(rank,directory):
        wenv=os.environ.copy();wenv['CUDA_VISIBLE_DEVICES']=str(worker_mapping[rank]['index']);wenv['PROBE_EXPECTED_GPU_UUID']=worker_mapping[rank]['uuid'];wenv['PYTHONPATH']=str(ROOT/'repo')
        return spawn([sys.executable,str(ROOT/'tools/afunix-scorer-probe/vllm-worker.py'),'--directory',str(directory),'--rank',str(rank)],wenv,directory.name+f'-rank{rank}')
    d=OUT/'eight-vllm';d.mkdir();(d/'queries.json').write_text(json.dumps(queries));(d/'rollout-config.json').write_text(json.dumps(effective));phase='eight-vllm-startup'
    children=[worker(i,d) for i in range(8)];waitfiles(children,d,'ready');phase='eight-vllm-faiss-e5-idle';time.sleep(3);sample()
    result['ipc_rounds']=[]
    from safetensors.torch import save_file
    from probe_round_checks import validate_round_receipts,request_rows
    for step,rl in enumerate((32,500,4096),1):
        raw_prompts,raw_responses,raw_attention=request_rows(ordinary,tokenizer.pad_token_id,step)
        prompts=torch.tensor(raw_prompts,dtype=torch.long);responses=torch.tensor(raw_responses,dtype=torch.long);attention=torch.tensor(raw_attention,dtype=torch.long)
        cpu=None;cpu_seconds=None
        if step<3:
            phase=f'cpu-reference-round{step}'
            started=time.perf_counter()
            cpu=torch.cat([reference_teacher.log_probs(prompts[i:i+1],responses[i:i+1],attention[i:i+1],1.,None,16) for i in range(16)])
            cpu_seconds=time.perf_counter()-started
            assert bool(torch.isfinite(cpu).all())
        assert all(c.poll() is None for c in scorer_children+children) and service.poll() is None and oracle.poll() is None
        if step==1:(d/'GO').touch()
        phase=f'afunix-gpu-scoring-round{step}'
        scoring_started_epoch=time.time();started=time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future=pool.submit(client.score,prompts,responses,attention,1.,16,step)
            concurrent_http=[]
            while not future.done():
                # HTTP runs with all8 vLLM caches resident while the scorer round is active.
                concurrent_http.append(http(queries,3,f'scorer-round{step}-http{len(concurrent_http)}'))
                time.sleep(2)
            gpu=future.result()
        gpu_seconds=time.perf_counter()-started;candidate_scoring_completed_epoch=time.time()
        if not concurrent_http:concurrent_http.append(http(queries,3,f'scorer-round{step}-http-final'))
        assert all(c.poll() is None for c in children+scorer_children) and service.poll() is None
        receipts=client.last_receipts
        validate_round_receipts(receipts,scorer_manifest,step)
        scorers_released_epoch=time.time()
        (OUT/f'receipts-round{step}.json').write_text(json.dumps(receipts,indent=2))
        # Send the exact full16 input bytes only after all eight workers RELEASED0.
        oracle_input=oracle_dir/f'input-round{step}.safetensors'
        temporary=oracle_input.with_name(oracle_input.name+'.tmp')
        assert not oracle_input.exists() and not temporary.exists()
        save_file(dict(prompts=prompts,responses=responses,attention=attention,row_indices=torch.arange(16)),str(temporary))
        os.replace(temporary,oracle_input)
        oracle_request={key:scorer_manifest[key] for key in IDENTITY}
        oracle_request.update(outer_step=step,input_sha256=sha(oracle_input),all8_released_zero=True,
            candidate_receipts_sha256=sha(OUT/f'receipts-round{step}.json'),scorers_released_epoch=scorers_released_epoch)
        atomic_json(oracle_dir/f'request-round{step}.json',oracle_request)
        phase=f'original-gpu-function-oracle-round{step}'
        oracle_receipt_path=oracle_dir/f'receipt-round{step}.json';deadline=time.monotonic()+1800
        while not oracle_receipt_path.exists():
            assert oracle.poll() is None,'Original GPU oracle failed during round'
            assert time.monotonic()<deadline,'Original GPU oracle round timeout';time.sleep(.2)
        oracle_receipt=json.loads(oracle_receipt_path.read_text())
        assert oracle_receipt['status']=='RELEASED' and not oracle_receipt.get('release_error') and not oracle_receipt.get('error')
        assert oracle_receipt['pid']==oracle.pid and oracle_receipt['uuid']==oracle_lease['uuid'] and oracle_receipt['outer_step']==step
        assert all(oracle_receipt[key]==scorer_manifest[key] for key in IDENTITY)
        assert oracle_receipt['input_sha256']==sha(oracle_input) and oracle_receipt['allocator_released']==dict(allocated_bytes=0,reserved_bytes=0)
        assert oracle_receipt['released_cpu']['parameter_sha256']==receipts[0]['released_cpu']['parameter_sha256']
        oracle_output_path=oracle_dir/f'reference-round{step}.safetensors'
        assert oracle_receipt['output_sha256']==sha(oracle_output_path)
        from safetensors.torch import load_file
        oracle_raw=load_file(str(oracle_output_path),device='cpu');assert set(oracle_raw)=={'original_gpu_log_probs'}
        original_gpu=oracle_raw['original_gpu_log_probs'];assert original_gpu.dtype==torch.float32
        assert original_gpu.shape==gpu.shape and bool(torch.isfinite(original_gpu).all()) and bool((original_gpu[attention==0]==0).all())
        delta=(gpu-original_gpu).abs();close=bool(torch.allclose(gpu,original_gpu,atol=1e-4,rtol=1e-4))
        cpu_delta=(gpu-cpu).abs() if cpu is not None else None
        cpu_close=bool(torch.allclose(gpu,cpu,atol=1e-4,rtol=1e-4)) if cpu is not None else None
        row_order_verified=all(receipt['row_indices_sha256']==digest(tensor_save(dict(row_indices=torch.arange(rank*2,(rank+1)*2)))) for rank,receipt in enumerate(receipts))
        masked_zero_verified=bool((gpu[attention==0]==0).all())
        assert row_order_verified and masked_zero_verified, 'Actual output row order/mask verification failed'
        tensors=dict(prompts=prompts,responses=responses,attention=attention,gpu_log_probs=gpu,original_gpu_log_probs=original_gpu)
        if cpu is not None:tensors['cpu_log_probs']=cpu
        save_file(tensors,str(OUT/f'parity-round{step}.safetensors'))
        comparison=dict(scoring_started_epoch=scoring_started_epoch,scoring_completed_epoch=candidate_scoring_completed_epoch,scorers_released_epoch=scorers_released_epoch,original_oracle_completed_epoch=time.time(),step=step,rows=16,rows_per_service=2,prompt_length=int(prompts.shape[1]),response_length=rl,scope='ORIGINAL_GPU_FUNCTION_PARITY_BLOCKING',numerical_reference='ORIGINAL_GPU_FUNCTION_ORACLE',cpu_seconds=cpu_seconds,gpu_ipc_round_seconds=gpu_seconds,
            finite=bool(torch.isfinite(gpu).all()),numeric_close=close,atol=1e-4,rtol=1e-4,max_absolute_delta=float(delta.max()),mean_absolute_delta=float(delta.mean()),
            original_gpu_oracle_receipt_path=str(oracle_receipt_path),original_gpu_oracle_receipt_sha256=sha(oracle_receipt_path),
            original_gpu_oracle_output_path=str(oracle_output_path),original_gpu_oracle_output_sha256=sha(oracle_output_path),
            original_gpu_oracle_input_path=str(oracle_input),original_gpu_oracle_input_sha256=sha(oracle_input),
            original_gpu_oracle_pid=oracle.pid,original_gpu_oracle_uuid=oracle_lease['uuid'],
            cpu_cross_backend_diagnostic=(dict(scope='NON_BLOCKING_CROSS_BACKEND_DIAGNOSTIC',numeric_close=cpu_close,atol=1e-4,rtol=1e-4,
                max_absolute_delta=float(cpu_delta.max()) if cpu_delta is not None else None,mean_absolute_delta=float(cpu_delta.mean()) if cpu_delta is not None else None,
                violation_count=int((cpu_delta>(1e-4+1e-4*cpu.abs())).sum()) if cpu_delta is not None else None,
                per_row_violation_counts=[int((cpu_delta[i]>(1e-4+1e-4*cpu[i].abs())).sum()) for i in range(16)]) if cpu is not None else None),
            parity_artifact_sha256=sha(OUT/f'parity-round{step}.safetensors'),receipts_sha256=sha(OUT/f'receipts-round{step}.json'),
            row_order_verified=row_order_verified,masked_zero_verified=masked_zero_verified,
            row_order_scope='Successful LocalTeacherClient validates each raw output row_indices tensor equals expected rank range and payloadSHA, then driver independently checks receipt indicesSHA; concatenate by rank. Independent original GPU function parity checks all16 rows/all3rounds. CPU cross-backend first2 diagnostics remain nonblocking, fully retained.',
            masked_zero_scope='Computed directly on reconstructed GPU payload tensor at attention==0; all shortcase positions active so first2 rounds vacuous.',
            all8_released_zero=all(r['status']=='RELEASED' and r['allocator_released']==dict(allocated_bytes=0,reserved_bytes=0) for r in receipts),all8_fixed_parameter_sha256=receipts[0]['released_cpu']['parameter_sha256'],
            vllm_pids=[c.pid for c in children],scorer_pids=[c.pid for c in scorer_children],concurrent_http_count=len(concurrent_http))
        result['ipc_rounds'].append(comparison);result['requests'].extend(concurrent_http);sample()
        assert comparison['finite'] and close,'Native8 IPC versus original GPU function parity failed; fixed threshold retained'
    result['original_gpu_oracle_exit_code']=oracle.wait(timeout=120)
    assert result['original_gpu_oracle_exit_code']==0 and (oracle_dir/'done.json').exists(),'Original GPU oracle did not finish all3 released rounds'
    result['original_gpu_oracle_done']=json.loads((oracle_dir/'done.json').read_text())
    assert not torch.cuda.is_initialized(),'Parent hardware driver unexpectedly initialized CUDA'
    result['parent_driver_cuda_initialized']=False
    result['owned_gpu_model_pids']=dict(scorers=[c.pid for c in scorer_children],vllm=[c.pid for c in children],retriever=service.pid,original_oracle=oracle.pid)
    assert len(set([c.pid for c in scorer_children+children]+[service.pid,oracle.pid]))==18
    result['owned_gpu_model_process_count']=18
    from probe_round_checks import coexistence_evidence
    result['coexistence_samples']=coexistence_evidence(samples,scorer_manifest['services'],[c.pid for c in children],worker_mapping,service.pid)
    assert result['coexistence_samples'], 'No actual sampler evidence of all8 FP32 scorer models + all8 vLLM + full GPU retrieval co-residency'
    waitfiles(children,d,'done');result['inference']=[json.loads((d/f'done-{i}.json').read_text()) for i in range(8)]
    (d/'RELEASE').touch();assert all(c.wait(timeout=120)==0 for c in children)
    assert not monitor_errors,monitor_errors
    result['status']='HARDWARE_AFUNIX_PARITY_RELEASE_CORESIDENCY_PASS_NO_TRAINING'
except BaseException as e:
    result['status']='FAIL_NO_TRAINING';result['error']=repr(e);raise
finally:
    try:
        result['monitor_shutdown']=launcher.stop_monitor(stop,globals().get('thread'),timeout=30.)
    except Exception as e:
        result['monitor_shutdown']=dict(stopped='thread' not in globals() or not thread.is_alive(),error=repr(e),join_timeout_seconds=30.)
        result['status']='FAIL_NO_TRAINING'
        monitor_errors.append(repr(e))
    try:owned.cleanup()
    except Exception as cleanup_error:
        result['cleanup_error']=repr(cleanup_error);result['status']='FAIL_NO_TRAINING'
    try:
        remaining=sorted(launcher.cgroup_pids(scope)-owned.baseline)
        result['own_cleanup_record']=getattr(owned,'cleanup_record',None)
        result['own_children_remaining_identity']=[launcher.process_identity(pid) for pid in remaining]
        result['own_children_remaining_pids']=remaining
        result['own_children_remaining']=len(remaining)
        result['cleanup_passed']=not remaining and 'cleanup_error' not in result
        if not result['cleanup_passed']:result['status']='FAIL_NO_TRAINING'
    except Exception as e:
        result['cleanup_final_inventory_error']=repr(e);result['cleanup_passed']=False;result['status']='FAIL_NO_TRAINING'
    if ALIAS.is_symlink() and ALIAS.resolve()==OUT.resolve():ALIAS.unlink()
    http_session.close()
    phase='released';
    try:
        released=sample()
        original_apps=set(baseline['processes']) if 'baseline' in globals() else set()
        # PID/UUID presence is stable; used-memory numbers can vary naturally.
        identity=lambda lines:{tuple(v.strip() for v in line.split(',')[:2]) for line in lines if ',' in line}
        missing=identity(original_apps)-identity(released['processes'])
        result['original_gpu_process_identities_missing']=sorted(missing)
        result['baseline_preserved']=not missing
        known_owners=[c.pid for c in globals().get('scorer_children',[])]+[c.pid for c in globals().get('children',[])]
        known_owners += [globals()[name].pid for name in ('service','oracle') if name in globals()]
        result['own_gpu_model_pids_remaining']=sorted({int(line.split(',')[1].strip()) for line in released['processes'] if int(line.split(',')[1].strip()) in set(known_owners)})
        result['final_gpu_owner_check_scope']='Actual final NVML PID inventory compared with own8scorers+8vLLM+retriever+originalGPUoracle; failure startup may have fewer created owners'
        if missing or result['own_gpu_model_pids_remaining']:result['status']='FAIL_NO_TRAINING'
    except Exception as e:
        result['baseline_preserved']=False;result['final_snapshot_error']=repr(e);result['status']='FAIL_NO_TRAINING'
    result['monitor_errors']=monitor_errors
    if monitor_errors:result['status']='FAIL_NO_TRAINING'
    result['peak_total_used_mib_by_uuid']={uid:max((g['memory_used_mib'] for s in samples for g in s['gpus'] if g['uuid']==uid),default=None) for uid in UUIDS}
    result['sampled_peak_summed_rss_kib']=max((s['own_resource_envelope']['summed_rss_kib'] for s in samples),default=None)
    result['resource_scope']='Sampled summed RSS of new own cgroup processes plus protected parent driver; shared pages counted repeatedly, not unique physical RAM; 1Hz may miss transient peaks. Raw perPID CPU ticks/HWM and parent CPU-reference residency retained.'
    result['original_baseline']=baseline if 'baseline' in globals() else None
    result['owned_process_baseline']=sorted(owned.baseline)
    result['owned_slurm_cgroup']=str(scope)
    (OUT/'result.json').write_text(json.dumps(result,indent=2))
    compact={key:result[key] for key in ('job_id','status','NO_TRAINING','error','cleanup_error','cleanup_passed','own_children_remaining','own_children_remaining_identity','monitor_shutdown','baseline_preserved','final_snapshot_error','startup_seconds','sampled_peak_summed_rss_kib','peak_total_used_mib_by_uuid','worker_gpu_mapping','inference','ipc_rounds','original_gpu_process_identities_missing') if key in result}
    compact['requests']=[{key:r[key] for key in ('label','queries','topk','wall_seconds','response_sha256')} for r in result['requests']]
    compact['full_result_path']=str(OUT/'result.json')
    print(json.dumps(compact),flush=True)

if result['status']!='HARDWARE_AFUNIX_PARITY_RELEASE_CORESIDENCY_PASS_NO_TRAINING':raise RuntimeError('Own hardware probe failed: inspect result.json')
