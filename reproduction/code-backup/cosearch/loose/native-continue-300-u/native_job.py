"""Owned Slurm resource/service wrapper for the unchanged author checkout.

Not an acceptance checker. No GPU work is permitted outside the allocation.
"""
import hashlib, json, os, re, shutil, signal, socket, subprocess, time, urllib.request
from pathlib import Path

BASE=Path('/dataset/shaoshitong/search-r1-rl/cosearch')
ROOT=BASE/'author-native'

def sanitize_runtime_env(env):
    # Preserve Slurm/Ray CUDA assignments; remove conflicting vendor aliases and bypass flags.
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy', 'ROCR_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'GPU_DEVICE_ORDINAL', 'RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES', 'RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES', 'RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES', 'RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES', 'RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES', 'RAY_EXPERIMENTAL_NOSET_NEURON_RT_VISIBLE_CORES', 'RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS', 'RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR'):
        env.pop(key, None)
    return env

def main():
    ready=json.loads((ROOT/'tools/preparation-ready.json').read_text())
    assert ready['cpu_preparation_passed'] and not ready['actual_gpu_resources_tested']
    for name,expected in ready['tools_sha256'].items():
        assert hashlib.sha256((ROOT/'tools'/name).read_bytes()).hexdigest()==expected,name
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==ready['tools_sha256']['native_job.py']
    assert os.environ.get('SLURM_JOB_ID')
    assert os.environ.get('SLURM_JOB_NODELIST')=='h100-16'
    assert os.environ.get('SLURM_NNODES')=='1'
    assert int(os.environ['SLURM_CPUS_PER_TASK'])>=64
    allocation=subprocess.check_output(['scontrol','show','job',os.environ['SLURM_JOB_ID']],text=True)
    match=re.search(r'MinMemoryNode=(\d+)([KMGTP]?)',allocation)
    assert match
    memory=int(match[1])*{'':1,'K':1/1024,'M':1,'G':1024,'T':1024**2,'P':1024**3}[match[2]]
    assert memory>=192000
    mask=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    assert len(mask)==8 and len(set(mask))==8 and all(x.isdigit() for x in mask)
    leases=json.loads(os.environ['SEARCH_R1_GPU_LEASES'])
    assert len(leases)==8 and os.environ['SEARCH_R1_GPU_PREFLIGHT']=='1'
    fds=tuple(int(x['fd']) for x in leases)
    run_id=os.environ.get('COSEARCH_NATIVE_RUN_ID','author-native-raw-five')
    assert re.fullmatch(r'author-native-[a-z0-9-]+',run_id)
    continue_to=os.environ.get('COSEARCH_NATIVE_CONTINUE_TO')
    if continue_to is not None:
        assert continue_to=='300' and os.environ.get('COSEARCH_NATIVE_RELOAD_ONLY')!='1'
    if os.environ.get('COSEARCH_NATIVE_RELOAD_ONLY')=='1' or continue_to:
        source_id=os.environ['COSEARCH_NATIVE_SOURCE_RUN_ID']
        assert source_id!=run_id and re.fullmatch(r'author-native-[a-z0-9-]+',source_id)
        source=ROOT/'runs'/source_id
        verified=json.loads((ROOT/'tools/source-checkpoint-verification.json').read_text())
        assert verified['source_run']==source_id and verified['actual_full_file_shas_reverified'] and verified['ranks']==8
        assert str(verified['source_job']) not in subprocess.check_output(['squeue','-h','-o','%i'],text=True).split(), 'Source allocation still active'
        step=int(os.environ.get('COSEARCH_NATIVE_RESUME_STEP','5')) if continue_to else 5
        assert 0<step<300
        checkpoint=source/'checkpoints'/f'global_step_{step}'
        assert verified['source_step']==step and verified['checkpoint']==str(checkpoint)
        assert checkpoint.is_dir() and hashlib.sha256((checkpoint/'data.pt').read_bytes()).hexdigest()==verified['data_sha256']
        if continue_to:assert (checkpoint/'data.pt').is_file()
    run=ROOT/'runs'/run_id
    run.mkdir(parents=True,exist_ok=False)
    (run/'evidence').mkdir()
    (run/'tools').mkdir()
    staged={}
    for path in (ROOT/'tools').iterdir():
        if path.is_file() and path.suffix in ['.py','.yaml','.json','.sh']:
            destination=run/'tools'/path.name
            shutil.copy2(path,destination)
            staged[path.name]=hashlib.sha256(destination.read_bytes()).hexdigest()
    for name,expected in ready['tools_sha256'].items():
        assert staged[name]==expected,('Frozen tool mismatch',name)
    (run/'external-tool-manifest.json').write_text(json.dumps(staged,indent=2))
    (run/'slurm-allocation.txt').write_text(allocation)
    inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    indices={x.split(',')[0].strip():x.split(',')[1].strip() for x in inventory.splitlines()}
    uuids=[indices[x] for x in mask]
    assert set(uuids)=={x['uuid'] for x in leases}
    import secrets
    auth_path=run/'ray-auth-token'
    auth_fd=os.open(auth_path, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
    with os.fdopen(auth_fd, 'w') as auth_stream:auth_stream.write(secrets.token_hex(32))
    env=os.environ.copy()
    sanitize_runtime_env(env)
    env.pop('RAY_AUTH_TOKEN', None)
    env.update(RAY_AUTH_MODE='token', RAY_AUTH_TOKEN_PATH=str(auth_path))
    import psutil
    assert '10.33.0.16' in {a.address for a in psutil.net_if_addrs().get('bond1', [])}, 'Private Ray interface must have a real local address'
    assert psutil.net_if_stats()['bond1'].isup, 'Private Ray interface must be up'
    env.update(GLOO_SOCKET_IFNAME='bond1', NCCL_SOCKET_IFNAME='bond1')
    env.update(NO_PROXY='127.0.0.1,localhost,10.33.0.16',no_proxy='127.0.0.1,localhost,10.33.0.16',
        HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1',RAY_USAGE_STATS_ENABLED='0',VLLM_NO_USAGE_STATS='1',
        RAY_ADDRESS='127.0.0.1:16386',COSEARCH_NATIVE_EVIDENCE_ROOT=str(run/'evidence'),
        COSEARCH_NATIVE_RUN_ID=run_id,COSEARCH_LEASED_UUIDS=','.join(uuids),
        COSEARCH_LEASED_UUID_TO_INDEX=json.dumps({indices[x]:int(x) for x in mask}),
        PYTHONPATH=f'{run}/tools:{ROOT}/repo:{ROOT}/repo/verl',
        PATH=f'{ROOT}/env/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin')
    python=str(ROOT/'env/bin/python')
    oldpython=str(BASE/'env/bin/python')
    for port in (18086,16386):
        with socket.socket() as sock:sock.bind(('127.0.0.1',port))
    processes=[];logs=[]
    def launch(command,name,child_env):
        stream=(run/name).open('w');logs.append(stream)
        child=subprocess.Popen(command,cwd=ROOT/'repo',env=child_env,stdout=stream,
            stderr=subprocess.STDOUT,start_new_session=True,pass_fds=fds)
        processes.append((child,os.pidfd_open(child.pid)))
        return child
    started=time.monotonic()
    try:
        retrieval_env=dict(env,PYTHONPATH=f'{BASE}/retriever-gpu-lib:{BASE}/repo',
            OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',
            PATH=f'{BASE}/env/bin:'+env['PATH'])
        service=launch([oldpython,str(BASE/'repo/reproduction/h20/retriever_local.py'),
            '--index_path',str(BASE/'assets-root/assets/wiki/e5_Flat.index'),
            '--corpus_path',str(BASE/'assets-root/assets/wiki/wiki-18.jsonl'),
            '--retriever_name','e5','--retriever_model',str(BASE/'assets-root/assets/e5-base-v2'),
            '--topk','50','--device','cuda','--faiss_gpu','--threads','8',
            '--host','127.0.0.1','--port','18086'],'retriever.log',retrieval_env)
        deadline=time.monotonic()+1200
        while True:
            assert service.poll() is None,'Owned retriever exited'
            try:
                with urllib.request.urlopen('http://127.0.0.1:18086/openapi.json',timeout=5) as response:
                    if response.status==200:break
            except OSError:pass
            assert time.monotonic()<deadline,'Retriever startup deadline'
            time.sleep(5)
        import psutil
        assert '10.33.0.16' in {a.address for v in psutil.net_if_addrs().values() for a in v}
        # Ray CLI belongs to the new native environment; no host-wide ray stop.
        head=launch([str(ROOT/'env/bin/ray'),'start','--head','--block','--node-ip-address=10.33.0.16',
            '--port=16386','--min-worker-port=21000','--max-worker-port=21999',
            '--num-gpus=8','--num-cpus=48','--object-store-memory=8589934592',
            '--include-dashboard=false','--temp-dir='+str(BASE)],'ray-head.log',env)
        deadline=time.monotonic()+180
        while True:
            assert head.poll() is None,'Owned Ray head exited'
            result=subprocess.run([str(ROOT/'env/bin/ray'),'status','--address=127.0.0.1:16386'],
                env=env,capture_output=True,timeout=20)
            if result.returncode==0:break
            (run/'ray-status-error.txt').write_text(result.stderr.decode('utf-8', errors='replace')[-12000:])
            assert time.monotonic()<deadline,'Ray startup deadline'
            time.sleep(5)
        (run/'launch.json').write_text(json.dumps(dict(job=os.environ['SLURM_JOB_ID'],
            node='h100-16',allocated_devices=mask,leased_uuids=uuids,
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT/'repo',text=True).strip(),
            source_manifest=json.loads((run/'tools/source-manifest.json').read_text()),
            startup_seconds=time.monotonic()-started,acceptance_passed=False),indent=2))
        # Driver uses owned UUIDs; Ray workers get their own numeric resource mask.
        driver_env=dict(env,CUDA_VISIBLE_DEVICES=','.join(uuids))
        driver_script='probe_gpu_routing.py' if os.environ.get('COSEARCH_NATIVE_GPU_PROBE_ONLY')=='1' else 'launch_observed.py'
        driver=launch([python,str(run/'tools'/driver_script)],'training.log',driver_env)
        while driver.poll() is None:
            assert service.poll() is None and head.poll() is None,'Owned service exited during training'
            raw=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,utilization.gpu',
                '--format=csv,noheader,nounits'],text=True)
            (run/'gpu-latest.txt').write_text(raw)
            time.sleep(10)
        (run/'driver-exit.json').write_text(json.dumps(dict(returncode=driver.returncode,
            finished_at=time.time(),total_wall_seconds=time.monotonic()-started,acceptance_passed=False)))
        return driver.returncode
    except BaseException as error:
        (run/'startup-failure.json').write_text(json.dumps(dict(error_type=type(error).__name__,error=str(error),phase='service_startup_or_driver_supervision',acceptance_passed=False)))
        if not (run/'driver-exit.json').exists():
            (run/'driver-exit.json').write_text(json.dumps(dict(returncode=1,total_wall_seconds=time.monotonic()-started,phase='startup',acceptance_passed=False)))
        raise
    finally:
        for child,fd in reversed(processes):
            if child.poll() is None:
                try:signal.pidfd_send_signal(fd,signal.SIGTERM)
                except ProcessLookupError:pass
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    try:signal.pidfd_send_signal(fd,signal.SIGKILL)
                    except ProcessLookupError:pass
                    child.wait()
            os.close(fd)
        for stream in logs:stream.close()

if __name__=='__main__':raise SystemExit(main())
