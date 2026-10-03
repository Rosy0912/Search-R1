"""Single-node owned five-update acceptance under an exclusive Slurm job."""
import json, os, re, signal, socket, subprocess, sys, time, urllib.request
from pathlib import Path

root=Path('/dataset/shaoshitong/search-r1-rl/cosearch')
assert os.environ.get('SLURM_JOB_ID'), 'Must run inside own allocation'
assert os.environ.get('SLURM_JOB_NODELIST') == 'h100-16', 'All computation must remain on h100-16'
assert os.environ.get('SLURM_NNODES') == '1'
assert int(os.environ['SLURM_CPUS_PER_TASK']) >= 64
allocation=subprocess.check_output(['scontrol','show','job',os.environ['SLURM_JOB_ID']],text=True)
mem_match=re.search(r'MinMemoryNode=(\d+)([KMGTP]?)',allocation)
assert mem_match, 'Cannot verify actual Slurm node memory allocation'
memory_mib=int(mem_match.group(1))*{'':1,'K':1/1024,'M':1,'G':1024,'T':1024**2,'P':1024**3}[mem_match.group(2)]
assert memory_mib >= 192000
run_id=os.environ.get('COSEARCH_RUN_ID','cosearch-local-gpu-five-20261003-e')
assert re.fullmatch(r'cosearch-local-gpu-five-[0-9]{8}-[a-z][a-z0-9-]*',run_id)
run=root/'runs'/run_id
run.mkdir(parents=True,exist_ok=False)
(run/'slurm-allocation.txt').write_text(allocation)
assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))==8
for folder in ('runtime/tmp','runtime/ray','output','cache'):
    (run/folder).mkdir(parents=True,exist_ok=True)
env=os.environ.copy()
for proxy_key in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):
    env.pop(proxy_key,None)
env.update(NO_PROXY='127.0.0.1,localhost',no_proxy='127.0.0.1,localhost')
env.update(PATH=f'{root}/env/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
    SEARCH_R1_REPO=str(root/'repo'),SEARCH_R1_ASSETS=str(root/'assets-root'),
    SEARCH_R1_BASELINE_OUTPUT=str(run/'output'),SEARCH_R1_BASELINE_RUN_ROOT=str(run/'runtime'),
    COSEARCH_RANKER=str(root/'assets-root/assets/qwen2.5-3b'),
    TMPDIR=str(run/'runtime/tmp'),HF_HOME=str(root/'cache/hf'),HF_DATASETS_CACHE=str(root/'cache/hf/datasets'),
    HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
    VLLM_NO_USAGE_STATS='1',NCCL_NVLS_ENABLE='0',VLLM_ATTENTION_BACKEND='XFORMERS',WANDB_MODE='offline',
    RAY_ADDRESS='127.0.0.1:16386',RAY_USAGE_STATS_ENABLED='0')
python=str(root/'env/bin/python')
for port in (18086,16386):
    with socket.socket() as sock: sock.bind(('127.0.0.1',port))
# Parent preflight holds all eight leases before retrieval initializes CUDA.
leases=json.loads(env['SEARCH_R1_GPU_LEASES'])
lease_fds=tuple(int(record['fd']) for record in leases)
uuids=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
allocated={line.split(',')[0].strip():line.split(',')[1].strip() for line in uuids.splitlines()}
selected=[allocated.get(v.strip(),v.strip()) for v in env['CUDA_VISIBLE_DEVICES'].split(',')]
assert set(selected)=={record['uuid'] for record in leases}
env.update(COSEARCH_RANKER_GPU_UUID=selected[-1],COSEARCH_RANKER_LEASE_UUIDS=json.dumps(selected))
# Do not use Ray's host-wide stop command. Slurm owns all child processes.
children=[]
logs=[]
def launch(command,filename,child_env):
    log=(run/filename).open('w');logs.append(log)
    p=subprocess.Popen(command,env=child_env,cwd=root/'repo',stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=lease_fds)
    children.append((p,os.pidfd_open(p.pid)))
    return p
retrieval_env=env.copy();retrieval_env.update(PYTHONPATH=f'{root}/retriever-gpu-lib:{root}/repo',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8')
try:
    service=launch([python,str(root/'repo/reproduction/h20/retriever_local.py'),
        '--index_path',str(root/'assets-root/assets/wiki/e5_Flat.index'),
        '--corpus_path',str(root/'assets-root/assets/wiki/wiki-18.jsonl'),
        '--retriever_name','e5','--retriever_model',str(root/'assets-root/assets/e5-base-v2'),
        '--topk','3','--device','cuda','--faiss_gpu','--threads','8','--host','127.0.0.1','--port','18086'],
        'retriever.log',retrieval_env)
    started=time.monotonic()
    while True:
        if service.poll() is not None: raise RuntimeError('Own local retriever failed')
        try:
            with urllib.request.urlopen('http://127.0.0.1:18086/openapi.json',timeout=5) as res:
                if res.status==200: break
        except OSError: pass
        if time.monotonic()-started>1200: raise TimeoutError('Local retriever startup timeout')
        time.sleep(5)
    probe=launch([python,str(root/'tools/native_gpu_ranker_probe.py')],'native-ranker-probe.log',dict(env,COSEARCH_PROBE_RUN=str(run)))
    while probe.poll() is None:
        if service.poll() is not None: raise RuntimeError('Retriever failed during GPU ranker probe')
        time.sleep(5)
    if probe.returncode != 0: raise RuntimeError('Native GPU ranker numerical/resource probe failed')
    if os.environ.get('COSEARCH_PROBE_ONLY') == '1':
        print('COSEARCH_PROBE_ONLY_COMPLETE',flush=True)
        raise SystemExit(0)
    import psutil
    ray_node_ip='10.33.0.16'
    local_ips={address.address for addresses in psutil.net_if_addrs().values() for address in addresses}
    assert ray_node_ip in local_ips, 'Ray advertised address must be an actual h100-16 interface'
    head=launch([python,str(root/'repo/reproduction/baselines/agent6-20261002/ray_cli.py'),'start','--head','--block','--node-ip-address='+ray_node_ip,
        '--port=16386','--min-worker-port=21000','--max-worker-port=21999','--num-gpus=8','--num-cpus=24','--object-store-memory=8589934592',
        '--include-dashboard=false','--temp-dir='+str(root)], 'ray-head.log',env)
    started=time.monotonic()
    while True:
        if head.poll() is not None: raise RuntimeError('Own Ray head failed')
        try:
            result=subprocess.run([python,str(root/'repo/reproduction/baselines/agent6-20261002/ray_cli.py'),'status','--address=127.0.0.1:16386'],env=env,
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=15)
        except subprocess.TimeoutExpired:
            if time.monotonic()-started>180: raise TimeoutError('Ray status startup timeout')
            continue
        if result.returncode==0: break
        if time.monotonic()-started>180: raise TimeoutError('Ray startup timeout')
        time.sleep(5)
    command=['bash','reproduction/baselines/run.sh','cosearch',
        'actor_rollout_ref.actor.ppo_micro_batch_size=64',
        'actor_rollout_ref.rollout.log_prob_micro_batch_size=128',
        'actor_rollout_ref.ref.log_prob_micro_batch_size=128',
        'retriever.url=http://127.0.0.1:18086/retrieve',
        'baselines.ranker.attn_implementation=sdpa','baselines.ranker.cpu_threads=32',
        'baselines.ranker.response_only_logits=true','baselines.ranker.phase_gpu=true',
        'baselines.ranker.reuse_generation_logits=true','baselines.ranker.inference_batch_size=4',
        'baselines.ranker.inference_token_budget=32768','baselines.ranker.gradient_checkpointing=true',
        'baselines.local_runtime_only=true','baselines.local_runtime_host=h100-16','trainer.total_training_steps=6',
        'data.val_data_num=64','data.val_batch_size=64','trainer.val_before_train=false','trainer.smoke_only=true','trainer.test_freq=5',
        'trainer.save_freq=100','trainer.acceptance_step=5',
        'trainer.experiment_name='+run.name]
    (run/'launch.json').write_text(json.dumps(dict(job_id=os.environ['SLURM_JOB_ID'],
        gpu_devices=os.environ['CUDA_VISIBLE_DEVICES'],command=command,local_node_only=True,
        endpoints=['127.0.0.1:18086','127.0.0.1:16386'],ranker='same base FP32 phase-shared local GPU, CPU resident between phases',retrieval='same-node CUDA FP16 E5 + eight-GPU sharded FP16 FAISS',reference_commit='0328f941547fa8b41f4fe5a7f84499c7c67a7bee',
        cpu_allocation=os.environ.get('SLURM_CPUS_PER_TASK'),memory_mb=memory_mib,
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=root/'repo').strip()),indent=2)+'\n')
    # Inspect and preserve the exact effective Hydra configuration before models load.
    with (run/'resolved.yaml').open('w') as output:
        subprocess.run(command+['--cfg','job','--resolve'],env=env,cwd=root/'repo',stdout=output,check=True)
    audit_code = "from omegaconf import OmegaConf; from search_r1.baselines.holdout import prepare_common_holdout; import hashlib,json,sys; c=OmegaConf.load(sys.argv[1]); m=prepare_common_holdout(c); t=OmegaConf.to_yaml(OmegaConf.create(OmegaConf.to_container(c,resolve=True))); print('AUDIT_IDENTITY='+json.dumps(dict(config=hashlib.sha256(t.encode()).hexdigest(),data=m['content_identity']['sha256'])))"
    audit_output=subprocess.check_output([python,'-c',audit_code,str(run/'resolved.yaml')],env=env,cwd=root/'repo',text=True)
    audit=json.loads([line.split('=',1)[1] for line in audit_output.splitlines() if line.startswith('AUDIT_IDENTITY=')][0])
    env.update(SEARCH_R1_BASELINE_AUDIT_DIR=str(run/'audit'),SEARCH_R1_BASELINE_RUN_ID=run.name,
        SEARCH_R1_BASELINE_COMMIT=subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=root/'repo').strip(),
        SEARCH_R1_BASELINE_CONFIG_SHA256=audit['config'],SEARCH_R1_BASELINE_DATA_IDENTITY=audit['data'])
    driver=launch(command,'training.log',env)
    print('LOCAL_FIVE_DRIVER_STARTED',json.dumps(dict(job=os.environ['SLURM_JOB_ID'],pid=driver.pid,run=str(run))),flush=True)
    peaks={};last_progress=0
    while driver.poll() is None:
        if service.poll() is not None or head.poll() is not None:
            raise RuntimeError('Own local retrieval/Ray child exited during training')
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],env=env,text=True)
        for line in raw.splitlines():
            uid,mem,util=[x.strip() for x in line.split(',')]
            old=peaks.setdefault(uid,dict(memory_peak_mib=0,utilization_peak_percent=0))
            old['memory_peak_mib']=max(old['memory_peak_mib'],float(mem))
            old['utilization_peak_percent']=max(old['utilization_peak_percent'],float(util))
        (run/'gpu-peaks.json').write_text(json.dumps(peaks,indent=2)+'\n')
        if time.monotonic()-last_progress>60:
            print('LOCAL_FIVE_ACTIVE',time.time(),flush=True);last_progress=time.monotonic()
        time.sleep(10)
    (run/'driver-exit.json').write_text(json.dumps(dict(returncode=driver.returncode,finished_at=time.time(),
        acceptance_passed=False,note='Five-update metrics and checkpoint reload still require verification'))+'\n')
    raise SystemExit(driver.returncode)
finally:
    for p,fd in reversed(children):
        if p.poll() is None:
            try: signal.pidfd_send_signal(fd,signal.SIGTERM)
            except ProcessLookupError: pass
            try: p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try: signal.pidfd_send_signal(fd,signal.SIGKILL)
                except ProcessLookupError: pass
                p.wait()
        os.close(fd)
    for log in logs: log.close()
