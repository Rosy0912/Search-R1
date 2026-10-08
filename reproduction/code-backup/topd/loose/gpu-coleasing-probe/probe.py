"""GPU FAISS/E5 and eight independent vLLM TP1 co-residency, NO_TRAINING."""
import concurrent.futures,hashlib,json,math,os,signal,socket,subprocess,sys,threading,time
from pathlib import Path
import pandas as pd
import requests
from probe_owned import OwnedProcesses,cgroup_scope
from probe_data import resolve_train,sha
from probe_gpu_mapping import numeric_worker_mapping
ROOT=Path(os.environ['AGENT_ROOT']).resolve();JOB=os.environ['SLURM_JOB_ID']
assert ROOT==Path('/dataset/shaoshitong/search-r1-rl/topd'), 'Unexpected own root'
assert socket.gethostname().split('.')[0]=='h100-11' and os.environ.get('SLURM_JOB_NODELIST')=='h100-11', 'Require own h100-11 allocation'
assert int(os.environ.get('SLURM_CPUS_PER_TASK','0'))>=64 and int(os.environ.get('SLURM_MEM_PER_NODE','0'))>=192*1024, 'Require allocated64CPU/192GiB'
assert int(os.environ.get('SLURM_GPUS_ON_NODE','0'))==8 and len(os.environ.get('SLURM_JOB_GPUS','').split(','))==8, 'Require real Slurm GRES8'
assert len(os.environ.get('CUDA_VISIBLE_DEVICES','').split(','))==8, 'Require exactly8 allocated visible devices'
OUT=ROOT/'logs'/f'gpu-coexist-{JOB}';OUT.mkdir(exist_ok=False)
assert os.environ.get('SEARCH_R1_GPU_PREFLIGHT')=='1'
leases=json.loads(os.environ['SEARCH_R1_GPU_LEASES']);assert len(leases)==8
for item in leases:os.fstat(int(item['fd']))
UUIDS={x['uuid'] for x in leases};scope,relative=cgroup_scope(JOB);owned=OwnedProcesses(scope,relative)
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
    return dict(time=time.time(),phase=phase,gpus=result,processes=apps.splitlines())
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
    tick=time.perf_counter();r=http_session.post('http://127.0.0.1:18085/retrieve',json=dict(queries=queries,topk=k,return_scores=True),timeout=(5,900));r.raise_for_status();payload=r.json()
    assert len(payload['result'])==len(queries)
    for row in payload['result']:
        assert len(row)==k
        assert all(isinstance(x['document']['contents'],str) and math.isfinite(float(x['score'])) for x in row)
    result=dict(label=label,queries=len(queries),topk=k,wall_seconds=time.perf_counter()-tick,
        response_sha256=hashlib.sha256(r.content).hexdigest(),
        content_sha256=[[hashlib.sha256(v['document']['contents'].encode()).hexdigest() for v in row] for row in payload['result']])
    (OUT/(label+'.json')).write_text(json.dumps(result,indent=2));return result
def interrupted(signum,frame):raise RuntimeError(f'Own hardware probe interrupted by signal {signum}')
signal.signal(signal.SIGTERM,interrupted)
signal.signal(signal.SIGINT,interrupted)
result=dict(job_id=JOB,NO_TRAINING=True,scope='real full-index GPU retrieval + base3B inference only; no actor optimizer/FSDP training residency',leases=[{'uuid':i['uuid']} for i in leases],requests=[],script_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*') if p.is_file() and p.suffix in ('.py','.sbatch')},common_config_sha256={name:sha(ROOT/'repo'/name) for name in ['verl/trainer/config/ppo_baselines.yaml','verl/trainer/config/ppo_trainer.yaml','verl/workers/rollout/vllm_rollout/vllm_rollout.py']})
try:
    baseline=sample();worker_mapping=numeric_worker_mapping(baseline['gpus'],UUIDS,os.environ['CUDA_VISIBLE_DEVICES'].split(','));result['worker_gpu_mapping']=worker_mapping;assert all(g['memory_used_mib']/g['memory_total_mib']<=.15 and g['utilization']<=15 for g in baseline['gpus'])
    from hydra import compose,initialize_config_dir
    from omegaconf import OmegaConf
    os.environ['SEARCH_R1_ASSETS']=str(ROOT/'assets-root')
    with initialize_config_dir(config_dir=str(ROOT/'repo/verl/trainer/config'),version_base=None):
        config=compose(config_name='ppo_baselines')
    rollout=config.actor_rollout_ref.rollout
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
        return spawn([sys.executable,str(ROOT/'tools/gpu-coleasing-probe/vllm-worker.py'),'--directory',str(directory),'--rank',str(rank)],wenv,directory.name+f'-rank{rank}')
    # First one .6 vLLM proves startup coexistence before requesting all eight.
    d=OUT/'single-vllm';d.mkdir();(d/'queries.json').write_text(json.dumps(queries));(d/'rollout-config.json').write_text(json.dumps(effective));phase='single-vllm-startup'
    c=worker(0,d);waitfiles([c],d,'ready');phase='single-vllm-idle';sample();(d/'GO').touch();waitfiles([c],d,'done');sample();(d/'RELEASE').touch();assert c.wait(timeout=120)==0
    d=OUT/'eight-vllm';d.mkdir();(d/'queries.json').write_text(json.dumps(queries));(d/'rollout-config.json').write_text(json.dumps(effective));phase='eight-vllm-startup'
    children=[worker(i,d) for i in range(8)];waitfiles(children,d,'ready');phase='eight-vllm-faiss-e5-idle';time.sleep(3);sample()
    phase='eight-vllm-inference-plus-gpu-http512';(d/'GO').touch()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        request=pool.submit(http,queries,3,'coexist-top3-512');waitfiles(children,d,'done');result['requests'].append(request.result())
    sample();result['inference']=[json.loads((d/f'done-{i}.json').read_text()) for i in range(8)];(d/'RELEASE').touch()
    assert all(c.wait(timeout=120)==0 for c in children);assert not monitor_errors,monitor_errors
    result['status']='HARDWARE_CORESIDENCY_PASS_NO_TRAINING'
except BaseException as e:
    result['status']='FAIL_NO_TRAINING';result['error']=repr(e);raise
finally:
    stop.set()
    if 'thread' in globals():thread.join(timeout=15)
    try:owned.cleanup()
    except Exception as cleanup_error:
        result['cleanup_error']=repr(cleanup_error);result['status']='FAIL_NO_TRAINING'
    http_session.close()
    phase='released';
    try:
        released=sample()
        original_apps=set(baseline['processes']) if 'baseline' in globals() else set()
        # PID/UUID presence is stable; used-memory numbers can vary naturally.
        identity=lambda lines:{tuple(v.strip() for v in line.split(',')[:2]) for line in lines if ',' in line}
        missing=identity(original_apps)-identity(released['processes'])
        result['original_gpu_process_identities_missing']=sorted(missing)
        if missing:result['status']='FAIL_NO_TRAINING'
    except Exception as e:
        result['final_snapshot_error']=repr(e);result['status']='FAIL_NO_TRAINING'
    result['monitor_errors']=monitor_errors
    result['peak_total_used_mib_by_uuid']={uid:max((g['memory_used_mib'] for s in samples for g in s['gpus'] if g['uuid']==uid),default=None) for uid in UUIDS}
    result['original_baseline']=baseline if 'baseline' in globals() else None
    result['owned_process_baseline']=sorted(owned.baseline)
    result['owned_slurm_cgroup']=str(scope)
    (OUT/'result.json').write_text(json.dumps(result,indent=2))
    compact={key:result[key] for key in ('job_id','status','NO_TRAINING','error','cleanup_error','final_snapshot_error','startup_seconds','peak_total_used_mib_by_uuid','worker_gpu_mapping','inference','original_gpu_process_identities_missing') if key in result}
    compact['requests']=[{key:r[key] for key in ('label','queries','topk','wall_seconds','response_sha256')} for r in result['requests']]
    compact['full_result_path']=str(OUT/'result.json')
    print(json.dumps(compact),flush=True)

if result['status']!='HARDWARE_CORESIDENCY_PASS_NO_TRAINING':raise RuntimeError('Own hardware probe failed: inspect result.json')
