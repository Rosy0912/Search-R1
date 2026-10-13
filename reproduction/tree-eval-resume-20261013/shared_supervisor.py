"""Separate systemd cgroup for explicit latest human GPU-sharing authorization.

Does not change Slurm's device restrictions or foreign jobs, and makes no
exclusive/Slurm GPU reservation claim. Own UUID leases prevent duplicate Tree
instances. Memory guard yields by stopping only our evaluator if headroom falls.
"""
import argparse,fcntl,json,os,signal,socket,subprocess,sys,time
from pathlib import Path
from supervisor import ROOT,OLD,FROZEN,HERE,require,sha,write

def main():
    p=argparse.ArgumentParser()
    for n in ('unit','prefix','manifest','release','score-output'):p.add_argument('--'+n,required=True)
    a=p.parse_args()
    require(socket.gethostname().split('.')[0]=='h100-10','Actual inventory chooses asset-local h100-10')
    require(a.unit.startswith('sst-tree-eval-shared-') and a.unit.endswith('.service'),'Wrong owned unit')
    lines=Path('/proc/self/cgroup').read_text().splitlines();relative=None;scope=None
    for line in lines:
        _,controllers,path=line.split(':',2)
        if a.unit in path.split('/') and (controllers=='' or 'name=systemd' in controllers.split(',') or 'freezer' in controllers.split(',')):
            base='/sys/fs/cgroup' if controllers=='' else '/sys/fs/cgroup/systemd' if 'name=systemd' in controllers.split(',') else '/sys/fs/cgroup/freezer'
            candidate=Path(base)/path.lstrip('/')
            if (candidate/'cgroup.procs').is_file():relative=path;scope=candidate
    require(scope is not None and (scope/'cgroup.procs').is_file(),'Own systemd cgroup unavailable')
    require(not any('job_' in l.split(':',2)[-1] for l in lines),'Shared runtime must not impersonate foreign Slurm job')
    active_ids=set(subprocess.check_output(['squeue','--local','-h','-o','%i'],text=True).split())
    require(not active_ids.intersection({'3360','3392'}),'Old/queued exclusive evaluator must be terminal before shared startup')
    release=json.loads(Path(a.release).read_text())
    require(release['review_status']=='ACCEPTED' and release['resource_protocol']=='memory_headroom_shared_v1','Reviewed sharing authorization missing')
    for n,h in release['files'].items():require(sha(HERE/n)==h,'Shared reviewed source changed: '+n)
    for n,h in release['external_files'].items():require(sha(Path(n))==h,'External source changed')
    package=FROZEN.parent.parent
    require(sha(package/'manifest.json')=='e2503977c8746edbc83b722d8224548f060909df9fc25d91a9cb88a947a79a21','Original package changed')
    for n,h in json.loads((package/'manifest.json').read_text())['files'].items():require(sha(package/n)==h,'Frozen source changed: '+n)
    require(sha(a.prefix)==release['prefix_sha256'] and sha(a.manifest)==release['manifest_sha256'],'Preserved scores/asset manifest changed')
    done=json.loads(Path(a.prefix+'.done.json').read_text())
    require(done['sha256']==release['prefix_sha256'] and done['actual_trained_outer']==200 and done['identity']['model_sha256']==release['model_sha256'],'Actual trained200 HF binding absent')
    scores=Path(a.score_output).resolve();require(scores.is_relative_to((OLD/'output/eval-resume').resolve()) and scores==Path(release['score_output']).resolve(),'Own persistent score path differs')
    scores.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(ROOT/'tools'))
    from tree_shared_owned_runtime_fixed import OwnedProcesses,available,gpu_sample
    owned=OwnedProcesses(scope,relative)
    import pynvml as nv
    nv.nvmlInit();selected=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    expected=json.loads((HERE/'expected-gpu-uuids-retained18.json').read_text())
    require(len(selected)==8 and set(selected)==set(expected),'Eight physical UUID identities differ')
    lockdir=Path('/tmp')/f'search-r1-gpu-leases-{os.getuid()}';lockdir.mkdir(mode=0o700,exist_ok=True)
    locks=[];records=[]
    for uid in sorted(selected):
        fd=os.open(lockdir/(uid+'.lock'),os.O_CREAT|os.O_RDWR,0o600);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);os.set_inheritable(fd,True);locks.append(fd);records.append(dict(uuid=uid,fd=fd))
    for uid in selected:
        h=nv.nvmlDeviceGetHandleByUUID(uid.encode());mem=nv.nvmlDeviceGetMemoryInfo(h)
        require(mem.free>=40*1024**3,'Need40GiB current free per GPU for bounded model+GPU retrieval and headroom')
    run=ROOT/'runs'/a.unit.removesuffix('.service');run.mkdir()
    for n in ('output','cache','tmp'):(run/n).mkdir()
    env=dict(os.environ)
    for k in list(env):
        if k.startswith(('SLURM_','SBATCH_','SQUEUE_','SCANCEL_','SACCT_','SCONTROL_')) or k in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):env.pop(k,None)
    env.update(PYTHONUNBUFFERED='1',PYTHONOPTIMIZE='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
      TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',VLLM_NO_USAGE_STATS='1',VLLM_ATTENTION_BACKEND='XFORMERS',NCCL_NVLS_ENABLE='0',
      NO_PROXY='127.0.0.1,localhost,::1',no_proxy='127.0.0.1,localhost,::1',TMPDIR=str(run/'tmp'),XDG_CACHE_HOME=str(run/'cache'),HF_HOME=str(run/'cache/hf'),
      PYTHONPATH=str(FROZEN)+':'+str(ROOT/'tools'),SEARCH_R1_BASELINE_COMMIT='8b6f99d1b3353145200eb3fa7c16a579a79a987c',
      SEARCH_R1_BASELINE_CONFIG_SHA256=sha(OLD/'resolved.yaml'),SEARCH_R1_BASELINE_DATA_IDENTITY='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24',
      SEARCH_R1_MODEL_IDENTITY=release['model_sha256'],SEARCH_R1_BASELINE_RUN_ID=run.name,SEARCH_R1_BASELINE_OUTPUT=str(scores),
      SEARCH_R1_GPU_LEASES=json.dumps(records),SEARCH_R1_GPU_PREFLIGHT='1',TREE_EVAL_SHARED_MEMORY_MODE='1',
      TREE_EVAL_RESOURCE_PROTOCOL='memory_headroom_shared_v1',TREE_EVAL_SYSTEMD_UNIT=a.unit,SLURM_JOB_ID='not-slurm;systemd:'+a.unit)
    available(18082)
    write(run/'allocation-identity.json',dict(unit=a.unit,allocation_kind='user_authorized_shared_systemd',physical_gpus=8,slurm_reserved_gpus=0,
      cgroup=relative,initial_gpu=gpu_sample(),release_sha256=sha(a.release),score_output=str(scores),training_updates=200,
      initial_required_free_gib=40,runtime_required_headroom_gib=12,own_memory_cap_gib=36,inference_gpu_memory_utilization=.20,foreign_sm_limit=None))
    result=dict(status='FAILED',unit=a.unit,predecessor_job=3360,evaluation_only=True,training_updates=200,start_epoch=time.time(),score_output=str(scores),
      allocation_kind='user_authorized_shared_systemd',physical_gpus=8,slurm_reserved_gpus=0,resource_protocol='memory_headroom_shared_v1',prefix_sha256=sha(a.prefix))
    def interrupted(sig,frame):raise RuntimeError('Own shared evaluation interrupted '+str(sig))
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,interrupted)
    try:
        guard=owned.spawn([sys.executable,str(HERE/'shared_guard.py'),'--output',str(run/'resource-samples.jsonl'),'--unit',a.unit],run/'guard.log',env)
        deadline=time.monotonic()+30
        while not (run/'resource-samples.ready').exists():
            require(guard.poll() is None and time.monotonic()<deadline,'Shared memory guard failed');time.sleep(.2)
        retriever=owned.spawn([sys.executable,str(ROOT/'repo/reproduction/h20/retriever_local.py'),'--index_path',str(ROOT/'assets-root/assets/wiki/e5_Flat.index'),
          '--corpus_path',str(ROOT/'assets-root/assets/wiki/wiki-18.jsonl'),'--retriever_model',str(ROOT/'assets-root/assets/e5-base-v2'),'--retriever_name','e5','--topk','3',
          '--faiss_gpu','--device','cuda','--threads','8','--host','127.0.0.1','--port','18082'],run/'retriever.log',
          dict(env,PYTHONPATH=str(ROOT/'retriever-gpu-lib')+':'+str(FROZEN),OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8'),pass_fds=tuple(locks))
        import requests
        session=requests.Session();session.trust_env=False;deadline=time.monotonic()+900
        while True:
            require(guard.poll() is None and retriever.poll() is None and time.monotonic()<deadline,'Shared retriever startup failed')
            try:
                if session.get('http://127.0.0.1:18082/openapi.json',timeout=2).ok:break
            except requests.RequestException:pass
            time.sleep(1)
        common=['--config',str(OLD/'resolved.yaml'),'--prefix',a.prefix,'--manifest',a.manifest,'--model',str(OLD/'checkpoints/actor/global_step_200'),'--output',str(scores)]
        worker=owned.spawn([sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=8',str(HERE/'eval_worker.py'),*common,'--local-batch-size','64'],run/'eval.log',env,pass_fds=tuple(locks))
        while worker.poll() is None:
            require(guard.poll() is None and retriever.poll() is None,'Own shared service stopped')
            with (run/'resource-samples.jsonl').open('rb') as f:f.seek(0,2);f.seek(max(0,f.tell()-32768));latest=json.loads(f.read().decode().splitlines()[-1])
            require(time.time()-latest['epoch']<10 and not latest.get('event'),'Shared guard stale or insufficient memory');time.sleep(1)
        result['worker_exit_code']=worker.returncode;require(worker.returncode==0,'Shared workers failed; preserve all committed batches')
        subprocess.run([sys.executable,str(HERE/'eval_merge.py'),*common],env=env,check=True)
        require((scores/'final-metrics.json').is_file(),'Full final result absent');result['status']='COMPLETE'
    except BaseException as err:result['error']=repr(err);raise
    finally:
        try:owned.cleanup();result['own_cleanup_residual']=owned.cleanup_residual;result['final_gpu']=gpu_sample()
        finally:
            result['end_epoch']=time.time();result['wall_seconds']=result['end_epoch']-result['start_epoch'];write(run/'run-result.json',result)

if __name__=='__main__':main()
