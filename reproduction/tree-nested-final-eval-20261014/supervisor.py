"""Own final-only resident evaluator; training termination is externally authorized."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from contract import (ROOT,RUN,TRAIN_SOURCE,CONFIG,CONFIG_SHA,HF,OUTPUT,FROZEN_SOURCE,GPU_UUIDS,
                      require,sha,read_json,verify_handoff,verify_code_manifest)
from eval_merge import atomic_immutable


def own_scope(unit):
    require(unit.startswith('sst-tree-nested-final-eval-') and unit.endswith('.service') and '/' not in unit,'owned eval unit required')
    entries=Path('/proc/self/cgroup').read_text().splitlines()
    require(not any('job_' in line.split(':',2)[-1] for line in entries),'evaluation must not impersonate Slurm')
    for line in entries:
        _,controllers,relative=line.split(':',2)
        if controllers=='' and unit in Path(relative).parts:
            scope=Path('/sys/fs/cgroup')/relative.lstrip('/')
            require((scope/'cgroup.procs').is_file(),'actual own cgroup required')
            return scope,relative
    raise ValueError('Own systemd cgroup v2 identity unavailable')


def main():
    p=argparse.ArgumentParser()
    for name in ('unit','handoff','handoff-sha','code-manifest','code-manifest-sha'):p.add_argument('--'+name,required=True)
    a=p.parse_args();tools=Path(__file__).resolve().parent
    code=verify_code_manifest(tools,a.code_manifest,a.code_manifest_sha)
    handoff=verify_handoff(a.handoff,a.handoff_sha)
    assets=read_json(handoff['asset_manifest']['path'],handoff['asset_manifest']['sha256'])
    scope,relative=own_scope(a.unit)
    require(scope.joinpath('memory.max').read_text().strip()==str(384*1024**3),'own CPU MemoryMax must remain384GiB')
    available_kb=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))
    require(available_kb*1024>=448*1024**3,'need384GiB own ceiling plus64GiB host headroom before own load')
    selected=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    require(len(selected)==8 and set(selected)==set(GPU_UUIDS),'all eight pinned same-card UUIDs required')
    locks=[];owned=None;run=None;result=None;nv=None
    try:
        lease_root=Path('/tmp')/f'search-r1-gpu-leases-{os.getuid()}';lease_root.mkdir(mode=0o700,exist_ok=True)
        records=[]
        for uid in sorted(selected):
            fd=os.open(lease_root/(uid+'.lock'),os.O_CREAT|os.O_RDWR,0o600)
            locks.append(fd);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);os.set_inheritable(fd,True)
            records.append(dict(uuid=uid,fd=fd))
        import pynvml as nv
        nv.nvmlInit()
        for uid in selected:
            info=nv.nvmlDeviceGetMemoryInfo(nv.nvmlDeviceGetHandleByUUID(uid.encode()))
            require(info.free>=40*1024**3,'same-card startup needs40GiB free including retained retrieval and resident engine')
        require(not (OUTPUT/'scores/final-metrics.json').exists(),'this final evaluation already has complete metrics; no duplicate')
        run=ROOT/'runs'/a.unit.removesuffix('.service');run.mkdir()
        for name in ('output','cache','tmp'):(run/name).mkdir()
        sys.path.insert(0,str(ROOT/'tools'))
        from tree_shared_owned_runtime_fixed import OwnedProcesses,available,gpu_sample
        owned=OwnedProcesses(scope,relative)
        available(18082)
        env=dict(os.environ)
        for key in list(env):
            if key.startswith(('SLURM_','SBATCH_','SQUEUE_','SCANCEL_','SACCT_','SCONTROL_',
                               'TREE_RECOVERY_','TREE_CURRENT_CHAIN_RESUME_','TREE_NESTED_CHAIN_RESUME_','SEARCH_R1_BASELINE_')) or key in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):
                env.pop(key,None)
        env.update(PYTHONPATH=str(tools)+':'+str(FROZEN_SOURCE)+':'+str(ROOT/'tools'),PYTHONOPTIMIZE='0',
            PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
            TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',VLLM_NO_USAGE_STATS='1',
            VLLM_ATTENTION_BACKEND='XFORMERS',NCCL_NVLS_ENABLE='0',NO_PROXY='127.0.0.1,localhost,::1',no_proxy='127.0.0.1,localhost,::1',
            TMPDIR=str(run/'tmp'),XDG_CACHE_HOME=str(run/'cache'),HF_HOME=str(run/'cache/hf'),
            SEARCH_R1_BASELINE_COMMIT=TRAIN_SOURCE,SEARCH_R1_BASELINE_CONFIG_SHA256=CONFIG_SHA,
            SEARCH_R1_BASELINE_DATA_IDENTITY='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24',
            SEARCH_R1_MODEL_IDENTITY=assets['model_sha256'],SEARCH_R1_BASELINE_RUN_ID=run.name,
            SEARCH_R1_BASELINE_OUTPUT=str(OUTPUT/'scores'),SEARCH_R1_GPU_LEASES=json.dumps(records),SEARCH_R1_GPU_PREFLIGHT='1',
            TREE_EVAL_SHARED_MEMORY_MODE='1',TREE_EVAL_RESOURCE_PROTOCOL='memory_headroom_shared_v1',
            TREE_EVAL_SYSTEMD_UNIT=a.unit,SLURM_JOB_ID='not-slurm;systemd:'+a.unit)
        result=dict(status='FAILED',unit=a.unit,evaluation_only=True,training_source_commit=TRAIN_SOURCE,
            evaluation_wrapper_source_commit=code['eval_wrapper_source_commit'],training_run=str(RUN),actual_completed_training_outer=200,
            handoff_sha256=a.handoff_sha,code_manifest_sha256=a.code_manifest_sha,score_output=str(OUTPUT/'scores'),
            archive_proof=handoff['archive'],effective_training_segments=read_json(handoff['archive']['path'],handoff['archive']['sha256'])['effective_training_segments'],
            original_training_normal_exit_claimed=False,training_signals_sent=False,optimizer_constructed=False,
            physical_gpus=8,slurm_reserved_gpus=0,start_epoch=time.time(),worker_exit_code=None)
        atomic_immutable(run/'allocation-identity.json',dict(unit=a.unit,cgroup=relative,
            physical_gpus=8,slurm_reserved_gpus=0,leased_gpu_uuids=selected,initial_gpu=gpu_sample(),
            cpu_memory_max_gib=384,required_host_mem_available_gib=448,handoff_sha256=a.handoff_sha,
            original_training_normal_exit_claimed=False))
        def interrupted(sig,frame):raise RuntimeError('Own final evaluator interrupted '+str(sig))
        for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,interrupted)
        try:
            guard=owned.spawn([sys.executable,str(tools/'shared_guard.py'),'--output',str(run/'resource-samples.jsonl'),'--unit',a.unit],run/'guard.log',env)
            deadline=time.monotonic()+30
            while not (run/'resource-samples.ready').exists():
                require(guard.poll() is None and time.monotonic()<deadline,'own shared guard startup failed');time.sleep(.2)
            retriever=owned.spawn([sys.executable,str(ROOT/'repo/reproduction/h20/retriever_local.py'),
                '--index_path',str(ROOT/'assets-root/assets/wiki/e5_Flat.index'),'--corpus_path',str(ROOT/'assets-root/assets/wiki/wiki-18.jsonl'),
                '--retriever_model',str(ROOT/'assets-root/assets/e5-base-v2'),'--retriever_name','e5','--topk','3',
                '--faiss_gpu','--device','cuda','--threads','8','--host','127.0.0.1','--port','18082'],run/'retriever.log',
                dict(env,PYTHONPATH=str(ROOT/'retriever-gpu-lib')+':'+str(FROZEN_SOURCE),OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8'),pass_fds=tuple(locks))
            import requests
            session=requests.Session();session.trust_env=False;deadline=time.monotonic()+900
            while True:
                require(guard.poll() is None and retriever.poll() is None and time.monotonic()<deadline,'own retriever startup failed')
                try:
                    if session.get('http://127.0.0.1:18082/openapi.json',timeout=2).ok:break
                except requests.RequestException:pass
                time.sleep(1)
            common=['--config',str(CONFIG),'--prefix',handoff['prefix']['path'],'--manifest',handoff['asset_manifest']['path'],
                    '--model',str(HF),'--output',str(OUTPUT/'scores')]
            numeric=[str(nv.nvmlDeviceGetIndex(nv.nvmlDeviceGetHandleByUUID(uid.encode()))) for uid in selected]
            require(len(set(numeric))==8,'eight leased UUID/numeric device mapping required')
            worker_env=dict(env,CUDA_VISIBLE_DEVICES=','.join(numeric),TREE_EVAL_LEASED_GPU_UUIDS=json.dumps(selected))
            atomic_immutable(run/'worker-gpu-mapping.json',dict(leased_uuids=selected,visible_numeric_indices=numeric))
            worker=owned.spawn([sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=8',
                str(tools/'eval_worker.py'),*common,'--local-batch-size','64'],run/'eval.log',worker_env,pass_fds=tuple(locks))
            while worker.poll() is None:
                require(guard.poll() is None and retriever.poll() is None,'own service stopped')
                with (run/'resource-samples.jsonl').open('rb') as stream:
                    stream.seek(0,2);stream.seek(max(0,stream.tell()-32768));last=json.loads(stream.read().decode().splitlines()[-1])
                require(time.time()-last['epoch']<10 and not last.get('event'),'guard stale or memory insufficient');time.sleep(1)
            result['worker_exit_code']=worker.returncode;require(worker.returncode==0,'workers failed; preserve committed scoring batches')
            subprocess.run([sys.executable,str(tools/'eval_merge.py'),*common],env=env,check=True)
            metrics=json.loads((OUTPUT/'scores/final-metrics.json').read_text())
            require(metrics['evaluated_count']==51713,'full final cardinality required')
            result.update(status='COMPLETE',evaluated_count=51713,final_metrics_sha256=sha(OUTPUT/'scores/final-metrics.json'),
                          final_metrics_done_sha256=sha(OUTPUT/'scores/final-metrics.done.json'),independent_final_acceptance=False)
        except BaseException as error:
            result['error']=repr(error);raise
        finally:
            for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,signal.SIG_IGN)
            try:
                owned.cleanup();result['own_cleanup_residual']=owned.cleanup_residual;result['final_gpu']=gpu_sample()
                require(not any(state!='Z' for state in owned.cleanup_residual.values()),'own live cleanup residual remains')
            except BaseException as error:
                result.update(status='FAILED',cleanup_error=repr(error));raise
            finally:
                result.update(end_epoch=time.time(),wall_seconds=time.time()-result['start_epoch'])
                atomic_immutable(run/'run-result.json',result)
    finally:
        for fd in locks:
            try:os.close(fd)
            except OSError:pass
        if nv is not None:
            try:nv.nvmlShutdown()
            except Exception:pass

if __name__=='__main__':main()
