"""Own eight-GPU evaluation allocation; no optimizer or training restart."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

ROOT = Path('/dataset/shaoshitong/search-r1-rl/tree-grpo')
OLD = ROOT / 'runs/tree-branpo-fresh200-3360'
FROZEN = ROOT / 'instruct300/candidates/branpo-fresh200-8b6f99d1b3353145200eb3fa7c16a579a79a987c/source/repo'
HERE = Path(__file__).resolve().parent


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(1024*1024), b''):
            h.update(data)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    fd = os.open(str(Path(path).parent), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prefix', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--release', required=True)
    p.add_argument('--score-output', required=True)
    a = p.parse_args()
    require(socket.gethostname().split('.')[0] == 'h100-10', 'Wrong node')
    job = os.environ['SLURM_JOB_ID']
    require(job.isdigit() and os.environ['SLURM_JOB_NODELIST'] == 'h100-10', 'Wrong allocation')
    require(len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) == 8, 'Eight GPUs required')
    require(os.environ.get('SLURM_RESTART_COUNT', '0') == '0', 'No scheduler requeue')
    require(not sys.flags.optimize, 'Unoptimized Python required')
    release = json.loads(Path(a.release).read_text())
    for relative, expected in release['files'].items():
        require(sha(HERE/relative) == expected, 'Evaluation release changed: '+relative)
    require(sha(a.prefix) == release['prefix_sha256'], 'Preserved prefix changed')
    require(sha(a.manifest) == release['manifest_sha256'], 'Evaluation asset manifest changed')
    for name,expected in release['external_files'].items():
        path=Path(name).resolve()
        require(path.is_relative_to(ROOT.resolve()),'External source outside own task')
        require(sha(path)==expected,'External frozen dependency changed: '+name)
    package=FROZEN.parent.parent
    require(sha(package/'manifest.json')=='e2503977c8746edbc83b722d8224548f060909df9fc25d91a9cb88a947a79a21','Original frozen package identity changed')
    original=json.loads((package/'manifest.json').read_text())
    for name,expected in original['files'].items():
        require(sha(package/name)==expected,'Original frozen package source changed: '+name)
    prefix_done=json.loads(Path(a.prefix+'.done.json').read_text())
    require(prefix_done['sha256']==release['prefix_sha256'] and prefix_done['original_job']==3360
            and prefix_done['actual_trained_outer']==200,'Bound preserved-prefix trained200 proof missing')
    require(prefix_done['identity']['model_sha256']==release['model_sha256'],'Trained HF identity differs')
    scores=Path(a.score_output).resolve()
    require(scores.is_relative_to((OLD/'output/eval-resume').resolve()), 'Score output must remain in own durable evaluation directory')
    require(str(scores)==release['score_output'], 'Reviewed persistent score directory differs')
    scores.mkdir(parents=True,exist_ok=True)
    require(release['review_status'] == 'ACCEPTED' and release['training_updates'] == 200,
            'Independent review and actual200 binding required')
    require(sha(OLD/'resolved.yaml') == 'c9415c0096853f78d870e205fc956f20045cde985836de5225f609eaf091d4b3', 'Old config changed')
    # The predecessor must have exited before a second GPU instance starts.
    state = subprocess.check_output(['squeue', '--local', '-h', '-j', '3360', '-o', '%T'], text=True).strip()
    require(not state, 'Predecessor still owns its allocation')
    sys.path.insert(0, str(ROOT/'tools'))
    from tree_shared_owned_runtime_fixed import OwnedProcesses, cgroup_scope, gpu_sample, available
    from search_r1.baselines.preflight import validate_inherited_gpu_leases
    validate_inherited_gpu_leases(8)
    scope, relative = cgroup_scope(job)
    owned = OwnedProcesses(scope, relative)
    run = ROOT/'runs'/('tree-final-eval-resume-'+job)
    run.mkdir()
    for name in ('output', 'cache', 'tmp'):
        (run/name).mkdir()
    fds = tuple(int(v['fd']) for v in json.loads(os.environ['SEARCH_R1_GPU_LEASES']))
    env = dict(os.environ)
    for k in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):
        env.pop(k, None)
    env.update(PYTHONUNBUFFERED='1', PYTHONOPTIMIZE='0', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1',
               HF_DATASETS_OFFLINE='1', VLLM_NO_USAGE_STATS='1', VLLM_ATTENTION_BACKEND='XFORMERS',
               NCCL_NVLS_ENABLE='0', NO_PROXY='127.0.0.1,localhost,::1', no_proxy='127.0.0.1,localhost,::1',
               TMPDIR=str(run/'tmp'), XDG_CACHE_HOME=str(run/'cache'), HF_HOME=str(run/'cache/hf'),
               PYTHONPATH=str(FROZEN)+':'+str(ROOT/'tools'),
               SEARCH_R1_BASELINE_COMMIT='8b6f99d1b3353145200eb3fa7c16a579a79a987c',
               SEARCH_R1_BASELINE_CONFIG_SHA256=sha(OLD/'resolved.yaml'),
               SEARCH_R1_BASELINE_DATA_IDENTITY='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24',
               SEARCH_R1_MODEL_IDENTITY=release['model_sha256'], SEARCH_R1_BASELINE_RUN_ID=run.name,
               SEARCH_R1_BASELINE_OUTPUT=str(scores))
    available(18082)
    write(run/'allocation-identity.json', dict(job=job, node='h100-10', current_cgroup=relative,
          release_sha256=sha(a.release), prefix_sha256=sha(a.prefix), manifest_sha256=sha(a.manifest),
          initial_gpu=gpu_sample(), cuda_visible_devices=env['CUDA_VISIBLE_DEVICES'],
          training_updates=200, evaluation_only=True, local_batch_size=64))
    result = dict(status='FAILED', job_id=job, predecessor_job=3360, training_updates=200,
                  evaluation_only=True, start_epoch=time.time(), prefix_sha256=sha(a.prefix),
                  exact_executor_resume=False, remaining_question_resume=True,score_output=str(scores))
    def interrupted(sig, frame):
        raise RuntimeError('Own evaluation interrupted '+str(sig))
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    try:
        guard = owned.spawn([sys.executable,str(HERE/'guard.py'),'--output',str(run/'resource-samples.jsonl')], run/'guard.log', env)
        deadline=time.monotonic()+30
        while not (run/'resource-samples.ready').exists():
            require(guard.poll() is None and time.monotonic()<deadline, 'Foreign resource guard failed')
            time.sleep(.2)
        retriever = owned.spawn([sys.executable,str(ROOT/'repo/reproduction/h20/retriever_local.py'),
          '--index_path',str(ROOT/'assets-root/assets/wiki/e5_Flat.index'),
          '--corpus_path',str(ROOT/'assets-root/assets/wiki/wiki-18.jsonl'),
          '--retriever_model',str(ROOT/'assets-root/assets/e5-base-v2'), '--retriever_name','e5',
          '--topk','3','--faiss_gpu','--device','cuda','--threads','8','--host','127.0.0.1','--port','18082'],
          run/'retriever.log',dict(env,PYTHONPATH=str(ROOT/'retriever-gpu-lib')+':'+str(FROZEN),
                                     OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8'),pass_fds=fds)
        import requests
        session=requests.Session();session.trust_env=False;deadline=time.monotonic()+900
        while True:
            require(guard.poll() is None and retriever.poll() is None and time.monotonic()<deadline,'Retriever startup failed')
            try:
                if session.get('http://127.0.0.1:18082/openapi.json',timeout=2).ok:
                    break
            except requests.RequestException:
                pass
            time.sleep(2)
        common=['--config',str(OLD/'resolved.yaml'),'--prefix',a.prefix,'--manifest',a.manifest,
                '--model',str(OLD/'checkpoints/actor/global_step_200'),'--output',str(scores)]
        worker = owned.spawn([sys.executable,'-m','torch.distributed.run','--standalone', '--nnodes=1',
                 '--nproc-per-node=8',str(HERE/'eval_worker.py'),*common,'--local-batch-size','64'],
                 run/'eval.log',env,pass_fds=fds)
        while worker.poll() is None:
            require(guard.poll() is None and retriever.poll() is None,'Own service stopped')
            latest=json.loads((run/'resource-samples.jsonl').read_text().splitlines()[-1])
            require(time.time()-latest['epoch']<30 and not latest.get('event'),'Guard stale or stopped')
            time.sleep(5)
        result['worker_exit_code']=worker.returncode
        require(worker.returncode==0,'Evaluation workers failed; durable completed batches preserved')
        subprocess.run([sys.executable,str(HERE/'eval_merge.py'),*common],env=env,check=True)
        require((scores/'final-metrics.json').is_file(),'Full final result missing')
        result['status']='COMPLETE'
    except BaseException as err:
        result['error']=repr(err)
        raise
    finally:
        try:
            owned.cleanup()
            result['own_cleanup_residual']=owned.cleanup_residual
            result['final_gpu']=gpu_sample()
        finally:
            result['end_epoch']=time.time()
            result['wall_seconds']=result['end_epoch']-result['start_epoch']
            write(run/'run-result.json',result)


if __name__=='__main__':
    main()
