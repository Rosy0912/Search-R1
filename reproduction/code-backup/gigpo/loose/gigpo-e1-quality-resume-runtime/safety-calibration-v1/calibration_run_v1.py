"""Owned same-node calibration executor. It never submits a job or starts formal300."""
import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from owned_runtime_resume_v1 import ROOT, require, sha, cgroup_scope, OwnedProcesses, gpu_sample, available
from calibration_contract_v1 import SCOPE, CHECKPOINT, MANIFEST, DATA, validate_binding, validate_guard_binding


def main():
    require(len(sys.argv) == 3, 'Need reviewed binding and exact SHA')
    binding_path = Path(sys.argv[1])
    require(sha(binding_path) == sys.argv[2], 'Calibration binding changed')
    binding = json.loads(binding_path.read_text())
    repo = Path(os.environ['GIGPO_RESUME_REPO'])
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    require(commit == binding['code_commit'] and not subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip(),
            'Dirty or wrong frozen source')
    validate_binding(binding, commit)
    require(repo.resolve() == Path(binding['repo']).resolve(), 'Different source repo path')
    for item in binding['files'] + binding['evidence']:
        require(Path(item['path']).stat().st_size == item['bytes'] and sha(item['path']) == item['sha256'],
                'Bound source/helper/physical/evidence bytes changed')
    native = binding['source_native_acceptance']
    require(sha(native['review_path']) == native['review_sha256'], 'Native review changed')
    review = json.loads(Path(native['review_path']).read_text())
    require(review['status'] == 'PASS' and review['code_commit'] == commit
            and review['restore_manifest_sha256'] == MANIFEST
            and review['result_sha256'] == native['result_sha256'], 'Not exact source native acceptance')
    require(sha(native['result_path']) == native['result_sha256'], 'Native actual result changed')
    native_result = json.loads(Path(native['result_path']).read_text())
    require(native_result['status'] == 'PASS' and native_result['code_commit'] == commit
            and native_result['validation_scope'] == 'GIGPO_POLICY_SAFETY_NATIVE_ONLY'
            and native_result['fixtures_passed'] and native_result['actual_outer_updates'] == 0
            and native_result['cleanup_residual_states'] == {} and not native_result['final_gpu']['processes'],
            'New eight-GPU native result missing or terminal resources not released')
    guard_identity = validate_guard_binding(binding, native_result['guard_identity'])
    for item in guard_identity.values():
        path = Path(item['path'])
        require(path.resolve() == path and path.is_file() and sha(path) == item['sha256'],
                'Actual guard/module path or bytes differ from full native acceptance')
    control = binding['quality_selection_control']
    os.environ.update(GIGPO_POLICY_NATIVE_BINDING=str(binding_path), GIGPO_POLICY_NATIVE_BINDING_SHA256=sys.argv[2],
        SEARCH_R1_BASELINE_COMMIT=commit, GIGPO_QUALITY_SELECTION=control['selection_path'],
        GIGPO_QUALITY_SELECTION_SHA256=control['selection_sha256'], GIGPO_QUALITY_REVIEW=control['selection_review_path'],
        GIGPO_QUALITY_REVIEW_SHA256=control['selection_review_sha256'], GIGPO_QUALITY_CHECKPOINT=CHECKPOINT,
        GIGPO_QUALITY_MANIFEST_SHA256=MANIFEST, GIGPO_QUALITY_GATE='30')
    job = os.environ['SLURM_JOB_ID']
    scheduler = subprocess.check_output(['scontrol', 'show', 'job', job, '-o'], text=True)
    require('Requeue=0' in scheduler.split() and 'Restarts=0' in scheduler.split(), 'Requeue/restart forbidden')
    require(socket.gethostname().split('.')[0] == 'h100-8' and os.environ['SLURM_JOB_NODELIST'] == 'h100-8', 'Wrong node')
    require(len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) == 8 and os.environ.get('SLURM_JOB_GPUS'), 'Not full8 GPU allocation')
    require(int(os.environ['SLURM_CPUS_PER_TASK']) >= 128 and int(os.environ['SLURM_MEM_PER_NODE']) >= 256 * 1024, 'Allocation too small')
    lock = (ROOT / 'gpu-job.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    from search_r1.baselines.preflight import validate_inherited_gpu_leases, gpu_snapshot, check_selected_gpus
    validate_inherited_gpu_leases(8)
    scope, relative = cgroup_scope(job)
    owned = OwnedProcesses(scope, relative)
    for port in (18081, 16381):
        available(port)
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        os.environ.pop(key, None)
    os.environ.update(NO_PROXY='localhost,127.0.0.1,::1', no_proxy='localhost,127.0.0.1,::1')
    headers = '/mnt/data/shaoshitong/search-r1-rl/gigpo/tools/python310-headers-v1/extracted/usr/include'
    header_cpath = ':'.join((headers, headers + '/python3.10', headers + '/x86_64-linux-gnu/python3.10'))
    os.environ.update(CPATH=header_cpath, SEARCH_R1_PYTHON_HEADER_CPATH=header_cpath)
    run_id = f'gigpo-e1-policy-safety-calibration25-job{job}-{time.time_ns()}'
    run = ROOT / 'runs' / run_id
    run.mkdir()
    for name in ('audit', 'tmp', 'ray', 'cache'):
        (run / name).mkdir()
    alias = Path('/tmp') / f'gigpo-c{job}'
    require(not alias.exists() and not alias.is_symlink(), 'Existing owned Ray alias')
    alias.symlink_to(run, target_is_directory=True)
    os.environ.update(GIGPO_E1_ROOT=str(ROOT), GIGPO_E1_RUN=str(run), AGENT_ROOT=str(ROOT),
        SEARCH_R1_ASSETS='/mnt/data/shaoshitong/search-r1-rl/gigpo/assets-root', SEARCH_R1_BASELINE_OUTPUT=str(run),
        SEARCH_R1_BASELINE_RUN_ROOT=str(alias), SEARCH_R1_BASELINE_RUN_ID=run_id,
        SEARCH_R1_BASELINE_AUDIT_DIR=str(run / 'audit'), SEARCH_R1_BASELINE_DATA_IDENTITY=DATA,
        RAY_ADDRESS='127.0.0.1:16381', TMPDIR=str(alias / 'tmp'), RAY_TMPDIR=str(alias / 'ray'),
        XDG_CACHE_HOME=str(alias / 'cache'), HF_HOME=str(alias / 'cache/hf'),
        HF_DATASETS_CACHE='/mnt/data/shaoshitong/search-r1-rl/gigpo/cache/retriever-datasets',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', HF_HUB_OFFLINE='1',
        HF_DATASETS_OFFLINE='1', WANDB_MODE='offline', VLLM_NO_USAGE_STATS='1', NCCL_NVLS_ENABLE='0', VLLM_ATTENTION_BACKEND='XFORMERS')
    result = dict(status='FAILED', scope=SCOPE, job_id=job, run_id=run_id, source_commit=commit,
                  binding_sha256=sha(binding_path), retained_training_updates_contributed=0,
                  restore_checkpoint=CHECKPOINT, restore_manifest_sha256=MANIFEST,
                  calibration_phase=binding['calibration_phase'], cgroup=str(scope), production_thresholds_not_selected=True,
                  guard_identity=guard_identity)
    started = time.monotonic()
    try:
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        from search_r1.baselines.holdout import prepare_common_holdout
        from search_r1.baselines.controller import validate_config
        from search_r1.baselines.policy_safety import validate_config as validate_safety
        with initialize_config_dir(config_dir=str(repo / 'verl/trainer/config'), version_base=None):
            config = compose(config_name='ppo_gigpo_instruct300_quality10_safe')
        config.actor_rollout_ref.actor.policy_safety.mode = 'calibration'
        config.trainer.experiment_name = 'gigpo-e1-policy-safety-calibration25'
        OmegaConf.resolve(config)
        split = prepare_common_holdout(config)
        validate_config(config)
        validate_safety(OmegaConf.to_container(config.actor_rollout_ref.actor.policy_safety, resolve=True))
        require(split['content_identity']['sha256'] == DATA and split['train_rows'] == 169028
                and split['holdout_rows'] == 512, 'Actual common immutable inputs differ')
        OmegaConf.save(config, run / 'resolved-config.yaml')
        os.environ['SEARCH_R1_BASELINE_CONFIG_SHA256'] = sha(run / 'resolved-config.yaml')
        inventory, processes = gpu_snapshot()
        selected = check_selected_gpus(os.environ['CUDA_VISIBLE_DEVICES'].split(','), inventory, processes)
        (ROOT / 'tools/expected-gpu-uuids.json').write_text(json.dumps([g['uuid'] for g in selected]))
        result['initial_gpu'] = gpu_sample()
        # Recheck at spawn, not just when reading the binding. Never silently run
        # the old ROOT guard while listing v2 only as documentary evidence.
        for item in guard_identity.values():
            require(sha(item['path']) == item['sha256'], 'Bound guard/module changed before spawn')
        guard = owned.spawn([sys.executable, guard_identity['resource_guard']['path'],
                             '--output', str(run / 'guard-resources.jsonl')], run / 'guard.log')
        deadline = time.monotonic() + 30
        while not (run / 'guard-resources.ready').exists():
            require(guard.poll() is None and time.monotonic() < deadline, 'Owned dual15% guard failed')
            time.sleep(1)
        retriever = owned.spawn([sys.executable, 'reproduction/h20/retriever_local.py',
            '--index_path', '/mnt/data/shaoshitong/search-r1-rl/gigpo/assets-root/assets/wiki/e5_Flat.index',
            '--corpus_path', '/mnt/data/shaoshitong/search-r1-rl/gigpo/assets-root/assets/wiki/wiki-18.jsonl',
            '--retriever_model', '/mnt/data/shaoshitong/search-r1-rl/gigpo/assets-root/assets/e5-base-v2',
            '--retriever_name', 'e5', '--topk', '3', '--faiss_gpu', '--device', 'cuda', '--threads', '8',
            '--host', '127.0.0.1', '--port', '18081'], run / 'retriever.log',
            env=dict(os.environ, PYTHONPATH='/mnt/data/shaoshitong/search-r1-rl/gigpo/retriever-gpu-lib' + os.pathsep + str(repo),
                     OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8'))
        import requests
        session = requests.Session()
        session.trust_env = False
        deadline = time.monotonic() + 900
        while True:
            require(retriever.poll() is None and guard.poll() is None and time.monotonic() < deadline, 'Owned GPU retriever failed')
            try:
                if session.get('http://127.0.0.1:18081/openapi.json', timeout=2).ok:
                    break
            except requests.RequestException:
                pass
            time.sleep(2)
        subprocess.run([sys.executable, str(ROOT / 'tools/check_http_retrieval.py')], check=True,
                       env=dict(os.environ, SEARCH_R1_HTTP_RECEIPT=str(run / 'local-http-retrieval-verification.json')))
        inventory, processes = gpu_snapshot()
        check_selected_gpus(os.environ['CUDA_VISIBLE_DEVICES'].split(','), inventory, processes)
        helpers = Path(__file__).resolve().parent
        head = owned.spawn([str(ROOT / 'env/bin/ray'), 'start', '--head', '--node-ip-address=127.0.0.1',
            '--port=16381', '--num-cpus=24', '--num-gpus=8', '--include-dashboard=false', '--object-store-memory=8589934592',
            '--temp-dir=' + str(alias / 'ray'), '--min-worker-port=21081', '--max-worker-port=22080',
            '--disable-usage-stats', '--block'], run / 'ray.log')
        deadline = time.monotonic() + 120
        while True:
            require(head.poll() is None and guard.poll() is None and time.monotonic() < deadline, 'Owned Ray failed')
            try:
                with socket.create_connection(('127.0.0.1', 16381), timeout=1):
                    break
            except OSError:
                time.sleep(2)
        subprocess.run([sys.executable, str(helpers / 'ray_audit_resume_v1.py'), str(run / 'ray-audit-environment-verification.json')], check=True)
        driver = owned.spawn([sys.executable, str(helpers / 'calibration_driver_v1.py'), str(run / 'resolved-config.yaml'), str(binding_path)],
                            run / 'calibration.log', pass_fds=tuple(int(v['fd']) for v in json.loads(os.environ['SEARCH_R1_GPU_LEASES'])))
        while driver.poll() is None:
            require(all(p.poll() is None for p in (guard, retriever, head)), 'Owned calibration dependency died')
            time.sleep(5)
        require(driver.returncode == 0, 'Actual calibration/probability identity failed; preserve evidence')
        observed = json.loads((run / 'driver-result.json').read_text())
        require(observed['retained_training_updates_contributed'] == 0
                and observed['scope'] == SCOPE and observed['production_thresholds_not_selected'], 'Calibration fabricated formal progress or thresholds')
        result.update(status=observed['status'], driver_exit=driver.returncode, observations=observed)
    except BaseException as error:
        result.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            owned.cleanup()
        except BaseException as error:
            result.update(status='FAILED', cleanup_error=repr(error))
            raise
        finally:
            alias.unlink(missing_ok=True)
            result.update(cleanup_residual_states=getattr(owned, 'cleanup_residual', {}), wall_seconds=time.monotonic() - started)
            try:
                result['final_gpu'] = gpu_sample()
                require(not result['final_gpu']['processes'], 'GPU compute remains after owned cleanup')
            except BaseException as error:
                result.update(status='FAILED', terminal_gpu_error=repr(error))
            (run / 'run-result.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    require(result['status'] != 'FAILED', 'Calibration/cleanup failed')
    if result['observations']['calibration_actual_outer_updates'] == 1:
        # Fresh semantic HF loading occurs after all owned GPU workers exit.
        cpu_owned = OwnedProcesses(scope, relative)
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        try:
            verifier = cpu_owned.spawn([sys.executable, str(ROOT / 'tools/verify-gigpo-checkpoint.py'),
                '--base', CHECKPOINT, '--checkpoint', str(run / 'checkpoints/actor/global_step_26'),
                '--output', str(run / 'checkpoint26-cpu-verification.json')], run / 'checkpoint26-cpu-verification.log',
                env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8',
                         TMPDIR=str(run / 'tmp'), HF_HOME=str(run / 'cache/hf'), XDG_CACHE_HOME=str(run / 'cache')))
            while verifier.poll() is None:
                time.sleep(1)
            require(verifier.returncode == 0, 'Isolated updated HF semantic CPU reload failed')
            result['semantic_hf_cpu_reload'] = dict(exit_code=0, path=str(run / 'checkpoint26-cpu-verification.json'),
                sha256=sha(run / 'checkpoint26-cpu-verification.json'))
        except BaseException as error:
            result.update(status='FAILED', semantic_hf_reload_error=repr(error))
            raise
        finally:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            try:
                cpu_owned.cleanup()
                result['semantic_hf_cleanup_residual_states'] = getattr(cpu_owned, 'cleanup_residual', {})
                result['final_gpu'] = gpu_sample()
                require(not result['final_gpu']['processes'], 'Unexpected GPU compute during CPU semantic reload')
            except BaseException as error:
                result.update(status='FAILED', semantic_hf_cleanup_error=repr(error))
                raise
            finally:
                result['wall_seconds'] = time.monotonic() - started
                (run / 'run-result.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, allow_nan=False), flush=True)


def interrupted(sig, frame):
    raise RuntimeError('Owned calibration executor interrupted by signal ' + str(sig))


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    main()
