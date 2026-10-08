"""Own single-node Slurm supervisor, leased GPU retrieval + standalone32B TP8."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import secrets
import socket
import subprocess
import sys
import threading
import time
from eval64 import asset_gate, require, sha, write_json

ROOT = Path('/dataset/shaoshitong/search-r1-rl/topd')


def local_env(source):
    env = dict(source)
    for key in list(env):
        if key.lower() in ('http_proxy', 'https_proxy', 'all_proxy') or key.startswith(('SEARCH_R1_TOPD_', 'SEARCH_R1_BASELINE_')):
            env.pop(key)
    env.pop('RAY_ADDRESS', None)
    env.update(NO_PROXY='127.0.0.1,localhost,::1', no_proxy='127.0.0.1,localhost,::1',
               HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               PYTHONPATH=str(ROOT / 'repo-terminal-fixed413'), TOKENIZERS_PARALLELISM='false')
    return env


def slurm_gate():
    require(os.environ.get('AGENT_ROOT') == str(ROOT), 'Require own root')
    require(socket.gethostname().split('.')[0] == 'h100-11'
            and os.environ.get('SLURM_JOB_NODELIST') == 'h100-11', 'Require own local node')
    require(os.environ.get('SLURM_JOB_ID'), 'Only real allocated Slurm job')
    require(int(os.environ.get('SLURM_GPUS_ON_NODE', '0')) == 8
            and len(os.environ.get('SLURM_JOB_GPUS', '').split(',')) == 8, 'Require8 Slurm GPUs')
    require(int(os.environ.get('SLURM_CPUS_PER_TASK', '0')) >= 128
            and int(os.environ.get('SLURM_MEM_PER_NODE', '0')) >= 384 * 1024,
            'Require128CPU/384GiB budget')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binding', required=True)
    args = parser.parse_args()
    slurm_gate()
    binding, _, _ = asset_gate(args.binding, ROOT)
    from search_r1.baselines.preflight import validate_inherited_gpu_leases
    import runtime_safety as safety
    from runtime_safety import nvml_snapshot, pid_identity
    from startup_guard import StartupGuard
    guard = StartupGuard(safety, expected_path=Path(__file__).parent/'runtime_safety.py',
                         trace_path=ROOT/'logs'/('trained64-window-'+os.environ['SLURM_JOB_ID']+'.jsonl'))
    active_gate = guard.active_gate
    validate_inherited_gpu_leases(8)  # before own retrieval/CUDA allocation
    leases = json.loads(os.environ['SEARCH_R1_GPU_LEASES'])
    fds = tuple(int(item['fd']) for item in leases)
    uuids = {item['uuid'] for item in leases}
    selected = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    spec = importlib.util.spec_from_file_location('owned_eval_launcher', ROOT / 'tools/topd-short-run.py')
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    scope, relative = launcher.cgroup_scope(os.environ['SLURM_JOB_ID'])
    owned = launcher.OwnedProcesses(scope, relative)
    out = ROOT / 'logs' / ('qwen25-32b-terminal-fixed413-eval64-' + os.environ['SLURM_JOB_ID'])
    out.mkdir(exist_ok=False)
    guard.ready_check = lambda: (out/'evaluation/readiness.json').is_file()
    env = local_env(os.environ)
    env.update(SEARCH_R1_EVAL64_SUPERVISOR_PID=str(os.getpid()), SEARCH_R1_EVAL64_CGROUP=relative,
               SEARCH_R1_EVAL64_TERMINAL_NONCE=secrets.token_hex(32))
    stop, monitor_thread = threading.Event(), None
    result = dict(scope='QWEN25_32B_G1_TRAINED_EVAL64_NO_TRAINING', status='RUNNING',
                  binding_sha256=sha(args.binding), cgroup=relative, leases=[dict(uuid=i['uuid']) for i in leases],
                  protocol='same64 sample42 greedy four-turn Search-R1; no actor/ref/optimizer/FSDP')
    sampler_errors = []
    baseline_apps = set()
    runtime = None

    def snapshot():
        sample = launcher.gpu_sample()
        sample.update(epoch=time.time(), own_pids=sorted(launcher.cgroup_pids(scope) - owned.baseline),
                      own_processes=[launcher.process_sample(p) for p in sorted(launcher.cgroup_pids(scope) - owned.baseline)])
        with (out / 'gpu-memory-timeseries.jsonl').open('a') as f:
            f.write(json.dumps(sample) + '\n')
        return sample

    def monitor():
        while not stop.wait(1):
            try:
                snapshot()
                if runtime is not None:
                    actual = active_gate(runtime)
                    with (out / 'active-occupancy.jsonl').open('a') as f:
                        f.write(json.dumps(dict(epoch=time.time(), **actual)) + '\n')
            except Exception as exc:
                sampler_errors.append(repr(exc))

    def interrupted(signum, frame):
        raise RuntimeError('Own evaluation interrupted: ' + str(signum))

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        baseline = snapshot()
        baseline_apps = set(baseline['processes'])
        write_json(out / 'baseline.json', baseline)
        nvml_baseline = nvml_snapshot()
        write_json(out / 'baseline-nvml.json', nvml_baseline)
        # No connection to a previously running service; bind ownership before spawn.
        launcher.available(18085)
        assets = [ROOT / p for p in ('assets-root/assets/wiki/e5_Flat.index',
                                     'assets-root/assets/wiki/wiki-18.jsonl', 'assets-root/assets/e5-base-v2')]
        for path in assets:
            require(path.exists() and ROOT in path.resolve().parents, 'Nonlocal retrieval asset')
        renv = dict(env, PYTHONPATH=str(ROOT / 'retriever-gpu-lib') + ':' + str(ROOT / 'repo-terminal-fixed413'),
                    OMP_NUM_THREADS='8', MKL_NUM_THREADS='8')
        command = [sys.executable, '-c', 'import torch,runpy,sys;server=sys.argv.pop(1);sys.argv[0]=server;runpy.run_path(server,run_name="__main__")',
                   str(ROOT / 'repo-terminal-fixed413/search_r1/search/retrieval_server.py'), '--index_path', str(assets[0]),
                   '--corpus_path', str(assets[1]), '--retriever_model', str(assets[2]), '--retriever_name', 'e5',
                   '--topk', '3', '--device', 'cuda', '--faiss_gpu', '--host', '127.0.0.1', '--port', '18085', '--threads', '8']
        service = owned.spawn(command, out / 'retriever.log', renv, pass_fds=fds)
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        import requests
        session = requests.Session()
        session.trust_env = False
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            require(service.poll() is None, 'Own GPU retrieval failed')
            try:
                if session.get('http://127.0.0.1:18085/openapi.json', timeout=2).ok:
                    break
            except requests.RequestException:
                pass
            time.sleep(2)
        else:
            raise TimeoutError('Retrieval startup')
        response = session.post('http://127.0.0.1:18085/retrieve', json=dict(queries=['What is the capital of France?'], topk=3, return_scores=True), timeout=(5, 900))
        response.raise_for_status()
        payload = response.json()
        require(len(payload['result']) == 1 and len(payload['result'][0]) == 3, 'Retrieval readiness top3 failure')
        write_json(out / 'retriever-readiness.json', dict(pid=service.pid, assets=[str(p.resolve()) for p in assets],
                   url='http://127.0.0.1:18085/retrieve', response=payload, scope='EXCLUDED_FROM64'))
        # Actual2983 retriever is5.8--7.5GiB/card (30GiB index is ALL8 total).
        # Preserve absolute dual15% before model plus independent own accounting.
        runtime = dict(supervisor_pid=os.getpid(), cgroup=relative,
                       retriever_identity=pid_identity(service.pid), baseline=nvml_baseline)
        write_json(out / 'active-runtime.json', runtime)
        env.update(SEARCH_R1_EVAL64_RUNTIME=str(out / 'active-runtime.json'),
                   SEARCH_R1_EVAL64_RUNTIME_SHA256=sha(out / 'active-runtime.json'))
        deadline = time.monotonic() + 180
        while True:
            require(service.poll() is None, 'Own retrieval died before TP8 startup')
            try:
                validate_inherited_gpu_leases(8)
                record = active_gate(runtime, pre_model=True)
                time.sleep(.2)
                validate_inherited_gpu_leases(8)
                record = active_gate(runtime, pre_model=True)
                break
            except RuntimeError:
                require(time.monotonic() < deadline, 'Active owned-retrieval/foreign15%/free-memory gate failed')
                time.sleep(2)
        write_json(out / 'pre-model-active-gate.json', record)
        result['leased_gpu_mapping'] = record['gpu_rows']
        child = owned.spawn([sys.executable, str(Path(__file__).parent / 'eval64.py'),
                             '--binding', str(Path(args.binding).resolve()), '--output', str(out / 'evaluation')],
                            out / 'evaluation.log', env, pass_fds=fds)
        result['roles'] = dict(retriever_pid=service.pid, evaluation_pid=child.pid,
                              tp_scope='one8GPU multiprocessing engine; actual child PIDs recorded by cgroup/NVML sampler')
        while child.poll() is None:
            require(service.poll() is None, 'Own retriever exited during evaluation')
            require(not sampler_errors, 'Active GPU accounting/sampling failed')
            if (out / 'evaluation/terminal-held.json').is_file():
                from terminal import authorize
                result['terminal'] = authorize(out / 'evaluation', child.pid, os.getpid(), os.environ['SLURM_JOB_ID'],
                    env['SEARCH_R1_EVAL64_TERMINAL_NONCE'], lambda: launcher.stop_monitor(stop, monitor_thread), sampler_errors)
                break
            time.sleep(1)
        require('terminal' in result, 'Evaluation exited without held-owner terminal handshake')
        require(child.wait(timeout=60) == 0, 'Authorized evaluation exit failed')
        from terminal import verify_ack
        result['terminal_child_ack'] = verify_ack(out / 'evaluation', result['terminal'])
        require(child.returncode == 0, 'Own32B evaluation failed; preserve partial raw evidence')
        evaluated = json.loads((out / 'evaluation/result.json').read_text())
        require(evaluated['status'] == 'COMPLETE64_UNREVIEWED_NO_TRAINING', 'Incomplete64')
        result.update(status='COMPLETE64_UNREVIEWED_NO_TRAINING', quality=evaluated['quality'],
                      evaluation_result_sha256=sha(out / 'evaluation/result.json'))
    except BaseException as exc:
        result.update(status='FAIL_NO_TRAINING', error=repr(exc))
        raise
    finally:
        try:
            result['monitor_stop'] = launcher.stop_monitor(stop, monitor_thread)
            owned.cleanup()
            result['cleanup'] = owned.cleanup_record
            final = snapshot()
            write_json(out / 'final-gpu.json', final)
            # Baseline preservation binds PID+UUID; memory may legitimately fluctuate.
            def identities(lines):
                return {(p.split(',')[0].strip(), int(p.split(',')[1].strip())) for p in lines}
            missing = identities(baseline_apps) - identities(final['processes'])
            result['foreign_baseline_missing'] = sorted(missing)
            require(not missing, 'Foreign baseline process disappeared')
            require(not sampler_errors, 'GPU sampling failure')
        except BaseException as exc:
            result.update(status='FAIL_NO_TRAINING', cleanup_error=repr(exc),
                          cleanup=getattr(owned, 'cleanup_record', None))
            raise
        finally:
            result['sampler_errors'] = sampler_errors
            result['parent_cuda_initialized'] = 'torch' in sys.modules and sys.modules['torch'].cuda.is_initialized()
            write_json(out / 'supervisor-result.json', result)


if __name__ == '__main__':
    main()
