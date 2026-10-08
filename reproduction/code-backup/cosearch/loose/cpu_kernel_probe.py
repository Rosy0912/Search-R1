"""Own CPU-only retrieval/ranker capability job; stop only the child pidfd."""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.request

root = Path('/dataset/shaoshitong/search-r1-rl/cosearch')
env = os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8',
           OPENBLAS_NUM_THREADS='8', TMPDIR=str(root/'tmp'), HF_HOME=str(root/'cache/hf'),
           HF_DATASETS_CACHE=str(root/'cache/hf/datasets'), XDG_CACHE_HOME=str(root/'cache'),
           HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', SEARCH_R1_REPO=str(root/'repo'))
python = str(root/'env/bin/python')
with socket.socket() as sock:
    sock.bind(('127.0.0.1', 18086))
import yaml
config = yaml.safe_load((root/'logs/resolved-prepared.yaml').read_text())
command = [python, str(root/'repo/reproduction/h20/retriever_local.py'),
    '--index_path', str(root/'assets-root/assets/wiki/e5_Flat.index'),
    '--corpus_path', str(root/'assets-root/assets/wiki/wiki-18.jsonl'),
    '--retriever_name', 'e5', '--retriever_model', str(root/'assets-root/assets/e5-base-v2'),
    '--topk', '3', '--device', 'cpu', '--host', '127.0.0.1', '--port', '18086', '--threads', '8']
started = time.perf_counter()
log = (root/'logs/cpu-retriever-sdpa.log').open('w')
service = subprocess.Popen(command, env=env, cwd=root/'repo', stdout=log, stderr=subprocess.STDOUT)
fd = os.pidfd_open(service.pid)
(root/'logs/cpu-sdpa-owner.json').write_text(json.dumps(dict(
    job=os.environ.get('SLURM_JOB_ID'), pid=service.pid, command=command,
    allowed_training_gpus=[], cpus=os.environ.get('SLURM_CPUS_PER_TASK'),
    memory_mb=os.environ.get('SLURM_MEM_PER_NODE')), indent=2)+'\n')
try:
    while True:
        if service.poll() is not None:
            raise RuntimeError('Own CPU retriever failed; see cpu-retriever-sdpa.log')
        try:
            with urllib.request.urlopen('http://127.0.0.1:18086/openapi.json', timeout=5) as response:
                if response.status == 200:
                    break
        except OSError:
            pass
        if time.perf_counter() - started > 1200:
            raise TimeoutError('CPU retriever did not become healthy within 20 minutes')
        time.sleep(10)
    print('CPU_RETRIEVER_HEALTHY_SECONDS', time.perf_counter() - started, flush=True)
    report = root/'runs/cpu-sdpa-20261003/ranker-probe.json'
    result = subprocess.run([python, str(root/'repo/reproduction/baselines/cosearch_probe.py'),
        '--model', str(root/'assets-root/assets/qwen2.5-3b'), '--train', config['data']['train_files'],
        '--url', 'http://127.0.0.1:18086/retrieve', '--output', str(report),
        '--threads', '32', '--questions', '1', '--repeats', '1', '--attention', 'sdpa'], env=env, cwd=root/'repo')
    print('STANDALONE_RANKER_PROBE_EXIT', result.returncode, flush=True)
    raise SystemExit(result.returncode)
finally:
    if service.poll() is None:
        signal.pidfd_send_signal(fd, signal.SIGTERM)
        try:
            service.wait(timeout=30)
        except subprocess.TimeoutExpired:
            signal.pidfd_send_signal(fd, signal.SIGKILL)
            service.wait()
    os.close(fd)
    log.close()
