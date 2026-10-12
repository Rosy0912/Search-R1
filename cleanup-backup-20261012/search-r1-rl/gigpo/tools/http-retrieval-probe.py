"""Own CPU retriever, real corpus and HTTP top3/top50, with process-bound cleanup."""
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import requests

root = Path(os.environ['AGENT_ROOT'])
with socket.socket() as sock:
    sock.bind(('127.0.0.1', 18081))
command = [sys.executable, str(root/'repo/reproduction/h20/retriever_local.py'),
           '--index_path', str(root/'assets-root/assets/wiki/e5_Flat.index'),
           '--corpus_path', str(root/'assets-root/assets/wiki/wiki-18.jsonl'),
           '--retriever_model', str(root/'assets-root/assets/e5-base-v2'),
           '--retriever_name', 'e5', '--topk', '3', '--faiss_gpu', '--device', 'cuda',
           '--host', '127.0.0.1', '--port', '18081', '--threads', '8']
session=requests.Session();session.trust_env=False
started = time.perf_counter()
with (root/'logs/http-retriever-server.log').open('w') as log:
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONPATH=str(root/'retriever-gpu-lib')+os.pathsep+str(root/'repo')))
    pidfd = os.pidfd_open(process.pid)
    try:
        deadline = time.monotonic()+720
        while time.monotonic() < deadline:
            assert process.poll() is None, 'Own CPU retriever exited before health check'
            try:
                response = session.get('http://127.0.0.1:18081/openapi.json', timeout=2)
                if response.ok:
                    break
            except requests.RequestException:
                pass
            time.sleep(2)
        else:
            raise RuntimeError('Own CPU retriever did not become healthy in 720 seconds')
        subprocess.run([sys.executable,str(root/'tools/check_http_retrieval.py')],check=True)
        stat = (Path('/proc')/str(process.pid)/'status').read_text()
        result = {'job_id':os.environ['SLURM_JOB_ID'], 'pid':process.pid,
                  'mode':'fixed original GPU E5 FP16 / complete eight-GPU sharded FP16 FAISS / complete wiki18',
                  'startup_and_probe_wall_seconds':time.perf_counter()-started,
                  'threads':8, 'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],
                  'process_memory':[line for line in stat.splitlines() if line.startswith(('VmRSS:', 'VmHWM:'))],
                  'local_http_reference_receipt':str(root/'logs/local-http-retrieval-verification.json')}
        (root/'logs/http-retrieval-probe.json').write_text(json.dumps(result, indent=2))
        print('GIGPO_REAL_HTTP_RETRIEVAL_TOP3_TOP50_PASS', flush=True)
    finally:
        if process.poll() is None:
            signal.pidfd_send_signal(pidfd, signal.SIGTERM)
        os.close(pidfd)
        process.wait(timeout=60)
