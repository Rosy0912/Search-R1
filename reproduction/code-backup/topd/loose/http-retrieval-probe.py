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
    sock.bind(('127.0.0.1', 18085))
command = [sys.executable, str(root/'repo/search_r1/search/retrieval_server.py'),
           '--index_path', str(root/'assets-root/assets/wiki/e5_Flat.index'),
           '--corpus_path', str(root/'assets-root/assets/wiki/wiki-18.jsonl'),
           '--retriever_model', str(root/'assets-root/assets/e5-base-v2'),
           '--retriever_name', 'e5', '--topk', '3', '--device', 'cpu',
           '--host', '127.0.0.1', '--port', '18085', '--threads', '8']
started = time.perf_counter()
with (root/'logs/http-retriever-server.log').open('w') as log:
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    pidfd = os.pidfd_open(process.pid)
    try:
        deadline = time.monotonic()+720
        while time.monotonic() < deadline:
            assert process.poll() is None, 'Own CPU retriever exited before health check'
            try:
                response = requests.get('http://127.0.0.1:18085/openapi.json', timeout=2)
                if response.ok:
                    break
            except requests.RequestException:
                pass
            time.sleep(2)
        else:
            raise RuntimeError('Own CPU retriever did not become healthy in 720 seconds')
        records = []
        for count in (3, 50):
            tick = time.perf_counter()
            response = requests.post('http://127.0.0.1:18085/retrieve',
                json={'queries':['What is the capital of France?'], 'topk':count,
                      'return_scores':True}, timeout=120)
            response.raise_for_status()
            payload = response.json()
            assert len(payload['result']) == 1 and len(payload['result'][0]) == count
            assert all(isinstance(d['document']['contents'], str) and d['document']['contents']
                       for d in payload['result'][0])
            records.append({'topk':count, 'wall_seconds':time.perf_counter()-tick,
                            'response':payload,
                            'document_sha256':[hashlib.sha256(d['document']['contents'].encode()).hexdigest()
                                               for d in payload['result'][0]]})
        stat = (Path('/proc')/str(process.pid)/'status').read_text()
        result = {'job_id':os.environ['SLURM_JOB_ID'], 'pid':process.pid,
                  'mode':'fixed original CPU E5 FP16 / complete CPU FAISS / complete wiki18',
                  'startup_and_probe_wall_seconds':time.perf_counter()-started,
                  'threads':8, 'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],
                  'process_memory':[line for line in stat.splitlines() if line.startswith(('VmRSS:', 'VmHWM:'))],
                  'records':records}
        (root/'logs/http-retrieval-probe.json').write_text(json.dumps(result, indent=2))
        print('TOPD_REAL_HTTP_RETRIEVAL_TOP3_TOP50_PASS', flush=True)
    finally:
        if process.poll() is None:
            signal.pidfd_send_signal(pidfd, signal.SIGTERM)
        os.close(pidfd)
        process.wait(timeout=60)
