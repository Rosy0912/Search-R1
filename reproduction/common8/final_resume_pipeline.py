"""Owned native services for CP200 evaluation only; no training launch paths.

The manifest is prospective *evaluation* configuration, not a fresh-training
manifest. Original model/data/index files are read in place and never deleted.
"""
import argparse
import csv
import fcntl
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.request
from urllib.parse import urlparse

from reproduction.a800.owned_processes import OwnedProcesses, publish, same
from reproduction.common8.native_allocation import allocated_indices, cuda_uuids, fields
from reproduction.common8.stage_retrieval import memory_bytes

_interrupt = None


def interrupted(sig, frame):
    global _interrupt
    _interrupt = sig  # Ownership registration must finish before cleanup.


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_manifest(base):
    path = base / 'preparation/final-evaluation-manifest.json'
    manifest = json.loads(path.read_text())
    assert manifest['role'] == 'final_only_cp200_resume'
    assert manifest['source_completed_outer'] == 200
    assert Path(manifest['base']).resolve() == base
    assert manifest['world_size'] == 8 and manifest['cpu_request'] == 96
    assert manifest['memory_request'] == '512G'
    for name in ('source_base', 'binding', 'config', 'journal', 'gpu_lock_directory'):
        assert Path(manifest[name]).is_absolute(), name
    binding = json.loads(Path(manifest['binding']).read_text())
    assert binding['role'] == manifest['role'] and binding['checkpoint_step'] == 200
    assert binding['run_id'] == manifest['run_id']
    assert sha(manifest['config']) == binding['resolved_config_sha256']
    assert sha(binding['checkpoint_manifest']) == binding['checkpoint_manifest_sha256']
    assert Path(binding['checkpoint']).is_dir()
    parsed = urlparse(manifest['retrieval_url'])
    assert parsed.scheme == 'http' and parsed.hostname in ('127.0.0.1', 'localhost')
    assert parsed.path == '/retrieve' and parsed.port and 1024 <= parsed.port <= 65535
    return manifest


def rows(query, kind='gpu'):
    raw = subprocess.check_output(['nvidia-smi', '--query-' + kind + '=' + query,
                                   '--format=csv,noheader,nounits'], text=True, timeout=30)
    return [[v.strip() for v in row] for row in csv.reader(io.StringIO(raw)) if row]


def native_admission(base, manifest):
    job_id = os.environ['SLURM_JOB_ID']
    detail = subprocess.check_output(['scontrol', 'show', 'job', '-d', job_id], text=True)
    job = fields(detail)
    node = socket.gethostname().split('.')[0]
    assert node == 'h100-8' and job['NodeList'] == node
    assert job['JobId'] == job_id and job['JobState'] == 'RUNNING' and job['NumNodes'] == '1'
    assert job['TresPerNode'] in ('gres:gpu:8', 'gres:gpu:h100:8')
    assert job['OverSubscribe'] == 'NO' and int(job['NumCPUs']) >= 96
    assert memory_bytes(job['TRES']) >= 512 * 2**30
    active = subprocess.check_output(['squeue', '-h', '-w', node, '-t', 'RUNNING,COMPLETING',
                                      '-o', '%i'], text=True).split()
    assert set(active) == {job_id}, 'Another native allocation remains; preserve it and refuse overlap'
    physical = {int(i): uuid for i, uuid in rows('index,uuid')}
    visible = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    allocated = allocated_indices(detail, node)
    expected = [physical[int(v)] if v.isdecimal() else v for v in visible]
    assert len(expected) == len(set(expected)) == 8
    assert set(expected) == {physical[i] for i in allocated}
    assert expected == manifest['physical_gpu_uuids'] and cuda_uuids() == expected
    permitted = manifest.get('preserved_foreign_gpu_processes', [])
    allowed = {(record['gpu_uuid'], str(record['identity']['pid']))
               for record in permitted if same(record['identity'])}
    apps = rows('gpu_uuid,pid', 'compute-apps')
    assert all(tuple(row) in allowed or int(row[1]) == os.getpid() for row in apps), \
        'Unbound GPU application; never stop it'
    proof = dict(role=manifest['role'], run_id=manifest['run_id'],
                 execution_id=manifest['execution_id'], accepted=True, node=node,
                 slurm_job_id=job_id, actual_cuda_runtime_uuids=expected,
                 source_completed_outer=200, cpu_request=96, memory_request='512G',
                 self_cgroup=(Path('/proc/self/cgroup')).read_text(), checked_unix=time.time())
    publish(base / 'logs/final-native-allocation.json', proof)
    return proof


class UUIDLocks:
    """New final-only UUID locks; not a claim about previous jobs' locking."""
    def __init__(self, manifest):
        assert sys.platform.startswith('linux'), 'Native OFD lease uses the Linux flock ABI'
        root = Path(manifest['gpu_lock_directory'])
        assert root.is_dir(), 'Explicit GPU lock directory must be prepared'
        self.fds = []
        template = manifest.get('gpu_lock_filename', '{uuid}.lock')
        assert '/' not in template and '{uuid}' in template
        try:
            for uuid in sorted(manifest['physical_gpu_uuids']):
                fd = os.open(root / template.format(uuid=uuid), os.O_RDWR | os.O_CREAT, 0o600)
                self.fds.append(fd)
                fcntl.fcntl(fd, fcntl.F_OFD_SETLK,
                            struct.pack('hhqqi4x', fcntl.F_WRLCK, os.SEEK_SET, 0, 0, 0))
        except BaseException:
            self.close()
            raise
    def close(self):
        for fd in self.fds:
            os.close(fd)
        self.fds = []


class FinalPipeline:
    def __init__(self, base, manifest):
        self.base, self.manifest = base, manifest
        self.repo = base / 'repo'
        self.python = str(base / 'env/bin/python')
        self.visible = os.environ['CUDA_VISIBLE_DEVICES']
        assert not (base / 'logs/process-ledger.json').exists(), 'Never replay an execution namespace'
        self.owned = OwnedProcesses(base / 'logs/process-ledger.json', manifest['run_id'])
        self.env = dict(os.environ, SEARCH_G1_PROFILE='common8', SEARCH_G1_BASE=str(base),
                        PYTHONPATH=str(self.repo), PYTHONUNBUFFERED='1', OMP_NUM_THREADS='8',
                        HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', ECO_USE_CF='0',
                        TMPDIR=str(base / 'tmp'), RAY_TMPDIR=manifest['ray_tmp_dir'],
                        HF_HOME=str(base / 'cache/hf'), XDG_CACHE_HOME=str(base / 'cache'),
                        TRITON_CACHE_DIR=str(base / 'cache/triton'),
                        TORCHINDUCTOR_CACHE_DIR=str(base / 'cache/inductor'),
                        NCCL_NVLS_ENABLE='0', VLLM_ATTENTION_BACKEND='XFORMERS',
                        NO_PROXY='127.0.0.1,localhost')
        for name in ('PYTHONHOME', 'RAY_ADDRESS', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY',
                     'http_proxy', 'https_proxy', 'all_proxy'):
            self.env.pop(name, None)
        include = base / 'bootstrap-tools/python-dev/usr/include'
        if include.is_dir():
            self.env['C_INCLUDE_PATH'] = str(include / 'python3.10') + ':' + str(include)
        self.env['PATH'] = str(Path(self.python).parent) + ':' + os.environ['PATH']
        libraries = list((base / 'env/lib/python3.10/site-packages/nvidia').glob('*/lib'))
        libraries.append(base / 'env/lib/python3.10/site-packages/torch/lib')
        self.env['LD_LIBRARY_PATH'] = ':'.join(str(path) for path in libraries if path.is_dir())

    def stage(self, name, **details):
        publish(self.base / 'logs/final-pipeline-status.json', dict(stage=name,
                run_id=self.manifest['run_id'], execution_id=self.manifest['execution_id'],
                checked_unix=time.time(), **details))

    def observe(self):
        for name, job in self.owned.jobs.items():
            if job['state'] != 'STOPPED':
                self.owned.observe(name)
        assert _interrupt is None, 'Final evaluation interrupted; clean only owned services'

    def start(self, name, args, extra=None):
        self.observe()
        env = dict(self.env, CUDA_VISIBLE_DEVICES=self.visible)
        env.update(extra or {})
        process = self.owned.start(name, args, env, self.base / ('logs/' + name + '.log'), self.repo)
        self.observe()
        return process

    def wait(self, name, process):
        while True:
            self.observe()
            code = process.poll()
            if code is not None:
                self.owned.jobs[name]['exit_code'] = code
                self.owned.save()
                assert code == 0, name + ' failed; retained owned log'
                assert not self.owned.observe(name), name + ' left owned subprocesses'
                self.owned.jobs[name]['state'] = 'STOPPED'
                self.owned.save()
                return
            time.sleep(1)

    def execute(self):
        self.stage('STARTING_FINAL_ONLY_RETRIEVER')
        try:
            retriever = self.start('final-retriever', [self.python, '-m',
                                   'reproduction.common8.final_resume_pipeline', '--base',
                                   str(self.base), '--component', 'retriever'],
                                   {'PYTHONPATH':str(Path(self.manifest['retriever_gpu_lib'])) + ':' + str(self.repo)})
            url = self.manifest['retrieval_url'].rsplit('/', 1)[0] + '/health'
            deadline = time.monotonic() + self.manifest.get('retriever_start_timeout_seconds', 1800)
            while True:
                self.observe()
                assert retriever.poll() is None, 'Retriever exited during initialization'
                try:
                    with urllib.request.urlopen(url, timeout=3) as response:
                        health = json.load(response)
                    assert health['execution_id'] == self.manifest['execution_id']
                    assert health['run_id'] == self.manifest['run_id'] and health['shard_count'] == 8
                    break
                except (OSError, TimeoutError):
                    assert time.monotonic() < deadline, 'Retriever initialization timed out'
                    time.sleep(2)
            self.stage('FINAL51713_FROM_CP200', actual_retriever=health)
            m = self.manifest
            evaluator = self.start('final-evaluator', [self.python, '-m',
                'reproduction.common8.final_resume_runner', '--binding', m['binding'],
                '--config', m['config'], '--source-base', m['source_base'], '--journal', m['journal']])
            self.wait('final-evaluator', evaluator)
            self.stage('FINAL_EVALUATOR_EXIT0', training_executed=False)
        except BaseException as error:
            self.stage('FAILED', error=repr(error))
            raise
        finally:
            self.owned.stop_all()
            pids = {str(r['pid']) for job in self.owned.jobs.values() for r in job['ledger'].values()}
            deadline = time.monotonic() + 60
            while True:
                remaining = [r for r in rows('gpu_uuid,pid', 'compute-apps') if r[1] in pids]
                if not remaining:
                    break
                assert time.monotonic() < deadline, 'Owned GPU contexts have not released'
                time.sleep(.5)
            publish(self.base / 'logs/cleanup-status.json', dict(cleaned=True,
                    run_id=self.manifest['run_id'], execution_id=self.manifest['execution_id'],
                    checked_unix=time.time(), original_assets_deleted=False,
                    foreign_processes_signalled=False))


def serve_retriever(base, manifest):
    import datasets
    import faiss
    import torch
    import uvicorn
    assert torch.cuda.device_count() == faiss.get_num_gpus() == 8
    assert cuda_uuids() == manifest['physical_gpu_uuids']
    assert Path(faiss.__file__).resolve().is_relative_to(Path(manifest['retriever_gpu_lib']).resolve())
    source = Path(manifest['retrieval_server_source'])
    spec = importlib.util.spec_from_file_location('final_original_retrieval', source)
    upstream = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = upstream
    spec.loader.exec_module(upstream)
    corpus_path = Path(manifest['retrieval_corpus_path'])
    assert corpus_path.is_file() and Path(manifest['retrieval_index_path']).is_file()
    def corpus(path):
        assert Path(path).resolve() == corpus_path.resolve()
        return datasets.load_dataset('json', data_files=str(corpus_path), split='train',
                num_proc=4, cache_dir=str(base / 'cache/corpus'))
    upstream.load_corpus = corpus
    upstream.config = upstream.Config(retrieval_method='e5', retrieval_topk=3,
            index_path=manifest['retrieval_index_path'], corpus_path=str(corpus_path), faiss_gpu=True,
            retrieval_model_path=manifest['retrieval_encoder'], retrieval_pooling_method='mean',
            retrieval_query_max_length=256, retrieval_use_fp16=True, retrieval_batch_size=512)
    upstream.retriever = upstream.get_retriever(upstream.config)
    assert upstream.retriever.index.count() == 8 and upstream.retriever.index.d == 768
    assert upstream.retriever.index.ntotal == len(upstream.retriever.corpus)
    original_batch = upstream.retriever.batch_search
    thread = threading.get_ident()
    def serial_batch(*args, **kwargs):
        assert threading.get_ident() == thread, 'FAISS must remain on its initialization thread'
        return original_batch(*args, **kwargs)
    upstream.retriever.batch_search = serial_batch
    @upstream.app.get('/health')
    async def health():
        return dict(ok=True, run_id=manifest['run_id'], execution_id=manifest['execution_id'],
                    shard_count=8, ntotal=upstream.retriever.index.ntotal, original_topk=3)
    uvicorn.run(upstream.app, host='127.0.0.1', port=urlparse(manifest['retrieval_url']).port, workers=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--component', choices=['pipeline', 'retriever'], default='pipeline')
    args = parser.parse_args()
    base = args.base.resolve()
    manifest = read_manifest(base)
    if args.component == 'retriever':
        return serve_retriever(base, manifest)
    for name in ('logs', 'tmp', 'ray', 'cache/hf', 'cache/triton', 'cache/inductor', 'cache/corpus'):
        (base / name).mkdir(parents=True, exist_ok=True)
    ray_tmp = Path(manifest['ray_tmp_dir'])
    assert ray_tmp.is_absolute() and str(ray_tmp).startswith('/tmp/g1f-')
    assert len(str(ray_tmp)) < 32, 'Ray Unix sockets require a short private root'
    ray_tmp.mkdir(mode=0o700, exist_ok=False)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    with (base / 'final-execution.lock').open('a') as claim:
        fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lease = UUIDLocks(manifest)
        try:
            native_admission(base, manifest)
            parsed = urlparse(manifest['retrieval_url'])
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', parsed.port))
            FinalPipeline(base, manifest).execute()
        finally:
            lease.close()


if __name__ == '__main__':
    main()
