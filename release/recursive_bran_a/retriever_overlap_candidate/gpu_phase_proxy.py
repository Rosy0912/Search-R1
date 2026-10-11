"""CPU service proxy for a phase-scoped spawned eight-GPU scan process.

The parent never imports the GPU backend or calls CUDA. Releasing the child
releases its CUDA contexts; CPU E5/index/corpus parent and original oracle live.
"""
import hashlib
import multiprocessing
import os
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np


def source_manifest():
    base = Path(__file__).resolve().parent
    names = ('gpu_phase_proxy.py','eight_gpu_streaming.py','pinned_overlap.py','flat_memmap.py')
    return {name:hashlib.sha256((base/name).read_bytes()).hexdigest() for name in names}


def process_identity():
    stat = Path('/proc/%d/stat' % os.getpid())
    tick = int(stat.read_text().rsplit(') ',1)[1].split()[19]) if stat.exists() else None
    return dict(pid=os.getpid(),uid=os.getuid(),starttick=tick)


class SpawnedRuntime:
    """Private child ownership, serialization and release; CPU-testable lifecycle."""
    def __init__(self, target, configuration, timeout=180):
        self.target, self.configuration, self.timeout = target, configuration, timeout
        self.context = multiprocessing.get_context('spawn')
        self.process = self.connection = None
        self.identity = None
        self.last_release = None

    def ensure_started(self):
        if self.process is not None:
            if self.process.is_alive(): return self.identity
            self.release()
        parent, child = self.context.Pipe(duplex=True)
        process = self.context.Process(target=self.target, args=(child,self.configuration),
            name='eight-gpu-phase-worker', daemon=False)
        process.start(); child.close()
        self.process, self.connection = process, parent
        try:
            ready = self.receive()
            if ready.get('status') != 'READY' or ready.get('identity',{}).get('pid') != process.pid:
                raise RuntimeError('GPU child startup rejected: ' + repr(ready))
            self.identity = ready['identity']
            return self.identity
        except BaseException:
            self.release()
            raise

    def receive(self):
        if not self.connection.poll(self.timeout):
            raise TimeoutError('Own phase worker did not respond within bounded wait')
        return self.connection.recv()

    def request(self, payload):
        self.ensure_started()
        try:
            self.connection.send(payload)
            result = self.receive()
            if result.get('status') != 'OK':
                raise RuntimeError('Phase worker request fault: ' + repr(result))
            return result
        except BaseException:
            self.release()
            raise

    def release(self):
        process, connection = self.process, self.connection
        if process is None:
            return self.last_release or dict(status='NO_GPU_CHILD',exitcode=None)
        identity, graceful, error = self.identity, False, None
        try:
            if process.is_alive() and connection is not None:
                connection.send(dict(op='release'))
                if connection.poll(self.timeout):
                    acknowledgement = connection.recv()
                    graceful = acknowledgement.get('status') == 'RELEASED'
                else:
                    error = 'Release acknowledgement timeout'
            process.join(timeout=self.timeout)
            if process.is_alive():
                # Only the proven child Process handle, never an oracle/foreign PID.
                process.terminate(); process.join(timeout=10)
                if process.is_alive(): process.kill(); process.join(timeout=10)
                graceful = False
            if process.is_alive():
                raise RuntimeError('Own GPU worker has not exited; HF must remain blocked')
        except (BrokenPipeError, EOFError, OSError) as exc:
            error = repr(exc)
            if process.is_alive(): process.terminate()
            process.join(timeout=10)
            if process.is_alive(): process.kill(); process.join(timeout=10)
            if process.is_alive(): raise RuntimeError('Own GPU worker release failed')
        finally:
            if connection is not None: connection.close()
        receipt = dict(status='GPU_CHILD_EXITED', identity=identity, pid=process.pid,
            exitcode=process.exitcode, graceful=graceful and process.exitcode==0,
            release_error=error, released_at=time.time(),
            scope='Actual own process joined; external nvidia-smi context disappearance must also be checked')
        self.process = self.connection = None
        self.last_release, self.identity = receipt, None
        return receipt


def gpu_worker(connection, configuration):
    scan = None
    try:
        if source_manifest() != configuration['source_manifest']:
            raise ValueError('Spawned GPU implementation source changed')
        import torch
        import faiss
        from flat_memmap import runtime_layout, validated_prefix_mmap, verify_unchanged
        from eight_gpu_streaming import EightGpuStreamingFlatIP
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        layout = runtime_layout(faiss, configuration['dimension'])
        host, proof = validated_prefix_mmap(configuration['index_path'],layout,
                                            hash_selected_rows=0)
        for key in ('path','metadata_before','original_index_rows','dimension','payload_offset','actual_prefix_sha256'):
            if proof[key] != configuration['index_identity'][key]:
                raise ValueError('GPU worker readonly index identity changed: ' + key)
        original = SimpleNamespace(host=host,d=host.shape[1],ntotal=len(host),
            vector_rows=32768,query_micro=64,device=torch.device('cuda:7'))
        scan = EightGpuStreamingFlatIP(original)
        identity = dict(process_identity(), source_manifest=source_manifest(),
            index_identity=proof, cuda_initialized=torch.cuda.is_initialized(),
            role='eight_gpu_sampling_phase_only', source_fixed=True, fp32=True, tf32=False)
        connection.send(dict(status='READY',identity=identity))
        while True:
            request = connection.recv()
            if request['op'] == 'release':
                scan.close(); scan = None
                connection.send(dict(status='RELEASED',identity=identity))
                return
            if request['op'] != 'scan': raise ValueError('Unknown private worker operation')
            verify_unchanged(proof)
            events = [] if request.get('collect_trace') else None
            started = time.perf_counter()
            result = scan.search_many(request['queries'],request['ks'],
                                      trace=events.append if events is not None else None)
            verify_unchanged(proof)
            connection.send(dict(status='OK',results=result,trace=events,
                scan_profile=scan.last_scan_profile,worker_identity=identity,
                worker_wall_seconds=time.perf_counter()-started))
    except BaseException as error:
        try: connection.send(dict(status='FAULT',error=repr(error),identity=process_identity()))
        except (OSError,EOFError): pass
        raise
    finally:
        if scan is not None:
            try: scan.close()
            except BaseException: pass
        connection.close()


class PhaseGpuScanner:
    def __init__(self, original, index_path, oracle_source=None, timeout=180):
        dimension = original.d
        if original.vector_rows != 32768 or original.query_micro != 64:
            raise ValueError("Original scanner tile/query shape differs")
        import threading
        import faiss
        from flat_memmap import runtime_layout, validated_prefix_mmap
        host, proof = validated_prefix_mmap(index_path,runtime_layout(faiss,dimension),
                                            hash_selected_rows=0)
        self.ntotal, self.d = len(host), host.shape[1]
        if (self.ntotal, self.d) != (original.ntotal, original.d):
            raise ValueError("Original CPU scanner and validated index dimensions differ")
        # Only virtual mmap metadata was used; keep no parent GPU tensors/context.
        del host
        self.vector_rows, self.query_micro = 32768, 64
        self.scan_lock = threading.Lock()
        self.runtime = SpawnedRuntime(gpu_worker,dict(index_path=proof['path'],
            dimension=dimension,index_identity=proof,source_manifest=source_manifest()),timeout)
        self.last_scan_profile = None
        self.last_worker_identity = None

    def search(self, queries, k):
        return self.search_many([queries],[k])[0]

    def search_many(self,list_queries,list_ks,trace=None):
        # Arrays alone cross the private pipe; parent E5/FAISS/corpus stay alive.
        queries = [np.ascontiguousarray(q,dtype=np.float32) for q in list_queries]
        with self.scan_lock:
            response = self.runtime.request(dict(op='scan',queries=queries,ks=list_ks,
                                                 collect_trace=trace is not None))
            self.last_scan_profile = response['scan_profile']
            self.last_worker_identity = response['worker_identity']
            if trace is not None:
                for event in response['trace']: trace(event)
            return response['results']

    @property
    def worker_pid(self):
        process = self.runtime.process
        return process.pid if process is not None and process.is_alive() else None

    def release_gpu(self):
        with self.scan_lock:
            receipt = self.runtime.release()
            return dict(receipt, released=True, gpu_worker_pid=None,
                        previous_gpu_worker_pid=receipt.get('pid'))

    def child_health(self):
        process = self.runtime.process
        return dict(gpu_child_live=process is not None and process.is_alive(),
            gpu_child_identity=self.runtime.identity,last_gpu_release=self.runtime.last_release,
            source_fixed=True,index_identity=self.runtime.configuration['index_identity'])

    def close(self):
        return self.release_gpu()
