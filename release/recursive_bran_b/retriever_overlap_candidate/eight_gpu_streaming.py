"""Eight exact FP32 tile workers; original-order top-k reduction on GPU7.

No shard-wide aggregate is allowed: each original tile's local top-k is retained
and fed into the old cat/top-k/gather merge in global tile/request/micro order.
Score/ID equivalence across GPU devices still requires the actual numerical gate.
"""
from dataclasses import dataclass, field
import os
import queue
import threading
from collections import Counter
import numpy as np
import torch
from pinned_overlap import (VECTOR_ROWS, QUERY_MICRO, Tile, tile_plan,
                            memory_plan, TorchCopyBackend, DoubleBufferTiles)


GPU_COUNT = 8
MERGE_GPU = 7


def partition_tiles(ntotal, gpu_count=GPU_COUNT):
    """Whole original tiles, round robin, to keep parallel workers in lockstep."""
    if gpu_count not in (4, 8):
        raise ValueError('Explicit four- or eight-device whole-tile partition required')
    groups = [[] for _ in range(gpu_count)]
    for tile in tile_plan(ntotal):
        groups[tile.number % gpu_count].append(tile)
    return groups


def eight_gpu_memory_plan(ntotal, dimension, total_queries=0, k=3, gpu_count=GPU_COUNT):
    base = memory_plan(ntotal, dimension)
    groups = partition_tiles(ntotal, gpu_count)
    return dict(index_scan_bytes=base['index_scan_bytes'],
        per_gpu_vector_transfer_bytes=[sum(t.stop-t.start for t in rows)*dimension*4 for rows in groups],
        per_gpu_tile_counts=[len(rows) for rows in groups],
        per_gpu_pinned_cpu_buffers_bytes=base['pinned_cpu_buffers_bytes'],
        per_gpu_device_buffers_bytes=base['device_buffers_bytes'],
        all_workers_pinned_cpu_buffers_bytes=gpu_count*base['pinned_cpu_buffers_bytes'],
        per_gpu_query_copy_bytes=total_queries*dimension*4,
        per_gpu_distances_peak_bytes=base['max_one_query_micro_distances_bytes'],
        cpu_two_ready_tiles_per_gpu_bytes_upper=gpu_count*2*total_queries*k*12,
        cpu_one_pending_producer_tile_per_gpu_bytes_upper=gpu_count*total_queries*k*12,
        merge_gpu=gpu_count-1, gpu_count=gpu_count, resident_full_index=False,
        limits='Not measured peaks: CUDA contexts, allocator reserve, FAISS heap/mmap pages, merge scratch and oracle coexistence excluded')


def merge_tile_original_order(best_s, best_i, scores, local_ids, start, k, ops):
    """Shared actual reduction used by GPU7 and CPU same-score/tie tests."""
    merged_s = ops.cat((best_s, scores), dim=1)
    merged_i = ops.cat((best_i, local_ids + start), dim=1)
    values, choice = ops.topk(merged_s, k, dim=1, sorted=True)
    return values, ops.gather(merged_i, 1, choice)


@dataclass
class ScanJob:
    requests: list
    tiles: list
    trace: object
    cancel: threading.Event
    output: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=2))
    done: threading.Event = field(default_factory=threading.Event)
    error: object = None
    actual_gemm_shapes: Counter = field(default_factory=Counter)
    actual_tile_starts: list = field(default_factory=list)
    thread_native_id: object = None


def publish(job, value):
    while not job.cancel.is_set():
        try:
            job.output.put(value, timeout=.1)
            return True
        except queue.Full:
            pass
    return False


class TileWorker(threading.Thread):
    def __init__(self, gpu, host):
        super().__init__(name='fp32-tile-gpu-%d' % gpu, daemon=True)
        self.gpu, self.host = gpu, host
        self.inbox = queue.Queue(maxsize=1)
        self.buffers = None
        self.start()

    def run(self):
        while True:
            job = self.inbox.get()
            if job is None:
                if self.buffers is not None: self.buffers.close()
                return
            try:
                self.execute(job)
            except BaseException as error:
                job.error = error
                job.cancel.set()
            finally:
                if self.buffers is not None:
                    try: self.buffers.close()
                    except BaseException as error:
                        job.error = job.error or error
                        job.cancel.set()
                job.done.set()

    @torch.no_grad()
    def execute(self, job):
        if not job.tiles: return
        job.thread_native_id = threading.get_native_id()
        device = torch.device('cuda:%d' % self.gpu)
        with torch.cuda.device(device):
            if self.buffers is None:
                self.compute = torch.cuda.Stream(device=device)
                with torch.cuda.stream(self.compute):
                    self.buffers = DoubleBufferTiles(TorchCopyBackend(self.host, device))
            with torch.cuda.stream(self.compute):
                requests = [(request_id, q, k, torch.from_numpy(q).to(device))
                            for request_id, q, k in job.requests if len(q)]
                self.buffers.stage(Tile(0, job.tiles[0].start, job.tiles[0].stop))
                for number, tile in enumerate(job.tiles):
                    if job.cancel.is_set(): break
                    if number+1 < len(job.tiles):
                        following = job.tiles[number+1]
                        self.buffers.stage(Tile(number+1, following.start, following.stop))
                    buffer_tile = Tile(number, tile.start, tile.stop)
                    vectors = self.buffers.acquire(buffer_tile)
                    job.actual_tile_starts.append(tile.start)
                    if job.trace is not None:
                        job.trace(dict(event='vector_transfer', start=tile.start,
                            shape=list(vectors.shape), gpu=self.gpu,
                            thread_native_id=threading.get_native_id()))
                    segments = []
                    for request_id, q, k, tq in requests:
                        local_k = min(k, len(vectors))
                        for offset in range(0, len(q), QUERY_MICRO):
                            stop = min(offset+QUERY_MICRO, len(q))
                            left, right = tq[offset:stop], vectors.T
                            distances = left @ right
                            job.actual_gemm_shapes[(request_id,offset,stop-offset,self.host.shape[1],
                                len(vectors),local_k,k,str(distances.dtype))] += 1
                            if distances.dtype != torch.float32:
                                raise ValueError('Worker GEMM was not FP32')
                            if job.trace is not None:
                                job.trace(dict(event='gemm', request=request_id, start=tile.start,
                                    offset=offset, left=list(left.shape), right=list(right.shape),
                                    local_k=local_k, k=k, gpu=self.gpu,
                                    dtype=str(distances.dtype), thread_native_id=threading.get_native_id()))
                            scores, ids = torch.topk(distances, local_k, dim=1, sorted=True)
                            segments.append((request_id, offset, stop, scores, ids))
                            del distances, scores, ids
                    # Transport values unchanged, with two device waits per tile,
                    # not two waits per request/query micro-batch.
                    packed_scores = torch.cat([value[3].reshape(-1) for value in segments]).cpu()
                    packed_ids = torch.cat([value[4].reshape(-1) for value in segments]).cpu()
                    host_segments, position = [], 0
                    for request_id, offset, stop, scores, ids in segments:
                        n = scores.numel(); shape = scores.shape
                        host_segments.append((request_id, offset, stop,
                            packed_scores[position:position+n].reshape(shape),
                            packed_ids[position:position+n].reshape(shape)))
                        position += n
                    self.buffers.consumed(buffer_tile)
                    if not publish(job, (tile, host_segments)): break


class EightGpuStreamingFlatIP:
    def __init__(self, original, gpu_count=GPU_COUNT):
        self.original = original
        self.host, self.d, self.ntotal = original.host, original.d, original.ntotal
        if gpu_count not in (4, 8):
            raise ValueError('Only an explicit four/eight visible-device profile is supported')
        self.gpu_count, self.merge_gpu = gpu_count, gpu_count-1
        self.device = torch.device('cuda:%d' % self.merge_gpu)
        self.vector_rows, self.query_micro = VECTOR_ROWS, QUERY_MICRO
        if (original.vector_rows != VECTOR_ROWS or original.query_micro != QUERY_MICRO or
                self.host.dtype != np.float32 or self.host.shape != (self.ntotal, self.d) or
                not self.host.flags.c_contiguous):
            raise ValueError('Original contiguous FP32 32768/64 host contract required')
        if torch.cuda.device_count() != gpu_count:
            raise ValueError('Visible devices differ from the explicitly authorized count')
        self.scan_lock = threading.Lock()
        self.groups = partition_tiles(self.ntotal, gpu_count)
        self.workers = [TileWorker(gpu, self.host) for gpu in range(gpu_count)]
        self.last_scan_profile = None

    def search(self, queries, k):
        return self.search_many([queries], [k])[0]

    def _requests(self, list_queries, list_ks):
        if len(list_queries) != len(list_ks) or len(list_queries) > 8:
            raise ValueError('Each of at most 8 requests requires its own top-k')
        result, total = [], 0
        for request_id, (queries, k) in enumerate(zip(list_queries, list_ks)):
            q = np.ascontiguousarray(queries, dtype=np.float32)
            if q.ndim != 2 or q.shape[1] != self.d or not np.isfinite(q).all():
                raise ValueError('Invalid query embeddings')
            if not 0 < int(k) <= self.ntotal or len(q) > 2560:
                raise ValueError('Invalid k or per-request query count')
            k = int(k); total += len(q)
            if total > 20480: raise ValueError('At most 20480 queries per shared scan')
            result.append((request_id, q, k))
        return result

    @torch.no_grad()
    def search_many(self, list_queries, list_ks, trace=None):
        if (torch.is_autocast_enabled() or torch.backends.cuda.matmul.allow_tf32 or
                torch.backends.cudnn.allow_tf32):
            raise ValueError('Exact FP32 candidate requires no CUDA autocast and TF32 disabled')
        requests = self._requests(list_queries, list_ks)
        if not any(len(q) for _, q, _ in requests):
            return [(np.empty((0,k),np.float32), np.empty((0,k),np.int64)) for _,q,k in requests]
        with self.scan_lock, torch.cuda.device(self.device):
            cancel = threading.Event()
            jobs = [ScanJob(requests, tiles, trace, cancel) for tiles in self.groups]
            best = [(torch.full((len(q), k), -float('inf'), dtype=torch.float32, device=self.device),
                     torch.full((len(q), k), -1, dtype=torch.int64, device=self.device))
                    if len(q) else (None, None) for _,q,k in requests]
            for worker, job in zip(self.workers, jobs): worker.inbox.put(job)
            tiles_consumed = 0
            try:
                for tile in tile_plan(self.ntotal):
                    job = jobs[tile.number % self.gpu_count]
                    while True:
                        if cancel.is_set():
                            raise RuntimeError('Eight-GPU tile worker fault: ' + repr(next(
                                (value.error for value in jobs if value.error is not None), None)))
                        try: observed, segments = job.output.get(timeout=.1); break
                        except queue.Empty:
                            if job.done.is_set():
                                raise RuntimeError('Tile worker ended before expected tile')
                    if observed != tile: raise ValueError('Original global tile order was lost')
                    for request_id, offset, stop, scores, ids in segments:
                        _, q, k = requests[request_id]
                        best_s, best_i = best[request_id]
                        values, chosen = merge_tile_original_order(best_s[offset:stop], best_i[offset:stop],
                            scores.to(self.device), ids.to(self.device), tile.start, k, torch)
                        best_s[offset:stop], best_i[offset:stop] = values, chosen
                    tiles_consumed += 1
                for job in jobs: job.done.wait()
                if any(job.error is not None for job in jobs):
                    raise RuntimeError('Worker error after tile production')
                self.last_scan_profile = dict(tiles_merged=tiles_consumed, merge_gpu=self.merge_gpu,
                    actual_worker_shapes=[dict(gpu=gpu,thread_native_id=job.thread_native_id,
                        tile_starts=job.actual_tile_starts, gemm_shape_counts=[dict(
                            request=key[0],offset=key[1],left=[key[2],key[3]],right=[key[3],key[4]],
                            local_k=key[5],k=key[6],dtype=key[7],count=count)
                            for key,count in sorted(job.actual_gemm_shapes.items())])
                        for gpu,job in enumerate(jobs)],
                    partition='round_robin_whole_original_tiles',
                    memory=eight_gpu_memory_plan(self.ntotal,self.d,sum(len(q) for _,q,_ in requests),gpu_count=self.gpu_count),
                    actual_gpu_memory=[dict(gpu=i, allocated_bytes=torch.cuda.max_memory_allocated(i),
                        reserved_bytes=torch.cuda.max_memory_reserved(i),
                        device_name=torch.cuda.get_device_properties(i).name)
                        for i in range(self.gpu_count)])
                return [(np.empty((0,k),np.float32), np.empty((0,k),np.int64)) if not len(q)
                        else (scores.cpu().numpy(), ids.cpu().numpy())
                        for (_,q,k),(scores,ids) in zip(requests,best)]
            finally:
                cancel.set()
                for job in jobs: job.done.wait()

    def close(self):
        with self.scan_lock:
            for worker in self.workers: worker.inbox.put(None)
            for worker in self.workers: worker.join()
