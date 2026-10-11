"""Independent bounded FP32 scan candidate; never changes the oracle service.

H2D bytes are unchanged. Only copying a later tile can overlap computation of
an earlier tile. Request/query/tile GEMM and top-k operation order stay fixed.
"""
from dataclasses import dataclass
import numpy as np
import torch

VECTOR_ROWS = 32768
QUERY_MICRO = 64


@dataclass(frozen=True)
class Tile:
    number: int
    start: int
    stop: int

    @property
    def slot(self):
        return self.number % 2


def tile_plan(ntotal):
    if type(ntotal) is not int or ntotal < 0:
        raise ValueError('Index rows must be a nonnegative integer')
    return [Tile(i, start, min(start + VECTOR_ROWS, ntotal))
            for i, start in enumerate(range(0, ntotal, VECTOR_ROWS))]


def memory_plan(ntotal, dimension):
    if type(dimension) is not int or dimension < 1:
        raise ValueError('Index dimension must be a positive integer')
    tiles = tile_plan(ntotal)
    slot_bytes = VECTOR_ROWS * dimension * 4
    return dict(index_scan_bytes=ntotal*dimension*4, tiles=len(tiles),
        vector_rows=VECTOR_ROWS, query_micro=QUERY_MICRO,
        pinned_cpu_buffers_bytes=2*slot_bytes, device_buffers_bytes=2*slot_bytes,
        max_one_query_micro_distances_bytes=QUERY_MICRO*VECTOR_ROWS*4,
        limits='Buffer bytes exclude CUDA context, query/output tensors, allocator reserves and concurrent oracle/training')


class TorchCopyBackend:
    def __init__(self, host, device):
        self.host, self.device = host, torch.device(device)
        self.compute = torch.cuda.current_stream(self.device)
        self.copy = torch.cuda.Stream(device=self.device)
        self.cpu = [torch.empty((VECTOR_ROWS, host.shape[1]), dtype=torch.float32,
                                pin_memory=True) for _ in range(2)]
        self.gpu = [torch.empty((VECTOR_ROWS, host.shape[1]), dtype=torch.float32,
                                device=self.device) for _ in range(2)]
        self.ready = [torch.cuda.Event() for _ in range(2)]
        self.used = [torch.cuda.Event() for _ in range(2)]
        self.copy.wait_stream(self.compute)

    def sync_ready(self, slot):
        self.ready[slot].synchronize()  # Host memory cannot be rewritten during DMA.

    def wait_used(self, slot):
        self.copy.wait_event(self.used[slot])  # Device slot cannot overwrite pending GEMM.

    def stage(self, tile):
        n, slot = tile.stop-tile.start, tile.slot
        self.cpu[slot][:n].copy_(torch.from_numpy(self.host[tile.start:tile.stop]))
        with torch.cuda.stream(self.copy):
            self.gpu[slot][:n].copy_(self.cpu[slot][:n], non_blocking=True)
            self.ready[slot].record(self.copy)

    def acquire(self, tile):
        self.compute.wait_event(self.ready[tile.slot])
        return self.gpu[tile.slot][:tile.stop-tile.start]

    def consumed(self, tile):
        self.used[tile.slot].record(self.compute)

    def close(self):
        # Also on exceptions: preserve allocations until every DMA/GEMM finishes.
        self.compute.synchronize()
        self.copy.synchronize()


class DoubleBufferTiles:
    """The same lifecycle is used by the real CUDA backend and CPU event tests."""
    def __init__(self, backend):
        self.backend = backend
        self.staged = [None, None]
        self.used = [False, False]

    def stage(self, tile):
        slot = tile.slot
        if self.staged[slot] is not None:
            if not self.used[slot]:
                raise ValueError('Cannot reuse device slot before its consumer records completion')
            self.backend.sync_ready(slot)
            self.backend.wait_used(slot)
        self.backend.stage(tile)
        self.staged[slot], self.used[slot] = tile, False

    def acquire(self, tile):
        if self.staged[tile.slot] != tile or self.used[tile.slot]:
            raise ValueError('Wrong tile or repeated consumer acquisition')
        return self.backend.acquire(tile)

    def consumed(self, tile):
        if self.staged[tile.slot] != tile or self.used[tile.slot]:
            raise ValueError('Wrong tile or duplicate consumer completion')
        self.backend.consumed(tile)
        self.used[tile.slot] = True

    def close(self):
        self.backend.close()


class PinnedOverlapFlatIP:
    def __init__(self, original):
        self.original = original
        self.host, self.d, self.ntotal = original.host, original.d, original.ntotal
        self.device = torch.device(original.device)
        if (original.vector_rows != VECTOR_ROWS or original.query_micro != QUERY_MICRO or
                self.host.dtype != np.float32 or self.host.shape != (self.ntotal, self.d) or
                not self.host.flags.c_contiguous or self.device.type != 'cuda'):
            raise ValueError('Candidate requires the original contiguous FP32 32768/64 CUDA contract')

    @torch.no_grad()
    def search_many(self, list_queries, list_ks, trace=None):
        if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
            raise ValueError('Candidate requires TF32 disabled before admission')
        if len(list_queries) != len(list_ks) or len(list_queries) > 8:
            raise ValueError('Each of at most 8 requests requires its own k')
        requests, total_queries = [], 0
        for request_id, (queries, k) in enumerate(zip(list_queries, list_ks)):
            q = np.ascontiguousarray(queries, dtype=np.float32)
            if q.ndim != 2 or q.shape[1] != self.d or not np.isfinite(q).all():
                raise ValueError('Invalid query embeddings')
            if not 0 < int(k) <= self.ntotal or len(q) > 2560:
                raise ValueError('Invalid k or per-request query count')
            k = int(k); total_queries += len(q)
            if total_queries > 20480:
                raise ValueError('At most 20480 queries per shared scan')
            tq = torch.from_numpy(q).to(self.device) if len(q) else None
            best_s = torch.full((len(q), k), -float('inf'), dtype=torch.float32, device=self.device) if len(q) else None
            best_i = torch.full((len(q), k), -1, dtype=torch.int64, device=self.device) if len(q) else None
            requests.append((request_id, q, k, tq, best_s, best_i))
        if any(tq is not None for _, _, _, tq, _, _ in requests):
            plan = tile_plan(self.ntotal)
            buffers = DoubleBufferTiles(TorchCopyBackend(self.host, self.device))
            try:
                buffers.stage(plan[0])
                for number, tile in enumerate(plan):
                    # Prefetch is independent of the current tile's arithmetic.
                    if number+1 < len(plan):
                        buffers.stage(plan[number+1])
                    vectors = buffers.acquire(tile)
                    if trace is not None:
                        trace(dict(event='vector_transfer', start=tile.start, shape=list(vectors.shape)))
                    for request_id, q, k, tq, best_s, best_i in requests:
                        if tq is None: continue
                        local_k = min(k, len(vectors))
                        for offset in range(0, len(q), QUERY_MICRO):
                            stop = min(offset+QUERY_MICRO, len(q))
                            if trace is not None:
                                trace(dict(event='gemm', request=request_id, start=tile.start,
                                    offset=offset, left=[stop-offset, self.d], right=[self.d, len(vectors)], local_k=local_k, k=k))
                            distances = tq[offset:stop] @ vectors.T
                            scores, ids = torch.topk(distances, local_k, dim=1, sorted=True)
                            merged_s = torch.cat((best_s[offset:stop], scores), dim=1)
                            merged_i = torch.cat((best_i[offset:stop], ids+tile.start), dim=1)
                            values, choice = torch.topk(merged_s, k, dim=1, sorted=True)
                            best_s[offset:stop] = values
                            best_i[offset:stop] = torch.gather(merged_i, 1, choice)
                            del distances, scores, ids, merged_s, merged_i, values, choice
                    buffers.consumed(tile)
            finally:
                buffers.close()
        return [(np.empty((0, k), np.float32), np.empty((0, k), np.int64)) if tq is None
                else (best_s.cpu().numpy(), best_i.cpu().numpy())
                for _, q, k, tq, best_s, best_i in requests]
