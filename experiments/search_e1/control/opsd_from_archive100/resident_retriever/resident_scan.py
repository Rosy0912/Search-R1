"""Full FP32 resident tiles; preserve original tile and top-k merge order.

Four devices hold contiguous ranges of whole original tiles. A single scanner
visits those ranges in global order. At a shard boundary it copies the running
FP32 scores and int64 IDs without arithmetic. There is no shard-local ranking
or extra final top-k. Cross-device numerical equality requires the live gate.
"""
from contextlib import nullcontext
import numpy as np
import torch
from torch_streaming_flat import TorchStreamingFlatIP


def tile_plan(ntotal, vector_rows=32768, devices=4):
    if ntotal <= 0 or vector_rows <= 0 or devices != 4:
        raise ValueError("Positive full index and exactly four shard devices required")
    tiles = (ntotal + vector_rows - 1) // vector_rows
    if tiles < devices:
        raise ValueError("At least one original tile per device required")
    quotient, remainder = divmod(tiles, devices)
    plan, first = [], 0
    for rank in range(devices):
        count = quotient + (rank < remainder)
        stop = first + count
        plan.append(dict(device_rank=rank, tile_first=first, tile_stop=stop,
                         row_first=first*vector_rows,
                         row_stop=min(stop*vector_rows, ntotal)))
        first = stop
    assert first == tiles
    return plan


class ResidentTiledFlatIP(TorchStreamingFlatIP):
    def __init__(self, index, devices, vector_rows=32768, query_micro=64):
        self.devices = [torch.device(d) for d in devices]
        if len(self.devices) != 4:
            raise ValueError("Exactly four resident shard devices required")
        if any(d.type == "cuda" for d in self.devices):
            if len(set(self.devices)) != 4 or any(d.type != "cuda" for d in self.devices):
                raise ValueError("Production uses four distinct CUDA devices")
        elif any(d.type != "cpu" for d in self.devices):
            raise ValueError("Only CUDA production or CPU-only fixture devices supported")
        super().__init__(index, self.devices[0], vector_rows, query_micro)
        self.plan = tile_plan(self.ntotal, vector_rows)
        self.resident = []
        for shard, device in zip(self.plan, self.devices):
            vectors = torch.from_numpy(self.host[shard["row_first"]:shard["row_stop"]]).to(device)
            self.resident.append(dict(**shard, device=str(device), vectors=vectors))

    def residency(self):
        return [dict(**{k:v for k,v in s.items() if k != "vectors"},
                     vector_bytes=s["vectors"].numel()*s["vectors"].element_size(),
                     vector_shape=list(s["vectors"].shape)) for s in self.resident]

    @torch.no_grad()
    def search_many(self, list_queries, list_ks, trace=None):
        if len(list_queries) != len(list_ks) or len(list_queries) > 8:
            raise ValueError("At most eight requests with corresponding top-k required")
        requests, total_queries = [], 0
        for request_id, (queries, k) in enumerate(zip(list_queries, list_ks)):
            q = np.ascontiguousarray(queries, dtype=np.float32)
            if q.ndim != 2 or q.shape[1] != self.d or not np.isfinite(q).all():
                raise ValueError("Invalid query embeddings")
            if not 0 < int(k) <= self.ntotal:
                raise ValueError("Invalid top-k")
            k = int(k)
            total_queries += len(q)
            if len(q) > 2560 or total_queries > 20480:
                raise ValueError("Original per-request and aggregate query limits exceeded")
            if len(q) == 0:
                requests.append([request_id, q, k, None, None, None])
                continue
            tq = torch.from_numpy(q).to(self.devices[0])
            best_s = torch.full((len(q), k), -float("inf"), dtype=torch.float32, device=self.devices[0])
            best_i = torch.full((len(q), k), -1, dtype=torch.int64, device=self.devices[0])
            requests.append([request_id, q, k, tq, best_s, best_i])

        if any(r[3] is not None for r in requests):
            for shard in self.resident:
                device = torch.device(shard["device"])
                context = torch.cuda.device(device) if device.type == "cuda" else nullcontext()
                with context:
                    for request in requests:
                        if request[3] is None or request[3].device == device:
                            continue
                        # Device copies preserve bits and ranking state; no new
                        # local top-k or differently shaped shard merge is added.
                        request[3] = request[3].to(device)
                        request[4] = request[4].to(device)
                        request[5] = request[5].to(device)
                        if trace is not None:
                            trace(dict(event="ranking_state_copy", request=request[0],
                                       at_global_tile=shard["tile_first"], device=str(device)))
                    for start in range(shard["row_first"], shard["row_stop"], self.vector_rows):
                        rows = min(self.vector_rows, self.ntotal-start)
                        local = start-shard["row_first"]
                        vectors = shard["vectors"][local:local+rows]
                        if trace is not None:
                            trace(dict(event="resident_vector_tile", start=start,
                                       shape=list(vectors.shape), device=str(device)))
                        for request_id, q, k, tq, best_s, best_i in requests:
                            if tq is None:
                                continue
                            local_k = min(k, len(vectors))
                            for offset in range(0, len(q), self.query_micro):
                                stop = min(offset+self.query_micro, len(q))
                                if trace is not None:
                                    trace(dict(event="gemm", request=request_id, start=start,
                                               offset=offset, left=[stop-offset, self.d],
                                               right=[self.d, len(vectors)], local_k=local_k,
                                               k=k, device=str(device)))
                                # These arithmetic operators and their order are
                                # identical to the original streaming scanner.
                                distances = tq[offset:stop] @ vectors.T
                                scores, ids = torch.topk(distances, local_k, dim=1, sorted=True)
                                merged_s = torch.cat((best_s[offset:stop], scores), dim=1)
                                merged_i = torch.cat((best_i[offset:stop], ids + start), dim=1)
                                values, choice = torch.topk(merged_s, k, dim=1, sorted=True)
                                best_s[offset:stop] = values
                                best_i[offset:stop] = torch.gather(merged_i, 1, choice)
                                del distances, scores, ids, merged_s, merged_i, values, choice
        results = []
        for _, q, k, tq, best_s, best_i in requests:
            if tq is None:
                results.append((np.empty((0,k),np.float32),np.empty((0,k),np.int64)))
            else:
                results.append((best_s.cpu().numpy(),best_i.cpu().numpy()))
        return results
