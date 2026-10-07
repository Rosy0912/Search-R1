"""Share vector transfers without sharing or changing request GEMM boundaries.

The authoritative operators are TorchStreamingFlatIP.search.  search_many changes
only the loop interleaving: vector tile -> request -> that request's original
query micro-batches.  It does not concatenate request embeddings.
"""
import numpy as np
import torch

from torch_streaming_flat import TorchStreamingFlatIP


class SegmentedStreamingFlatIP(TorchStreamingFlatIP):
    @torch.no_grad()
    def search_many(self, list_queries, list_ks, trace=None):
        if len(list_queries) != len(list_ks):
            raise ValueError("Each request must have a corresponding top-k")
        if len(list_queries) > 8:
            raise ValueError("At most 8 requests per shared vector scan")
        requests = []
        total_queries = 0
        for request_id, (queries, k) in enumerate(zip(list_queries, list_ks)):
            q = np.ascontiguousarray(queries, dtype=np.float32)
            if q.ndim != 2 or q.shape[1] != self.d or not np.isfinite(q).all():
                raise ValueError("Invalid query embeddings")
            if not 0 < int(k) <= self.ntotal:
                raise ValueError("Invalid top-k")
            k = int(k)
            if len(q) > 2560:
                raise ValueError("At most 2560 queries per request")
            total_queries += len(q)
            if total_queries > 20480:
                raise ValueError("At most 20480 queries per shared vector scan")
            if len(q) == 0:
                requests.append((request_id, q, k, None, None, None))
                continue
            tq = torch.from_numpy(q).to(self.device)
            best_s = torch.full((len(q), k), -float("inf"), dtype=torch.float32, device=self.device)
            best_i = torch.full((len(q), k), -1, dtype=torch.int64, device=self.device)
            requests.append((request_id, q, k, tq, best_s, best_i))

        # Empty requests have precisely the original search output and need no scan.
        if any(tq is not None for _, _, _, tq, _, _ in requests):
            for start in range(0, self.ntotal, self.vector_rows):
                vectors = torch.from_numpy(self.host[start:start + self.vector_rows]).to(self.device)
                if trace is not None:
                    trace(dict(event="vector_transfer", start=start, shape=list(vectors.shape)))
                for request_id, q, k, tq, best_s, best_i in requests:
                    if tq is None:
                        continue
                    local_k = min(k, len(vectors))
                    for offset in range(0, len(q), self.query_micro):
                        stop = min(offset + self.query_micro, len(q))
                        if trace is not None:
                            trace(dict(event="gemm", request=request_id, start=start,
                                       offset=offset, left=[stop-offset, self.d],
                                       right=[self.d, len(vectors)], local_k=local_k, k=k))
                        # Keep these operators and their per-request shapes exactly
                        # equal to TorchStreamingFlatIP.search, including tile tails.
                        distances = tq[offset:stop] @ vectors.T
                        scores, ids = torch.topk(distances, local_k, dim=1, sorted=True)
                        merged_s = torch.cat((best_s[offset:stop], scores), dim=1)
                        merged_i = torch.cat((best_i[offset:stop], ids + start), dim=1)
                        values, choice = torch.topk(merged_s, k, dim=1, sorted=True)
                        best_s[offset:stop] = values
                        best_i[offset:stop] = torch.gather(merged_i, 1, choice)
                        del distances, scores, ids, merged_s, merged_i, values, choice
                del vectors

        results = []
        for _, q, k, tq, best_s, best_i in requests:
            if tq is None:
                results.append((np.empty((0, k), np.float32), np.empty((0, k), np.int64)))
            else:
                results.append((best_s.cpu().numpy(), best_i.cpu().numpy()))
        return results

