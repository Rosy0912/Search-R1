"""Exact FP32 Flat-IP scan with bounded device memory and full host vectors."""
import numpy as np
import torch
import faiss


class TorchStreamingFlatIP:
    def __init__(self, index, device, vector_rows=32768, query_micro=64):
        if index.metric_type != faiss.METRIC_INNER_PRODUCT or not hasattr(index, 'get_xb'):
            raise ValueError('Only exact flat inner-product indices are supported')
        if not 0 < vector_rows <= 32768 or not 0 < query_micro <= 64:
            raise ValueError('Invalid bounded-memory scan dimensions')
        self.ntotal, self.d = int(index.ntotal), int(index.d)
        self.metric_type = faiss.METRIC_INNER_PRODUCT
        self.cpu_index = index
        self.host = faiss.rev_swig_ptr(index.get_xb(), self.ntotal * self.d).reshape(self.ntotal, self.d)
        self.device = torch.device(device)
        self.vector_rows, self.query_micro = vector_rows, query_micro
        self.bounds = [(0, self.ntotal)]

    @torch.no_grad()
    def search(self, queries, k):
        q = np.ascontiguousarray(queries, dtype=np.float32)
        if q.ndim != 2 or q.shape[1] != self.d or not np.isfinite(q).all():
            raise ValueError('Invalid query embeddings')
        if not 0 < int(k) <= self.ntotal:
            raise ValueError('Invalid top-k')
        k = int(k)
        if len(q) == 0:
            return np.empty((0, k), np.float32), np.empty((0, k), np.int64)
        tq = torch.from_numpy(q).to(self.device)
        best_s = torch.full((len(q), k), -float('inf'), dtype=torch.float32, device=self.device)
        best_i = torch.full((len(q), k), -1, dtype=torch.int64, device=self.device)
        for start in range(0, self.ntotal, self.vector_rows):
            vectors = torch.from_numpy(self.host[start:start + self.vector_rows]).to(self.device)
            local_k = min(k, len(vectors))
            for offset in range(0, len(q), self.query_micro):
                stop = min(offset + self.query_micro, len(q))
                distances = tq[offset:stop] @ vectors.T
                scores, ids = torch.topk(distances, local_k, dim=1, sorted=True)
                merged_s = torch.cat((best_s[offset:stop], scores), dim=1)
                merged_i = torch.cat((best_i[offset:stop], ids + start), dim=1)
                values, choice = torch.topk(merged_s, k, dim=1, sorted=True)
                best_s[offset:stop] = values
                best_i[offset:stop] = torch.gather(merged_i, 1, choice)
                del distances, scores, ids, merged_s, merged_i, values, choice
            del vectors
        return best_s.cpu().numpy(), best_i.cpu().numpy()
