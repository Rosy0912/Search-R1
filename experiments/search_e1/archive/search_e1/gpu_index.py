"""Exact float32 inner-product search, sharded across allocated CUDA devices."""
import numpy as np
import torch

class ShardedFlatIP:
    def __init__(self, index, devices=None, chunk_size=262144):
        import faiss
        if index.metric_type != faiss.METRIC_INNER_PRODUCT:
            raise ValueError('This backend requires an inner-product index')
        if not isinstance(index, faiss.IndexFlat):
            raise ValueError('This backend requires a flat index')
        self.d, self.ntotal = index.d, index.ntotal
        self.chunk_size = chunk_size
        devices = devices or list(range(torch.cuda.device_count()))
        if not devices:
            raise ValueError('No allocated CUDA devices')
        # Do not change FP32 precision through TF32.
        torch.backends.cuda.matmul.allow_tf32 = False
        self.shards = []
        for i, device in enumerate(devices):
            start = self.ntotal * i // len(devices)
            end = self.ntotal * (i + 1) // len(devices)
            vectors = index.reconstruct_n(start, end-start)
            tensor = torch.from_numpy(vectors).to(torch.device('cuda', device))
            self.shards.append((start, tensor))
            print({'index_gpu': device, 'vectors': end-start, 'bytes': tensor.numel()*tensor.element_size()}, flush=True)

    @torch.no_grad()
    def search(self, queries, k):
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        if k <= 0 or k > self.ntotal:
            raise ValueError('Invalid top-k')
        candidates = []
        for start, vectors in self.shards:
            q = torch.from_numpy(queries).to(vectors.device)
            scores = torch.full((len(q), k), -torch.inf, device=vectors.device)
            indices = torch.full((len(q), k), -1, device=vectors.device, dtype=torch.long)
            for offset in range(0, len(vectors), self.chunk_size):
                block = vectors[offset:offset+self.chunk_size]
                values, positions = torch.topk(q @ block.T, min(k,len(block)), dim=1)
                positions = positions + offset + start
                combined_scores = torch.cat((scores, values), dim=1)
                combined_indices = torch.cat((indices, positions), dim=1)
                scores, selected = torch.topk(combined_scores, k, dim=1)
                indices = combined_indices.gather(1, selected)
            candidates.append((scores.cpu(), indices.cpu()))
        scores = torch.cat([v[0] for v in candidates],dim=1)
        indices = torch.cat([v[1] for v in candidates],dim=1)
        scores, selected = scores.topk(k,dim=1)
        return scores.numpy(), indices.gather(1, selected).numpy()
