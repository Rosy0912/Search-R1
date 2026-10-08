"""Benchmark exact FAISS top-3 CPU searches without changing the active service."""
import argparse
import json
import time
import numpy as np
import faiss

p = argparse.ArgumentParser()
p.add_argument('--index', required=True)
p.add_argument('--output', required=True)
p.add_argument('--threads', type=int, nargs='+', default=[4, 16])
p.add_argument('--queries', type=int, default=128)
a = p.parse_args()
index = faiss.read_index(a.index)
rng = np.random.default_rng(42)
queries = rng.standard_normal((a.queries, index.d)).astype('float32')
faiss.normalize_L2(queries)
reference = None
results = []
for threads in a.threads:
    faiss.omp_set_num_threads(threads)
    begin = time.monotonic()
    distances, indices = index.search(queries, 3)
    elapsed = time.monotonic() - begin
    if reference is None:
        reference = (distances.copy(), indices.copy())
    else:
        if not np.array_equal(indices, reference[1]):
            raise RuntimeError('Top-3 indices changed across thread settings')
        np.testing.assert_allclose(distances, reference[0], rtol=1e-5, atol=1e-6)
    row = {'threads': threads, 'queries': a.queries, 'seconds': elapsed, 'top3_consistent': True}
    results.append(row)
    print(json.dumps(row), flush=True)
with open(a.output, 'w') as f:
    json.dump(results, f, indent=2)
