"""Compare exact GPU-sharded top-3 against the existing CPU FAISS index."""
import argparse
import json
import time
import numpy as np
import faiss
import torch
from search_e1.gpu_index import ShardedFlatIP

p=argparse.ArgumentParser()
p.add_argument('--index',required=True)
p.add_argument('--output',required=True)
p.add_argument('--queries',type=int,default=32)
a=p.parse_args()
faiss.omp_set_num_threads(4)
index=faiss.read_index(a.index)
queries=np.random.default_rng(42).standard_normal((a.queries,index.d)).astype('float32')
faiss.normalize_L2(queries)
begin=time.monotonic()
ref_scores,ref_indices=index.search(queries,3)
cpu_seconds=time.monotonic()-begin
print({'cpu_seconds':cpu_seconds},flush=True)
gpu=ShardedFlatIP(index)
begin=time.monotonic()
scores,indices=gpu.search(queries,3)
gpu_seconds=time.monotonic()-begin
if not np.array_equal(indices,ref_indices):
    raise RuntimeError('GPU top-3 differs from CPU FAISS')
np.testing.assert_allclose(scores,ref_scores,rtol=1e-4,atol=1e-5)
result={'queries':a.queries,'cpu_seconds':cpu_seconds,'gpu_seconds':gpu_seconds,
        'speedup':cpu_seconds/gpu_seconds,'top3_identical':True,
        'max_score_error':float(np.max(np.abs(scores-ref_scores))),
        'gpu_count':torch.cuda.device_count(),'precision':'float32, TF32 disabled'}
with open(a.output,'w') as f:json.dump(result,f,indent=2)
print(json.dumps(result),flush=True)
