"""Prepared optional idle-GPU7 kernel comparison; never starts a service.

Default 8 tiles is a SMALL kernel/profile, not full-index retrieval acceptance.
A full scan still requires fresh resource admission and actual score/ID gates.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
import subprocess
import numpy as np
import torch
from pinned_overlap import PinnedOverlapFlatIP, memory_plan, VECTOR_ROWS


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--oracle-source', required=True)
    parser.add_argument('--index', required=True)
    parser.add_argument('--profile-tiles', type=int, default=8)
    parser.add_argument('--full-scan', action='store_true')
    parser.add_argument('--query-embeddings', help='Optional remote FP32 .npy, not raw text')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--authorize-idle-gpu7-bench', action='store_true')
    args = parser.parse_args()
    if not args.authorize_idle_gpu7_bench or os.environ.get('CUDA_VISIBLE_DEVICES') != '7':
        raise ValueError('Requires explicit idle-GPU7 admission and CUDA_VISIBLE_DEVICES=7')
    if args.profile_tiles < 1 or not 1 <= args.repeats <= 5:
        raise ValueError('Invalid bounded benchmark size')
    gpu_rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free',
        '--format=csv,noheader,nounits'], text=True).splitlines()
    target = next([part.strip() for part in row.split(',')] for row in gpu_rows
                  if row.split(',')[0].strip() == '7')
    if int(target[2]) < 20480:
        raise ValueError('GPU7 direct free is below idle admission; do not benchmark during training')
    owners = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'], text=True).splitlines()
    target_pids = [int(row.split(',')[1]) for row in owners if row.split(',')[0].strip() == target[1]]
    if any(pid != 2239529 for pid in target_pids):
        raise ValueError('GPU7 has another compute process; HOLD rather than benchmark')
    source = Path(args.oracle_source).resolve()
    sys.path.insert(0, str(source))
    import faiss
    original_module = importlib.import_module('torch_streaming_flat')
    segmented_module = importlib.import_module('segmented_scan')
    source_identity = {p: sha(source/p) for p in ('torch_streaming_flat.py', 'segmented_scan.py')}
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Loads one independent CPU index. Full flat index is about 64.6 GB host RAM;
    # --profile-tiles reduces scanning, NOT the CPU allocation of faiss.read_index.
    index = faiss.read_index(args.index)
    original = segmented_module.SegmentedStreamingFlatIP(index, 'cuda:0', vector_rows=32768, query_micro=64)
    actual_ntotal = original.ntotal
    if not args.full_scan:
        original.ntotal = min(original.ntotal, (args.profile_tiles-1)*VECTOR_ROWS+7)
        original.host = original.host[:original.ntotal]
    candidate = PinnedOverlapFlatIP(original)
    if args.query_embeddings:
        inputs = np.load(args.query_embeddings, allow_pickle=False)
        if inputs.dtype != np.float32 or inputs.ndim != 2 or inputs.shape[1] != original.d:
            raise ValueError('Expected exact original FP32 query matrix')
    else:
        inputs = np.array(original.host[:256], dtype=np.float32, copy=True)
    if len(inputs) < 128:
        raise ValueError('At least 128 query embeddings required for selected micro tails')
    cohorts = ([inputs[:5]], [inputs[:19]], [inputs[:40]], [inputs[:65]],
               [inputs[:5], inputs[5:24], inputs[24:64]])
    results = []
    for requests in cohorts:
        ks = [3]*len(requests)
        oracle_trace, candidate_trace = [], []
        baseline = original.search_many(requests, ks, trace=oracle_trace.append)
        tested = candidate.search_many(requests, ks, trace=candidate_trace.append)
        if oracle_trace != candidate_trace:
            raise ValueError('Request/GEMM/tile/top-k operation geometry/order differs')
        if not all(np.array_equal(s, ts) and np.array_equal(i, ti)
                   for (s,i),(ts,ti) in zip(baseline, tested)):
            raise ValueError('Exact FP32 scores or IDs differ')
        timings = {}
        for name, adapter in (('oracle', original), ('candidate', candidate)):
            wall = []
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start_free, _ = torch.cuda.mem_get_info()
            for _ in range(args.repeats):
                begin = time.perf_counter()
                adapter.search_many(requests, ks)
                torch.cuda.synchronize()
                wall.append(time.perf_counter()-begin)
            end_free, _ = torch.cuda.mem_get_info()
            timings[name] = dict(wall_seconds=wall, direct_free_before=start_free,
                direct_free_after=end_free, torch_peak_allocated=torch.cuda.max_memory_allocated(),
                torch_peak_reserved=torch.cuda.max_memory_reserved())
        results.append(dict(query_counts=[len(q) for q in requests], score_id_exact=True,
            operation_geometry_exact=True, timings=timings))
    print(json.dumps(dict(scope='FULL_INDEX_KERNEL_SCORE_ID_GATE' if args.full_scan else 'SMALL_TILE_PROFILE_ONLY',
        original_source_sha256=source_identity, candidate_sha256=sha(Path(__file__).with_name('pinned_overlap.py')),
        actual_index_rows=actual_ntotal, scanned_rows=original.ntotal,
        model_encoding_or_http_gate_performed=False, query_source='provided_fp32' if args.query_embeddings else 'index_vectors_fixture',
        memory=memory_plan(original.ntotal, original.d), results=results), sort_keys=True), flush=True)


if __name__ == '__main__': run()
