"""CPU operator fixture only. This is never a full-index/live-GPU acceptance.

When FAISS is absent locally, a minimal pointer fixture supplies only the flat
array interface consumed by the unchanged original scanner. It is recorded as
such; no synthetic input is substituted into live experiments.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
import torch
from operator_trace import RecordMM


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    controls = Path(__file__).resolve().parents[2]
    original_path = controls / "torch_streaming_flat.py"
    fixture_faiss = False
    try:
        import faiss
    except ImportError:
        fixture_faiss = True
        faiss = types.ModuleType("faiss")
        faiss.METRIC_INNER_PRODUCT = 0
        faiss.rev_swig_ptr = lambda array, size: array.reshape(-1)
        sys.modules["faiss"] = faiss
    spec = importlib.util.spec_from_file_location("torch_streaming_flat", original_path)
    original_module = importlib.util.module_from_spec(spec)
    sys.modules["torch_streaming_flat"] = original_module
    spec.loader.exec_module(original_module)
    from segmented_scan import SegmentedStreamingFlatIP
    torch.set_num_threads(16)
    torch.set_float32_matmul_precision("highest")
    sizes = [1, 63, 64, 65, 128, 129, 320]
    dimension, ntotal = 16, 32771
    generator = np.random.default_rng(4207)
    checks = []
    for scenario in ["random_with_duplicate_vectors", "all_scores_tied"]:
        host = generator.normal(size=(ntotal, dimension)).astype(np.float32)
        if scenario == "all_scores_tied":
            host[:] = 1
        else:
            host[-3:] = host[:3]
        if fixture_faiss:
            index = types.SimpleNamespace(ntotal=ntotal, d=dimension,
                                          metric_type=faiss.METRIC_INNER_PRODUCT,
                                          get_xb=lambda: host)
        else:
            index = faiss.IndexFlatIP(dimension)
            index.add(host)
        original = original_module.TorchStreamingFlatIP(index, "cpu")
        candidate = SegmentedStreamingFlatIP(index, "cpu")
        queries = [generator.normal(size=(size, dimension)).astype(np.float32) for size in sizes]
        for parity in [0, 1]:
            ks = [3 + ((i + parity) % 2) for i in range(len(sizes))]
            expected, original_shapes = [], []
            for q, k in zip(queries, ks):
                with RecordMM() as trace:
                    expected.append(original.search(q, k))
                original_shapes.append(trace.shapes)
            declared = []
            with RecordMM() as candidate_trace:
                actual = candidate.search_many(queries, ks, trace=declared.append)
            declared_mm = [event for event in declared if event["event"] == "gemm"]
            actual_shapes = [[e["left"], e["right"]] for e in declared_mm]
            if actual_shapes != candidate_trace.shapes:
                raise AssertionError("Declared shape trace does not match actual aten.mm")
            transfers = [e for e in declared if e["event"] == "vector_transfer"]
            if [e["shape"] for e in transfers] != [[32768, dimension], [3, dimension]]:
                raise AssertionError("A vector tile was transferred more than once")
            for i, (size, k, (es, ei), (scores, ids)) in enumerate(zip(sizes, ks, expected, actual)):
                request_shapes = [[e["left"], e["right"]] for e in declared_mm if e["request"] == i]
                if request_shapes != original_shapes[i]:
                    raise AssertionError("Per-request original GEMM boundary changed")
                if not np.array_equal(es, scores) or not np.array_equal(ei, ids):
                    raise AssertionError("Scores or IDs changed, including tie ordering")
                checks.append(dict(scenario=scenario, queries=size, topk=k,
                                   scores_exact=True, ids_exact=True, actual_gemm_shapes=request_shapes))
    empty = np.empty((0, dimension), dtype=np.float32)
    actual_empty = candidate.search_many([empty], [4])
    assert actual_empty[0][0].shape == actual_empty[0][1].shape == (0, 4)
    # Queue checks exercise FIFO admission and the eight-request cap separately
    # from scanner operator equality.
    from request_queue import RequestQueue, PendingRequest
    queue = RequestQueue(collection_seconds=0)
    submitted = [PendingRequest([str(i)] * 2560, 3, True) for i in range(9)]
    for request in submitted:
        queue.submit(request)
    first, second = queue.next_batch(), queue.next_batch()
    assert first == submitted[:8] and second == submitted[8:]
    queue.close("fixture complete")
    failed_queue = RequestQueue(collection_seconds=0)
    failed_request = PendingRequest(["queued failure"], 3, True)
    failed_queue.submit(failed_request)
    failed_queue.close("fixture injected worker failure")
    assert failed_request.future.done() and isinstance(failed_request.future.exception(), RuntimeError)
    rejected = False
    try:
        failed_queue.submit(PendingRequest(["late"], 3, True))
    except RuntimeError:
        rejected = True
    assert rejected
    output = dict(accepted_cpu_fixture=True, full_gpu_accepted=False,
                  scope="CPU fixture operator/shapes/IDs/ties only; no live acceptance",
                  original_sha256=sha(original_path), candidate_sha256=sha(Path(__file__).with_name("segmented_scan.py")),
                  torch=torch.__version__, numpy=np.__version__, faiss_pointer_fixture=fixture_faiss,
                  ntotal=ntotal, d=dimension, checks=checks,
                  queue_fifo=True, max_requests=8, max_queries=20480, empty_request=True,
                  queued_failure_future_completed=True, closed_queue_rejects=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(json.dumps({"cpu_checks": len(checks), "exact": True, "full_gpu_accepted": False,
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
