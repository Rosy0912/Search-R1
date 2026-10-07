"""Unadmitted candidate: independent CPU encodes, shared segmented GPU scans.

CPU loading is the only startup work.  A process- and source-bound activation
file enables GPU probing; a separate accepted gate receipt enables /retrieve.
This file does not switch any relay or existing training endpoint.
"""
import os
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
os.environ["E1_ENCODER_DEVICE"] = "cpu"

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import secrets
import threading
import time
import traceback

import faiss
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Header, HTTPException
from search_r1.search import retrieval_server as original_server
import torch_streaming_flat
from segmented_scan import SegmentedStreamingFlatIP
from request_queue import PendingRequest, RequestQueue
from operator_trace import RecordMM
from contextlib import nullcontext


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


class Candidate:
    def __init__(self, args):
        self.args = args
        self.base = Path(args.base)
        (self.base / "receipts").mkdir(parents=True, exist_ok=True)
        (self.base / "logs").mkdir(parents=True, exist_ok=True)
        self.timings = self.base / "receipts/requests.jsonl"
        self.events = self.base / "receipts/events.jsonl"
        self.activation = self.base / "receipts/enable-gpu.json"
        self.admission = self.base / "receipts/admit-requests.json"
        self.queue = RequestQueue(args.collection_seconds, args.max_pending)
        self.cpu_ready = False
        self.gpu_ready = False
        self.scan = None
        self.activation_error = None
        self.probe_token = os.environ.get("E1_SEGMENTED_PROBE_TOKEN", "")
        if len(self.probe_token) < 32:
            raise ValueError("A private probe token of at least 32 characters is required")

        torch.set_num_threads(16)
        faiss.omp_set_num_threads(32)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        # The existing execution runtime provides the unchanged CPU placement
        # overlay. Refuse an upstream GPU-only encoder before constructing it.
        if "E1_ENCODER_DEVICE" not in inspect.getsource(original_server.load_model):
            raise RuntimeError("Use the preserved CPU encoder execution runtime")
        if "v.to(next(self.model.parameters()).device)" not in inspect.getsource(original_server.Encoder.encode):
            raise RuntimeError("Encoder input placement is not the verified CPU overlay")

        assets = Path(args.assets)
        config = original_server.Config(
            retrieval_method="e5", retrieval_topk=3,
            index_path=str(assets / "index/e5_Flat.index"),
            corpus_path=str(assets / "corpus/wiki-18.jsonl"),
            faiss_gpu=False, retrieval_model_path=str(assets / "encoder"),
            retrieval_use_fp16=False, retrieval_batch_size=128,
        )
        self.retriever = original_server.get_retriever(config)
        cpu_index = self.retriever.index
        if cpu_index.ntotal != 21015324 or cpu_index.d != 768 or len(self.retriever.corpus) != cpu_index.ntotal:
            raise RuntimeError("Incomplete or unexpected full retrieval index/corpus")
        if any(p.device.type != "cpu" or p.dtype != torch.float32 for p in self.retriever.encoder.model.parameters()):
            raise RuntimeError("Encoder must stay entirely CPU FP32")
        if torch.cuda.is_initialized():
            raise RuntimeError("CPU startup unexpectedly initialized CUDA")
        self.actual_threads_after_cpu_load = dict(torch=int(torch.get_num_threads()),
            torch_interop=int(torch.get_num_interop_threads()), faiss=int(faiss.omp_get_max_threads()))
        asset_manifest = Path(args.asset_manifest)
        identity = dict(
            assets_manifest=dict(path=str(asset_manifest), sha256=digest(asset_manifest)),
            sources={str(p): digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
            original_scan=dict(path=torch_streaming_flat.__file__, sha256=digest(torch_streaming_flat.__file__)),
            original_encoder=dict(path=original_server.__file__, sha256=digest(original_server.__file__)),
            ntotal=int(cpu_index.ntotal), d=int(cpu_index.d),
            encoder="original CPU FP32 E5", encoder_batch_size=128,
            encoder_threads=16, vector_rows=32768, query_micro=64,
            configured_threads=dict(torch=16, faiss=32),
            actual_threads_after_cpu_load=self.actual_threads_after_cpu_load,
            max_requests=8, max_queries_per_request=2560, max_queries_per_scan=20480,
            tf32=False, collection_seconds=args.collection_seconds,
            index_file=self.file_identity(assets / "index/e5_Flat.index"),
            corpus_file=self.file_identity(assets / "corpus/wiki-18.jsonl"),
        )
        self.identity_sha256 = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.identity = identity
        atomic_json(self.base / "receipts/source-identity.json", dict(
            pid=os.getpid(), source_identity_sha256=self.identity_sha256, identity=identity,
            cuda_initialized=False, cpu_loaded_unadmitted=True,
        ))
        self.cpu_ready = True
        threading.Thread(target=self.activate_and_run, name="single-segmented-scanner", daemon=True).start()

    @staticmethod
    def file_identity(path):
        st = path.stat()
        return dict(path=str(path), realpath=str(path.resolve()), size=st.st_size,
                    mtime_ns=st.st_mtime_ns, inode=st.st_ino, device=st.st_dev)

    def record_event(self, event):
        with self.events.open("a") as handle:
            handle.write(json.dumps(dict(time=time.time(), pid=os.getpid(), **event)) + "\n")

    def read_bound_control(self, path):
        try:
            control = json.loads(path.read_text())
        except (OSError, ValueError):
            return None
        if control.get("pid") != os.getpid() or control.get("source_identity_sha256") != self.identity_sha256:
            return None
        return control

    def admitted(self):
        control = self.read_bound_control(self.admission)
        if not self.gpu_ready or not control or control.get("accepted") is not True:
            return False
        # Bind admission to the reviewed full live gate, rather than a boolean
        # activation switch. Root reviews/writes this receipt after comparison.
        try:
            gate_path = Path(control["full_gpu_http_gate_path"])
            gate = json.loads(gate_path.read_text())
            return (digest(gate_path) == control["full_gpu_http_gate_sha256"]
                    and gate.get("accepted") is True
                    and gate.get("candidate_pid") == os.getpid()
                    and gate.get("source_identity_sha256") == self.identity_sha256
                    and gate.get("scope") == "full GPU and HTTP oracle equality")
        except (KeyError, OSError, ValueError):
            return False

    def health(self):
        return dict(ready=self.admitted(), cpu_ready=self.cpu_ready, gpu_ready=self.gpu_ready,
                    pid=os.getpid(), source_identity_sha256=self.identity_sha256,
                    ntotal=21015324, d=768, index="full FP32 segmented streaming FlatIP",
                    encoder="original CPU FP32 E5", encoder_batch_size=128,
                    query_micro=64, vector_rows=32768, tf32=False,
                    max_requests=8, max_queries_per_scan=20480,
                    collection_seconds=self.args.collection_seconds,
                    actual_threads_after_cpu_load=self.actual_threads_after_cpu_load,
                    pending_requests=len(self.queue.pending), activation_error=self.activation_error)

    def activate_and_run(self):
        try:
            while self.read_bound_control(self.activation) is None:
                time.sleep(0.5)
            if torch.cuda.device_count() != 1:
                raise RuntimeError("Expose only the approved physical GPU 7")
            self.scan = SegmentedStreamingFlatIP(self.retriever.index, "cuda:0", vector_rows=32768, query_micro=64)
            self.gpu_ready = True
            self.record_event(dict(event="gpu_probe_enabled", cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES")))
            while True:
                batch = self.queue.next_batch()
                if not batch:
                    return
                self.process_batch(batch)
        except BaseException as error:
            self.activation_error = repr(error)
            self.gpu_ready = False
            self.queue.close("Segmented scanner failed: " + repr(error))
            self.record_event(dict(event="scanner_failed", error=repr(error), traceback=traceback.format_exc()))

    def process_batch(self, batch):
        dequeued = time.monotonic()
        valid, embeddings, timings = [], [], {}
        for request in batch:
            start = time.monotonic()
            try:
                actual_encode_threads = dict(torch=int(torch.get_num_threads()),
                                             faiss=int(faiss.omp_get_max_threads()))
                # Requests are intentionally encoded independently. Each original
                # request retains its own 128-query E5 padding/batch boundaries.
                pieces = [self.retriever.encoder.encode(request.queries[offset:offset + 128])
                          for offset in range(0, len(request.queries), 128)]
                embedded = np.concatenate(pieces, axis=0)
                if embedded.dtype != np.float32 or embedded.shape != (len(request.queries), 768):
                    raise RuntimeError("Unexpected encoder output")
                valid.append(request)
                embeddings.append(embedded)
                timings[request.request_id] = dict(encode_seconds=time.monotonic()-start,
                    encode_started_seconds=start-request.enqueued,
                    actual_encoder_worker_threads=actual_encode_threads)
            except Exception as error:
                request.future.set_exception(RuntimeError("CPU encoding failed: " + repr(error)))
                self.record_event(dict(event="request_failed", request_id=request.request_id,
                                       stage="encode", error=repr(error)))
        if not valid:
            return
        scan_started = time.monotonic()
        compare_original = any(r.compare_original for r in valid)
        declared_shapes = []
        try:
            shape_recorder = RecordMM() if compare_original else None
            with shape_recorder if shape_recorder is not None else nullcontext():
                result_arrays = self.scan.search_many(embeddings, [r.k for r in valid],
                    trace=declared_shapes.append if compare_original else None)
            if compare_original:
                actual_shapes = [[e["left"], e["right"]] for e in declared_shapes if e["event"] == "gemm"]
                if actual_shapes != shape_recorder.shapes:
                    raise RuntimeError("Declared scan shapes do not match actual aten.mm")
            if len(result_arrays) != len(valid):
                raise RuntimeError("Segmented scan omitted a request")
            for request, (scores, ids) in zip(valid, result_arrays):
                if scores.shape != (len(request.queries), request.k) or ids.shape != scores.shape:
                    raise RuntimeError("Segmented scan returned an invalid result shape")
            scan_seconds = time.monotonic() - scan_started
        except Exception as error:
            for request in valid:
                request.future.set_exception(RuntimeError("Full segmented scan failed: " + repr(error)))
                self.record_event(dict(event="request_failed", request_id=request.request_id,
                                       stage="scan", error=repr(error)))
            return
        for request, (scores, ids) in zip(valid, result_arrays):
            assembly_started = time.monotonic()
            try:
                comparison = None
                if request.compare_original:
                    index = valid.index(request)
                    with RecordMM() as oracle_trace:
                        oracle_scores, oracle_ids = self.scan.search(embeddings[index], request.k)
                    candidate_shapes = [[e["left"], e["right"]] for e in declared_shapes
                                        if e["event"] == "gemm" and e["request"] == index]
                    comparison = dict(scores_exact=bool(np.array_equal(scores, oracle_scores)),
                                      ids_exact=bool(np.array_equal(ids, oracle_ids)),
                                      shape_exact=candidate_shapes == oracle_trace.shapes,
                                      actual_gemm_shapes=candidate_shapes,
                                      vector_tiles=sum(e["event"] == "vector_transfer" for e in declared_shapes),
                                      batch_requests=len(valid), batch_queries=sum(len(r.queries) for r in valid),
                                      full_ntotal=int(self.scan.ntotal), source_identity_sha256=self.identity_sha256,
                                      candidate_pid=os.getpid(),
                                      original_scan_sha256=digest(torch_streaming_flat.__file__))
                    if not all(comparison[key] for key in ["scores_exact", "ids_exact", "shape_exact"]):
                        raise RuntimeError("Full GPU original scanner equality failed: " + json.dumps(comparison))
                documents = original_server.load_docs(self.retriever.corpus, ids.reshape(-1).tolist())
                results = [documents[offset:offset+request.k] for offset in range(0, len(documents), request.k)]
                out = [[{"document": document, "score": score} for document, score in zip(docs, ss)]
                       if request.return_scores else docs for docs, ss in zip(results, scores.tolist())]
                timing = dict(time=time.time(), request_id=request.request_id, queries=len(request.queries),
                              topk=request.k, return_scores=request.return_scores,
                              batch_requests=len(valid), batch_queries=sum(len(r.queries) for r in valid),
                              queue_wait_seconds=dequeued-request.enqueued,
                              scan_started_seconds=scan_started-request.enqueued,
                              shared_scan_seconds=scan_seconds,
                              assemble_seconds=time.monotonic()-assembly_started,
                              total_seconds=time.monotonic()-request.enqueued,
                              **timings[request.request_id])
                with self.timings.open("a") as handle:
                    handle.write(json.dumps(timing) + "\n")
                response = {"result": out}
                if comparison is not None:
                    response["oracle_comparison"] = comparison
                request.future.set_result(response)
            except Exception as error:
                request.future.set_exception(RuntimeError("Document assembly failed: " + repr(error)))
                self.record_event(dict(event="request_failed", request_id=request.request_id,
                                       stage="documents", error=repr(error)))

    def retrieve(self, request, probe_token=None, compare_original=False):
        if request.topk is not None and not 1 <= request.topk <= 4:
            raise HTTPException(400, "topk must be 1..4")
        if len(request.queries) > 2560:
            raise HTTPException(400, "At most 2560 queries per request")
        if probe_token is not None:
            if not secrets.compare_digest(probe_token, self.probe_token):
                raise HTTPException(403, "Invalid private probe token")
            if not self.gpu_ready:
                raise HTTPException(503, "GPU probing has not been activated")
        elif not self.admitted():
            raise HTTPException(503, "Candidate awaits full GPU/HTTP admission")
        if not request.queries:
            return {"result": []}
        pending = PendingRequest(list(request.queries), request.topk or 3, request.return_scores,
                                 compare_original=compare_original)
        try:
            return self.queue.submit(pending).result()
        except Exception as error:
            raise HTTPException(503, "Retrieval failed: " + repr(error)) from error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=os.environ.get("E1_BASE"), required="E1_BASE" not in os.environ)
    parser.add_argument("--assets", default="/data/shaoshitong/search-e1-retriever-gpu-20261004/assets")
    parser.add_argument("--asset-manifest", required=True)
    parser.add_argument("--host", default="10.120.17.101")
    parser.add_argument("--port", type=int, default=18004)
    parser.add_argument("--collection-seconds", type=float, default=0.025)
    parser.add_argument("--max-pending", type=int, default=128)
    args = parser.parse_args()
    candidate = Candidate(args)
    app = FastAPI()

    @app.get("/health")
    def health():
        return candidate.health()

    @app.post("/retrieve")
    def retrieve(request: original_server.QueryRequest):
        return candidate.retrieve(request)

    @app.post("/probe/retrieve")
    def probe(request: original_server.QueryRequest, x_e1_probe_token: str = Header(default="")):
        return candidate.retrieve(request, probe_token=x_e1_probe_token)

    @app.post("/probe/compare")
    def compare(request: original_server.QueryRequest, x_e1_probe_token: str = Header(default="")):
        return candidate.retrieve(request, probe_token=x_e1_probe_token, compare_original=True)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
