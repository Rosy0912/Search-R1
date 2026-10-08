import argparse
import uvicorn
import numpy as np
import json
import time
from search_r1.search import retrieval_server as server

p = argparse.ArgumentParser()
p.add_argument('--index', required=True)
p.add_argument('--corpus', required=True)
p.add_argument('--model', required=True)
p.add_argument('--port', type=int, default=18000)
p.add_argument('--backend', choices=['cpu', 'gpu'], default='cpu')
p.add_argument('--verify-data')
p.add_argument('--verification-output')
a = p.parse_args()
server.config = server.Config(retrieval_method='e5', retrieval_topk=3, index_path=a.index,
                              corpus_path=a.corpus, faiss_gpu=False, retrieval_model_path=a.model,
                              retrieval_use_fp16=True, retrieval_batch_size=128)
server.retriever = server.get_retriever(server.config)
if a.backend == 'gpu':
    from search_e1.gpu_index import ShardedFlatIP
    cpu_index = server.retriever.index
    gpu_index = ShardedFlatIP(cpu_index)
    if a.verify_data:
        from search_e1.io import read_jsonl, write_json
        questions = [r['question'] for r in read_jsonl(a.verify_data)[:8]]
        embeddings = server.retriever.encoder.encode(questions)
        begin = time.monotonic()
        cpu_scores, cpu_ids = cpu_index.search(embeddings, 3)
        cpu_seconds = time.monotonic() - begin
        begin = time.monotonic()
        gpu_scores, gpu_ids = gpu_index.search(embeddings, 3)
        gpu_seconds = time.monotonic() - begin
        if not np.array_equal(cpu_ids, gpu_ids):
            raise RuntimeError('GPU top-3 differs on real-question verification')
        np.testing.assert_allclose(gpu_scores, cpu_scores, rtol=1e-4, atol=1e-5)
        report = dict(queries=len(questions), top3_identical=True, cpu_seconds=cpu_seconds,
                      gpu_seconds=gpu_seconds, max_score_error=float(np.max(np.abs(gpu_scores-cpu_scores))))
        print(json.dumps(report), flush=True)
        if a.verification_output:
            write_json(a.verification_output, report)
    server.retriever.index = gpu_index
    del cpu_index
uvicorn.run(server.app, host='127.0.0.1', port=a.port)
