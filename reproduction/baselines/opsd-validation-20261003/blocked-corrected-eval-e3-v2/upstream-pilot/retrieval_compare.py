import hashlib, json, os, time
from pathlib import Path
import numpy as np
import torch
import faiss
from search_r1.search.retrieval_server import Encoder, load_corpus

root = Path(os.environ['SEARCH_R1_BASE'])
assets = Path(os.environ['SEARCH_R1_ASSETS']) / 'assets'
out = Path(os.environ['OPSD_EVIDENCE'])
threads = 8
faiss.omp_set_num_threads(threads)
torch.set_num_threads(threads)
queries = []
for source in ('nq', 'hotpotqa'):
    with (assets/source/'train.jsonl').open() as f:
        examples = [json.loads(line)['question'] for line in f]
    queries.extend([examples[0], examples[len(examples)//2], examples[-1]])
queries.extend(['What is the capital of France?', 'Which river flows through London?'])
start = time.perf_counter()
cpu_encoder = Encoder('e5', str(assets/'e5-base-v2'), 'mean', 256, True, device='cpu')
cpu_queries = cpu_encoder.encode(queries)
cpu_encoder_seconds = time.perf_counter() - start
assert np.isfinite(cpu_queries).all()
start = time.perf_counter()
gpu_encoder = Encoder('e5', str(assets/'e5-base-v2'), 'mean', 256, True, device='cuda')
gpu_queries = gpu_encoder.encode(queries)
torch.cuda.synchronize()
gpu_encoder_seconds = time.perf_counter() - start
assert np.isfinite(gpu_queries).all()
start = time.perf_counter()
cpu_index = faiss.read_index(str(assets/'wiki/e5_Flat.index'))
cpu_index_load_seconds = time.perf_counter() - start
start = time.perf_counter()
cs, ci = cpu_index.search(cpu_queries, 50)
cpu_search_seconds = time.perf_counter() - start
assert faiss.get_num_gpus() == 8
options = faiss.GpuMultipleClonerOptions()
options.shard = True
options.useFloat16 = True
start = time.perf_counter()
gpu_index = faiss.index_cpu_to_all_gpus(cpu_index, co=options)
gpu_clone_seconds = time.perf_counter() - start
start = time.perf_counter()
gs, gi = gpu_index.search(gpu_queries, 50)
gpu_search_seconds = time.perf_counter() - start
assert np.isfinite(cs).all() and np.isfinite(gs).all()
corpus = load_corpus(str(assets/'wiki/wiki-18.jsonl'))
def doc_hash(index):
    return hashlib.sha256(corpus[int(index)]['contents'].encode()).hexdigest()
content = {}
for k in (3, 50):
    cpu_doc = [[doc_hash(i) for i in row[:k]] for row in ci]
    gpu_doc = [[doc_hash(i) for i in row[:k]] for row in gi]
    content[str(k)] = dict(cpu=cpu_doc, gpu=gpu_doc,
        ordered_content_equal_rows=sum(a == b for a, b in zip(cpu_doc, gpu_doc)),
        id_overlap=sum(len(set(a[:k]) & set(b[:k])) for a, b in zip(ci, gi))/(len(queries)*k),
        content_overlap=sum(len(set(a) & set(b)) for a, b in zip(cpu_doc, gpu_doc))/max(1,sum(len(set(a)) for a in cpu_doc)))
result = dict(queries=queries, query_source='fixed train rows and canonical queries; no final-test selection',
    cpu_encoder='same immutable E5 fp16, 8 CPU threads', original_gpu='E5 CUDA fp16 plus 8-GPU sharded fp16 FAISS',
    index_vectors=cpu_index.ntotal, index_dimension=cpu_index.d, cpu_ids=ci.tolist(), gpu_ids=gi.tolist(),
    cpu_scores=cs.tolist(), gpu_scores=gs.tolist(), content=content,
    embedding_max_abs_delta=float(np.max(np.abs(cpu_queries-gpu_queries))),
    sorted_score_max_abs_delta=float(np.max(np.abs(cs-gs))),
    cpu_encoder_seconds=cpu_encoder_seconds, gpu_encoder_seconds=gpu_encoder_seconds,
    cpu_index_load_seconds=cpu_index_load_seconds, cpu_search_seconds=cpu_search_seconds,
    gpu_clone_seconds=gpu_clone_seconds, gpu_search_seconds=gpu_search_seconds)
(out/'retrieval-comparison.json').write_text(json.dumps(result, indent=2))
print(json.dumps({k:v for k,v in result.items() if k not in ('cpu_ids','gpu_ids','cpu_scores','gpu_scores','content')}, sort_keys=True), flush=True)
print('RETRIEVAL_OVERLAP', {k:v['id_overlap'] for k,v in content.items()}, flush=True)
assert content['3']['id_overlap'] >= .95 and content['50']['id_overlap'] >= .95, 'Inspect real retrieval disagreement before training'
assert result['sorted_score_max_abs_delta'] < .01, 'Inspect numeric retrieval disagreement before training'
print('REAL_CPU_GPU_RETRIEVAL_COMPARISON_OK', flush=True)
