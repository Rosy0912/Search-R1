"""Original eight-shard FP16 retriever with real CPU/GPU parity and owned cache.

No retrieval mathematics or document order changes. The CPU index is retained
only for the initial independent comparison, then released before training.
"""
import hashlib
import importlib.util
import json
import os
import re
import threading
import time
import sys
import gc
from pathlib import Path
import numpy as np

from reproduction.common8.configuration import REPO
from reproduction.common8.stage_retrieval import ASSETS, binding, memory_bytes, atomic
from reproduction.common8.verify_data import digest
from reproduction.common8.validation_identity import expected
from reproduction.common8.temporary_cache import create as create_cache, admit_capacity


def compare_search(queries, cpu_scores, cpu_ids, gpu_scores, gpu_ids, cpu_vectors, gpu_vectors, metric):
    assert metric in ('inner_product', 'l2')
    count, dim = queries.shape
    assert count == 64 and dim == 768
    assert cpu_scores.shape == cpu_ids.shape == gpu_scores.shape == gpu_ids.shape == (64, 3)
    assert cpu_vectors.shape == gpu_vectors.shape == (64, 3, 768)
    assert all(np.isfinite(v).all() for v in (queries, cpu_scores, gpu_scores, cpu_vectors, gpu_vectors))
    assert (cpu_ids >= 0).all() and (gpu_ids >= 0).all()
    assert all(len(set(row)) == 3 for row in cpu_ids)
    assert all(len(set(row)) == 3 for row in gpu_ids)
    q = queries.astype(np.float64)[:, None, :]
    def exact(v):
        v = v.astype(np.float64)
        return (q * v).sum(-1) if metric == 'inner_product' else ((q - v)**2).sum(-1)
    exact_cpu, exact_gpu = exact(cpu_vectors), exact(gpu_vectors)
    cpu_error = float(np.abs(exact_cpu - cpu_scores).max())
    gpu_error = float(np.abs(exact_gpu - gpu_scores).max())
    recall = sum(len(set(a) & set(b)) for a, b in zip(cpu_ids, gpu_ids)) / (64 * 3)
    # Preserve disclosed FP16 quantization, while rejecting wrong/missing shards.
    assert cpu_error <= 1e-4, 'CPU index scores do not match its original reconstructed vectors'
    assert gpu_error <= 0.02, 'GPU scores differ materially from original vectors'
    assert recall >= 0.95, 'GPU top3 differs materially from original full CPU index'
    gap = cpu_scores[:, -1:] - exact_gpu if metric == 'inner_product' else exact_gpu - cpu_scores[:, -1:]
    assert float(gap.max()) <= 2 * gpu_error + 1e-5, 'GPU selected documents outside observed quantization tolerance'
    return dict(metric=metric, top3_set_recall=recall, exact_rank_agreement=float((cpu_ids == gpu_ids).mean()),
                cpu_vector_score_max_abs=cpu_error, gpu_vector_score_max_abs=gpu_error,
                selected_score_gap_max=float(gap.max()), score_bound=0.02, recall_bound=0.95)


def main(base):
    import datasets
    import faiss
    import torch
    import uvicorn
    from reproduction.common8.native_allocation import cuda_uuids
    base = Path(base).resolve()
    run, job, allocation = binding(base)
    assert memory_bytes(job['TRES']) >= 512 * 2**30
    assert os.environ['CUDA_VISIBLE_DEVICES'].split(',') == allocation['visible_devices']
    assert cuda_uuids() == allocation['actual_cuda_runtime_uuids']
    assert faiss.get_num_gpus() == torch.cuda.device_count() == 8
    assert Path(faiss.__file__).resolve().is_relative_to((base / 'retriever-gpu-lib').resolve())
    for name in ('reproduction/common8/encoder-assets.json', 'reproduction/common8/retriever.py',
                 'search_r1/search/retrieval_server.py'):
        assert digest(REPO / name) == run['source_sha256'][name]
    staged = json.loads((base / 'logs/retrieval-assets.json').read_text())
    assert staged['accepted'] is True and staged['run_id'] == run['run_id']
    assert staged['slurm_job_id'] == job['JobId'] and set(staged['files']) == set(ASSETS)
    claim = json.loads((base / 'logs/retrieval-staging-claim.json').read_text())
    for name, item in ASSETS.items():
        path = Path(staged['files'][name]['path'])
        assert path == Path(claim['root']) / name and not path.is_symlink()
        assert path.stat().st_size == item['size'] and digest(path) == item['sha256']
    assets = json.loads((REPO / 'reproduction/common8/encoder-assets.json').read_text())
    encoder = base / 'assets/e5-base-v2'
    for name, item in assets['files'].items():
        assert (encoder / name).stat().st_size == item['size'] and digest(encoder / name) == item['sha256']
    cache = create_cache(base, run, job)
    spec = importlib.util.spec_from_file_location('common8_upstream_retriever', REPO / 'search_r1/search/retrieval_server.py')
    upstream = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = upstream; spec.loader.exec_module(upstream)
    cpu_indices = []
    original_read = faiss.read_index
    def read(path, *args, **kwargs):
        assert Path(path) == Path(staged['files']['e5_Flat.index']['path'])
        index = original_read(path, *args, **kwargs); cpu_indices.append(index); return index
    def corpus(path):
        assert Path(path) == Path(staged['files']['wiki-18.jsonl']['path'])
        admit_capacity(base, cache)
        return datasets.load_dataset('json', data_files=path, split='train', num_proc=4, cache_dir=str(cache))
    upstream.load_corpus = corpus
    upstream.config = upstream.Config(retrieval_method='e5', retrieval_topk=3,
        index_path=staged['files']['e5_Flat.index']['path'], corpus_path=staged['files']['wiki-18.jsonl']['path'],
        faiss_gpu=True, retrieval_model_path=str(encoder), retrieval_pooling_method='mean',
        retrieval_query_max_length=256, retrieval_use_fp16=True, retrieval_batch_size=512)
    faiss.read_index = read
    try: upstream.retriever = upstream.get_retriever(upstream.config)
    finally: faiss.read_index = original_read
    assert len(cpu_indices) == 1
    cpu = cpu_indices[0]; retriever = upstream.retriever
    assert cpu.d == retriever.index.d == 768
    assert cpu.ntotal == retriever.index.ntotal == len(retriever.corpus)
    assert retriever.index.count() == 8, 'GPU index is not the original eight-shard layout'
    record, frame, ids = expected(base, 'stage')
    assert record['sha256'] == run['input_sha256']['stage.parquet']
    queries = []
    for chat in frame['prompt']:
        content = chat[0]['content']; marks = list(re.finditer(r'Question:\s*', content))
        assert marks; queries.append(content[marks[-1].end():].strip())
    vectors = retriever.encoder.encode(queries)
    assert vectors.dtype == np.float32 and vectors.shape == (64, 768)
    cpu_scores, cpu_ids = cpu.search(vectors, 3)
    gpu_scores, gpu_ids = retriever.index.search(vectors, 3)
    assert (gpu_ids < cpu.ntotal).all() and (cpu_ids < cpu.ntotal).all()
    cpu_vectors = cpu.reconstruct_batch(cpu_ids.reshape(-1)).reshape(64, 3, 768)
    gpu_vectors = cpu.reconstruct_batch(gpu_ids.reshape(-1)).reshape(64, 3, 768)
    metric = {faiss.METRIC_INNER_PRODUCT: 'inner_product', faiss.METRIC_L2: 'l2'}[cpu.metric_type]
    parity = compare_search(vectors, cpu_scores, cpu_ids, gpu_scores, gpu_ids, cpu_vectors, gpu_vectors, metric)
    documents = [[hashlib.sha256(json.dumps(retriever.corpus[int(i)], sort_keys=True,
                  ensure_ascii=False, separators=(',', ':')).encode()).hexdigest() for i in row] for row in gpu_ids]
    proof = dict(accepted=True, run_id=run['run_id'], slurm_job_id=job['JobId'], index_sha256=ASSETS['e5_Flat.index']['sha256'],
        corpus_sha256=ASSETS['wiki-18.jsonl']['sha256'], encoder_assets=assets, ntotal=int(cpu.ntotal), dimension=768,
        shard_count=8, use_float16=True, query_prefix='query: ', pooling='mean', encoder_dtype='float16',
        question_ids=ids, stage_sha256=record['sha256'], gpu_document_ids=gpu_ids.tolist(),
        gpu_document_sha256=documents, parity=parity, cache=str(cache), checked_unix=time.time(),
        scope='actual original CPU/GPU index and document-order comparison; HTTP readiness and training coexistence separate')
    target = base / 'logs/retrieval-admission.json'
    assert not target.exists()
    index_bytes = int(cpu.ntotal) * int(cpu.d) * 4
    def rss():
        return int(Path('/proc/self/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')
    before = rss()
    cpu_indices.clear(); del cpu  # Release the retained 64GB CPU baseline before model training.
    gc.collect()
    after = rss()
    assert before - after >= index_bytes * 0.8, 'The full CPU baseline index remains resident'
    proof['cpu_index_release'] = dict(before_rss_bytes=before, after_rss_bytes=after,
        expected_vector_bytes=index_bytes, observed_released_bytes=before-after)
    atomic(target, proof)
    initialized_thread = threading.get_ident()
    original_batch = retriever.batch_search
    def serial_batch(*args, **kwargs):
        assert threading.get_ident() == initialized_thread, 'FAISS resources moved off their initialization thread'
        return original_batch(*args, **kwargs)
    retriever.batch_search = serial_batch
    @upstream.app.get('/health')
    async def health():
        return dict(ok=True, run_id=run['run_id'], slurm_job_id=job['JobId'], shard_count=8,
                    admission_sha256=digest(target), ntotal=proof['ntotal'])
    uvicorn.run(upstream.app, host='127.0.0.1', port=18010, workers=1)


if __name__ == '__main__': main(os.environ['SEARCH_G1_BASE'])
