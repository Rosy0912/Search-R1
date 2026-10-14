"""Independently deployed, numerically checked GPU scanner; no live rerouting."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import sys
import time

def main():
    import faiss
    import numpy as np
    import torch
    import uvicorn
    from search_r1.search import retrieval_server as native
    from .gpu_index import FourGPUFlatIP
    base=Path(sys.argv[1]);job=os.environ['SLURM_JOB_ID']
    directory=base/'receipts'/('gpu-retriever-'+job);directory.mkdir(exist_ok=False)
    assets=json.loads((base/'receipts/assets.json').read_text())
    benchmark=json.loads((base/'receipts/gpu-retrieval-benchmark-138640/result.json').read_text())
    index_stat=Path(assets['index']).stat()
    assert benchmark['accepted'] and benchmark['index_path']==assets['index']
    assert (index_stat.st_size,index_stat.st_mtime_ns)==(benchmark['index_size'],benchmark['index_mtime_ns'])
    torch.set_num_threads(48);faiss.omp_set_num_threads(48)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    native.config=native.Config(retrieval_method='e5',index_path=assets['index'],corpus_path=assets['corpus'],
        retrieval_topk=3,faiss_gpu=False,retrieval_model_path=assets['encoder'],retrieval_pooling_method='mean',
        retrieval_query_max_length=256,retrieval_use_fp16=False,retrieval_batch_size=512,encoder_device='cpu')
    start=time.perf_counter();native.retriever=native.get_retriever(native.config)
    r=native.retriever;oracle=r.index;scan=FourGPUFlatIP(oracle)
    assert scan.ntotal==21015324 and scan.d==768
    assert all(p.dtype==torch.float32 and p.device.type=='cpu' for p in r.encoder.model.parameters())
    with (base/'assets/raw/nq/train.jsonl').open() as f:queries=[json.loads(next(f))['question'] for _ in range(64)]
    emb=r.encoder.encode(queries)
    t=time.perf_counter();expected_s,expected_i=oracle.search(emb,3);cpu_seconds=time.perf_counter()-t
    t=time.perf_counter();actual_s,actual_i=scan.search(emb,3);gpu_seconds=time.perf_counter()-t
    assert np.array_equal(expected_i,actual_i),'Candidate top3 differs from unchanged CPU oracle'
    delta=float(np.max(np.abs(expected_s-actual_s)));assert delta<=2e-5
    # Exercise the unchanged full HTTP-format implementation, not just vector scores.
    probe=native.QueryRequest(queries=queries[:4],topk=3,return_scores=True)
    cpu_payload=asyncio.run(native.retrieve_endpoint(probe))
    r.index=scan
    gpu_payload=asyncio.run(native.retrieve_endpoint(probe))
    assert [[x['document'] for x in row] for row in cpu_payload['result']]==[[x['document'] for x in row] for row in gpu_payload['result']]
    assert max(abs(x['score']-y['score']) for a,b in zip(cpu_payload['result'],gpu_payload['result']) for x,y in zip(a,b))<=2e-5
    record=dict(at=time.time(),root=str(base),job=job,pid=os.getpid(),url='http://'+os.environ['SLURM_JOB_NODELIST']+':29147',
        source=os.environ['RECURSIVE_SOURCE_COMMIT'],encoder_dtype='float32',index_dtype='float32',
        backend='four-A100 resident FP32 full FlatIP; tiles32768 querymicro64;TF32off',vectors=scan.ntotal,dimensions=scan.d,
        validation=dict(queries=64,top3_identical=True,max_score_error=delta,http_documents_identical=True,
            CPU_scan_seconds=cpu_seconds,GPU_scan_seconds=gpu_seconds,universal_equivalence_claim=False),
        init_seconds=time.perf_counter()-start,live_training_rerouted=False)
    (directory/'validation.json').write_text(json.dumps(record,indent=2))
    (directory/'http-probe.json').write_text(json.dumps(gpu_payload,ensure_ascii=False))
    ready=base/'receipts/gpu-retriever-ready.json'
    with ready.open('x') as f:json.dump(record,f,indent=2)
    @native.app.get('/health')
    async def health():return record
    @native.app.middleware('http')
    async def timing(request,call_next):
        before=time.perf_counter();response=await call_next(request)
        if request.url.path=='/retrieve':
            with (directory/'requests.jsonl').open('a') as f:
                f.write(json.dumps(dict(at=time.time(),seconds=time.perf_counter()-before,status=response.status_code))+'\n')
        return response
    print('GPU_RETRIEVER_READY='+json.dumps(record),flush=True)
    uvicorn.run(native.app,host='0.0.0.0',port=29147,log_level='warning')

if __name__=='__main__':main()
