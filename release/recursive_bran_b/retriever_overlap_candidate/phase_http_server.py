"""Private CPU E5/corpus parent with a reusable, explicitly selected GPU child.

The original K4 oracle is untouched. Only a hash-bound numerical gate enables
training requests; probe requests and release require the private control token.
"""
import argparse
import hashlib
import hmac
import json
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--assets',required=True);p.add_argument('--asset-proof',required=True)
    p.add_argument('--base',required=True);p.add_argument('--token-file',required=True)
    p.add_argument('--gpu-count',type=int,choices=(4,8),required=True)
    p.add_argument('--host',default='10.123.4.12');p.add_argument('--port',type=int,default=18006)
    a=p.parse_args();base=Path(a.base);(base/'receipts').mkdir(parents=True,exist_ok=True)
    assets=Path(a.assets);token=Path(a.token_file).read_text().strip()
    assert a.gpu_count==4 and os.environ.get('CUDA_VISIBLE_DEVICES')=='0,1,2,3', 'Current HKUST5 authorization is four cards only'
    import numpy as np
    import torch
    import faiss
    from fastapi import FastAPI,HTTPException,Header
    import uvicorn
    from search_r1.search import retrieval_server as original
    from flat_memmap import file_metadata
    from gpu_phase_proxy import PhaseGpuScanner,process_identity,source_manifest
    torch.set_num_threads(16);faiss.omp_set_num_threads(32)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    proof=json.loads(Path(a.asset_proof).read_text())
    assert proof['index']['sha256']=='69c98463fdb41fc08737d88513c597725f311c44f7ba6dca4b05d8c7c658d166'
    for name,key in [('index/e5_Flat.index','index'),('corpus/wiki-18.jsonl','corpus')]:
        assert file_metadata(assets/name)==proof[key]['metadata'], 'Transferred asset changed after checksum verification'
    encoder=original.Encoder('e5',str(assets/'encoder'),'mean',256,False,device='cpu')
    corpus=original.load_corpus(str(assets/'corpus/wiki-18.jsonl'))
    assert len(corpus)==21015324
    scan=PhaseGpuScanner(SimpleNamespace(d=768,ntotal=21015324,vector_rows=32768,query_micro=64),
        str(assets/'index/e5_Flat.index'),gpu_count=a.gpu_count)
    assert not torch.cuda.is_initialized(), 'CPU parent must never create a CUDA context'
    implementation=source_manifest()
    implementation.update(server=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        encoder_source=hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest())
    identity=dict(process_identity(),host='xie-lab-4',gpu_shards=a.gpu_count,
        implementation=implementation,asset_proof=proof,cpu_encoder='FP32 E5 mean max256 batch128')
    identity_sha=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    record=dict(identity,source_identity_sha256=identity_sha,cuda_initialized=False)
    (base/'receipts/source-identity.json').write_text(json.dumps(record,indent=2))
    lock=threading.Lock();app=FastAPI()
    def private(value):
        if not hmac.compare_digest(value,token):raise HTTPException(403,'Invalid private control token')
    def admitted():
        try:
            binding=json.loads((base/'receipts/admitted.json').read_text())
            path=Path(binding['gate_path']);raw=path.read_bytes();gate=json.loads(raw)
            return (hashlib.sha256(raw).hexdigest()==binding['gate_sha256'] and gate['accepted'] is True
                and gate['candidate_pid']==os.getpid() and gate['candidate_starttick']==identity['starttick']
                and gate['source_identity_sha256']==identity_sha and gate['gpu_shards']==a.gpu_count
                and gate['scope']=='four_gpu_full_scan_and_http_oracle_equality')
        except (OSError,KeyError,ValueError):return False
    def retrieve(request):
        k=request.topk or 3
        if not 1<=k<=4 or len(request.queries)>2560:raise HTTPException(400,'topk 1..4; at most 2560 queries')
        if not request.queries:return {'result':[]}
        with lock:
            t=time.perf_counter()
            # Preserve each original batch128 encoder call boundary.
            pieces=[encoder.encode(request.queries[i:i+128]) for i in range(0,len(request.queries),128)]
            embeddings=np.concatenate(pieces,axis=0);encoded=time.perf_counter()
            scores,ids=scan.search(embeddings,k);searched=time.perf_counter()
            docs=original.load_docs(corpus,ids.reshape(-1).tolist())
            rows=[docs[i:i+k] for i in range(0,len(docs),k)]
            result=[[dict(document=doc,score=score) for doc,score in zip(row,ss)]
                if request.return_scores else row for row,ss in zip(rows,scores.tolist())]
            print('RETRIEVER_TIMING='+json.dumps(dict(queries=len(request.queries),encoder_seconds=encoded-t,
                scan_seconds=searched-encoded,documents_seconds=time.perf_counter()-searched,
                gpu_worker_pid=scan.worker_pid,gpu_shards=a.gpu_count)),flush=True)
            return {'result':result}
    @app.get('/health')
    def health():
        return dict(ready=True,pid=os.getpid(),starttick=identity['starttick'],ntotal=scan.ntotal,d=scan.d,
            gpu_shards=a.gpu_count,tf32=False,query_micro=64,encoder='CPU FP32 E5',
            source_identity_sha256=identity_sha,admitted=admitted(),gpu_worker_pid=scan.worker_pid,
            last_scan_profile=scan.last_scan_profile,cpu_parent_cuda_initialized=torch.cuda.is_initialized())
    @app.post('/probe/retrieve')
    def probe(request:original.QueryRequest,x_e1_probe_token:str=Header(default='')):
        private(x_e1_probe_token);return retrieve(request)
    @app.post('/retrieve')
    def training(request:original.QueryRequest):
        if not admitted():raise HTTPException(503,'Exact full-index numerical gate not admitted')
        return retrieve(request)
    @app.post('/release-gpu')
    def release(x_e1_probe_token:str=Header(default='')):
        private(x_e1_probe_token)
        with lock:
            result=scan.release_gpu()
            (base/'receipts/last-gpu-release.json').write_text(json.dumps(result,indent=2))
            return result
    print('PHASE_CPU_READY='+json.dumps(record),flush=True)
    try:uvicorn.run(app,host=a.host,port=a.port)
    finally:scan.close()


if __name__=='__main__':main()
