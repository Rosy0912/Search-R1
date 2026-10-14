"""Private CPU FP32 E5/FlatIP service; no dependence on the retained E study."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time

def main():
    base=Path(sys.argv[1]).resolve()
    import faiss
    import torch
    import uvicorn
    from search_r1.search import retrieval_server as original
    torch.set_num_threads(48)
    faiss.omp_set_num_threads(48)
    a=json.loads((base/'receipts/assets.json').read_text())
    from .prepare_assets import sha
    assert sha(a['index'])=='69c98463fdb41fc08737d88513c597725f311c44f7ba6dca4b05d8c7c658d166'
    assert sha(a['corpus'])=='85a787a692e73fdd657e86411bfd1ac810ed193b1eea4a478de449d4871c06b9'
    original.config=original.Config(retrieval_method='e5',index_path=a['index'],corpus_path=a['corpus'],
        retrieval_topk=3,faiss_gpu=False,retrieval_model_path=a['encoder'],retrieval_pooling_method='mean',
        retrieval_query_max_length=256,retrieval_use_fp16=False,retrieval_batch_size=512,encoder_device='cpu')
    original.retriever=original.get_retriever(original.config)
    r=original.retriever
    assert r.index.d==768 and r.index.ntotal>20_000_000 and r.index.metric_type==faiss.METRIC_INNER_PRODUCT
    assert all(p.dtype==torch.float32 and p.device.type=='cpu' for p in r.encoder.model.parameters())
    query=original.QueryRequest(queries=['What is the capital of France?','Who wrote Pride and Prejudice?'],topk=3,return_scores=True)
    before=time.time()
    probe=asyncio.run(original.retrieve_endpoint(query))
    assert len(probe['result'])==2 and all(len(x)==3 for x in probe['result'])
    record=dict(at=time.time(),root=str(base),job=os.environ['SLURM_JOB_ID'],pid=os.getpid(),
        url='http://'+socket.gethostname().split('.')[0]+':29145',encoder_dtype='float32',index_dtype='float32',
        backend='CPU FAISS full FlatIP',vectors=r.index.ntotal,dimensions=r.index.d,
        probe_seconds=time.time()-before,probe_sha256=hashlib.sha256(json.dumps(probe,sort_keys=True).encode()).hexdigest())
    (base/'receipts/retriever-probe.json').write_text(json.dumps(probe,ensure_ascii=False))
    (base/'receipts/retriever-ready.json').write_text(json.dumps(record,indent=2))
    @original.app.get('/health')
    async def health():return record
    print('RETRIEVER_READY='+json.dumps(record),flush=True)
    uvicorn.run(original.app,host='0.0.0.0',port=29145,log_level='warning')

if __name__=='__main__':main()
