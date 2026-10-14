"""Independent four-GPU FP32 scan candidate; never changes a live endpoint."""
import concurrent.futures
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
    from search_r1.search.retrieval_server import Encoder
    base=Path(sys.argv[1]);out=base/'receipts'/('gpu-retrieval-benchmark-'+os.environ['SLURM_JOB_ID'])
    out.mkdir(exist_ok=False)
    assert torch.cuda.device_count()==4
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    torch.set_num_threads(16);faiss.omp_set_num_threads(32)
    assets=json.loads((base/'receipts/assets.json').read_text())
    index_path=Path(assets['index']);before=index_path.stat()
    start=time.perf_counter();index=faiss.read_index(str(index_path))
    assert index.ntotal==21015324 and index.d==768 and index.metric_type==faiss.METRIC_INNER_PRODUCT
    vectors=faiss.rev_swig_ptr(index.get_xb(),index.ntotal*index.d).reshape(index.ntotal,index.d)
    # Preserve the published32768-vector tile shape and FP32 query micro64.
    tiles=[(i,min(i+32768,index.ntotal)) for i in range(0,index.ntotal,32768)]
    groups=[tiles[len(tiles)*i//4:len(tiles)*(i+1)//4] for i in range(4)]
    gpu_vectors=[]
    for gpu,group in enumerate(groups):
        first,last=group[0][0],group[-1][1]
        gpu_vectors.append(torch.from_numpy(vectors[first:last]).to('cuda:'+str(gpu)))
        torch.cuda.synchronize(gpu)
    loaded=time.perf_counter()-start
    encoder=Encoder(model_name='e5',model_path=assets['encoder'],pooling_method='mean',
                    max_length=256,use_fp16=False,device='cpu')
    raw=base/'assets/raw/nq/train.jsonl'
    with raw.open() as f:queries=[json.loads(next(f))['question'] for _ in range(64)]
    t=time.perf_counter();emb=encoder.encode(queries);encoding=time.perf_counter()-t
    t=time.perf_counter();cpu_s,cpu_i=index.search(emb,3);cpu_seconds=time.perf_counter()-t
    np.savez(out/'oracle.npz',embeddings=emb,scores=cpu_s,ids=cpu_i)
    def scan(gpu):
        with torch.cuda.device(gpu),torch.no_grad():
            q=torch.from_numpy(emb).to('cuda:'+str(gpu));group=groups[gpu];origin=group[0][0]
            best_s=torch.full((len(q),3),float('-inf'),device=q.device)
            best_i=torch.full((len(q),3),-1,dtype=torch.long,device=q.device)
            for first,last in group:
                scores,ids=torch.topk(q@gpu_vectors[gpu][first-origin:last-origin].T,3,dim=1)
                joined_s=torch.cat((best_s,scores),dim=1)
                joined_i=torch.cat((best_i,ids+first),dim=1)
                best_s,choice=torch.topk(joined_s,3,dim=1)
                best_i=torch.gather(joined_i,1,choice)
            torch.cuda.synchronize(gpu)
            return best_s.cpu(),best_i.cpu()
    t=time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:parts=list(pool.map(scan,range(4)))
    scores=torch.cat([x[0] for x in parts],dim=1);ids=torch.cat([x[1] for x in parts],dim=1)
    values,choice=torch.topk(scores,3,dim=1);selected=torch.gather(ids,1,choice)
    gpu_seconds=time.perf_counter()-t
    actual_s,actual_i=values.numpy(),selected.numpy()
    equal=bool(np.array_equal(actual_i,cpu_i));delta=float(np.abs(actual_s-cpu_s).max())
    after=index_path.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    result=dict(at=time.time(),job=os.environ['SLURM_JOB_ID'],source=os.environ['RECURSIVE_SOURCE_COMMIT'],
        hardware=[torch.cuda.get_device_name(i) for i in range(4)],queries=64,index_rows=index.ntotal,
        query_sha256=hashlib.sha256(json.dumps(queries).encode()).hexdigest(),
        index_path=str(index_path),index_size=before.st_size,index_mtime_ns=before.st_mtime_ns,
        identical_top3_ids=equal,maximum_score_error=delta,accepted=equal and delta<=2e-5,
        load_seconds=loaded,CPU_encoder_seconds=encoding,CPU_scan_seconds=cpu_seconds,GPU_scan_seconds=gpu_seconds,
        peak_allocated_bytes=[torch.cuda.max_memory_allocated(i) for i in range(4)],
        production_endpoint_changed=False,scope='64 fixed TRAIN queries; numerical/performance candidate only, not proof of universal equality or deployed training acceleration')
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    if not result['accepted']:raise RuntimeError('Candidate differs from fixed CPU oracle; do not admit')

if __name__=='__main__':main()
