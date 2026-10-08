"""Isolated real-index check of the upstream sharded FP16 GPU FAISS path."""
import json
import runpy
import time
from pathlib import Path
import torch
import faiss
import numpy as np
BASE=Path('/data/search-r1')
assert faiss.get_num_gpus()==8
Encoder=runpy.run_path(str(BASE/'repo/search_r1/search/retrieval_server.py'))['Encoder']
encoder=Encoder('e5',str(BASE/'assets/e5-base-v2'),'mean',256,False)
queries=[]
for source in ['nq','triviaqa','popqa','hotpotqa','2wikimultihopqa','musique','bamboogle']:
    folder=BASE/'assets'/source
    file=folder/('test.jsonl' if (folder/'test.jsonl').exists() else 'dev.jsonl')
    examples=[json.loads(line)['question'] for line in file.read_text().splitlines()]
    queries.extend([examples[0],examples[len(examples)//2]])
q=encoder.encode(queries)
cpu=faiss.read_index(str(BASE/'assets/wiki/e5_Flat.index'))
t=time.perf_counter()
cs,ci=cpu.search(q,3)
cpu_seconds=time.perf_counter()-t
co=faiss.GpuMultipleClonerOptions();co.shard=True;co.useFloat16=True
t=time.perf_counter()
gpu=faiss.index_cpu_to_all_gpus(cpu,co=co)
clone_seconds=time.perf_counter()-t
gs,gi=gpu.search(q,3)
overlap=sum(len(set(a)&set(b)) for a,b in zip(ci,gi))/(len(q)*3)
large=np.tile(q,(37,1))[:512].copy()
t=time.perf_counter();gpu.search(large,3);gpu_seconds=time.perf_counter()-t
result={'faiss_version':faiss.__version__,'query_count':len(q),'index_vectors':cpu.ntotal,
        'index_dimension':cpu.d,'mode':'8-GPU sharded FP16 upstream path',
        'cpu_seconds_14_queries':cpu_seconds,'gpu_seconds_512_queries':gpu_seconds,
        'clone_seconds':clone_seconds,'top3_id_overlap':overlap,
        'max_score_delta':float(np.max(np.abs(cs-gs))),
        'cpu_ids':ci.tolist(),'gpu_ids':gi.tolist()}
(BASE/'logs/gpu-retrieval-preflight.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2),flush=True)
assert overlap>=0.95,'Unexpected retrieval disagreement; inspect before migration'
assert np.isfinite(gs).all() and gpu_seconds<30
