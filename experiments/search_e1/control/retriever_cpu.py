"""Task-private retrieval with one FP32 CPU index shared inside HKUST only."""
import os
probe_devices=os.environ['CUDA_VISIBLE_DEVICES']
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['E1_ENCODER_DEVICE']='cpu'
import torch,faiss,uvicorn
from pathlib import Path
from search_r1.search import retrieval_server as server
base=Path(os.environ['E1_BASE']);torch.set_num_threads(16);faiss.omp_set_num_threads(32)
server.config=server.Config(retrieval_method='e5',retrieval_topk=3,index_path=str(base/'assets/index/e5_Flat.index'),corpus_path=str(base/'assets/corpus/wiki-18.jsonl'),faiss_gpu=False,retrieval_model_path=str(base/'assets/encoder'),retrieval_use_fp16=False,retrieval_batch_size=128)
server.retriever=server.get_retriever(server.config)
# Compare CPU FP32 versus historical GPU FP16 encoder on the same exact index.
# This sample is diagnostic, not proof of equivalence for all future queries.
import subprocess,json,numpy as np
probe_env=dict(os.environ,CUDA_VISIBLE_DEVICES=probe_devices)
subprocess.run([str(base/'env/bin/python'),str(base/'source/experiments/search_e1/control/gpu_admission.py')],check=True,env=probe_env)
queries=[json.loads(line)['question'] for line in (base/'data/validation.jsonl').open()][:16]
cpu=server.retriever.encoder.encode(queries)
(base/'receipts/retriever-probe-queries.json').write_text(json.dumps(queries))
subprocess.run([str(base/'env/bin/python'),str(base/'source/experiments/search_e1/control/retriever_fp16_probe.py')],check=True,env=probe_env)
gpu=np.load(base/'receipts/retriever-gpu-fp16-embeddings.npy')
cs,ci=server.retriever.index.search(cpu,3);gs,gi=server.retriever.index.search(gpu,3)
(base/'receipts/retriever-encoder-comparison.json').write_text(json.dumps(dict(queries=queries,cpu_top3=ci.tolist(),gpu_fp16_top3=gi.tolist(),same_order=sum((ci==gi).all(axis=1).tolist()),total=len(queries),scope='16 fixed holdout questions; both training arms always use CPU FP32'),indent=2))
# The GPU comparator exited; this service process has never created CUDA context.
print('EXACT_FP32_CPU_INDEX_READY' ,server.retriever.index.ntotal,flush=True)
uvicorn.run(server.app,host='10.123.4.11',port=18000)
