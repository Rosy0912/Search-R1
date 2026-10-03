"""Task-private retrieval with one FP32 CPU index shared inside HKUST only."""
import os
os.environ['E1_ENCODER_DEVICE']='cpu'
import torch,faiss,uvicorn
from pathlib import Path
from search_r1.search import retrieval_server as server
base=Path(os.environ['E1_BASE']);torch.set_num_threads(16);faiss.omp_set_num_threads(32)
server.config=server.Config(retrieval_method='e5',retrieval_topk=3,index_path=str(base/'assets/index/e5_Flat.index'),corpus_path=str(base/'assets/corpus/wiki-18.jsonl'),faiss_gpu=False,retrieval_model_path=str(base/'assets/encoder'),retrieval_use_fp16=False,retrieval_batch_size=128)
server.retriever=server.get_retriever(server.config)
# Compare CPU FP32 versus historical GPU FP16 encoder on the same exact index.
# This sample is diagnostic, not proof of equivalence for all future queries.
import subprocess,json
subprocess.run([str(base/'env/bin/python'),str(base/'source/experiments/search_e1/control/gpu_admission.py')],check=True)
queries=[json.loads(line)['question'] for line in (base/'data/validation.jsonl').open()][:16]
cpu=server.retriever.encoder.encode(queries)
os.environ['E1_ENCODER_DEVICE']='cuda'
gpu_encoder=server.Encoder('e5',str(base/'assets/encoder'),server.config.retrieval_pooling_method,server.config.retrieval_query_max_length,True)
gpu=gpu_encoder.encode(queries)
cs,ci=server.retriever.index.search(cpu,3);gs,gi=server.retriever.index.search(gpu,3)
(base/'receipts/retriever-encoder-comparison.json').write_text(json.dumps(dict(queries=queries,cpu_top3=ci.tolist(),gpu_fp16_top3=gi.tolist(),same_order=sum((ci==gi).all(axis=1).tolist()),total=len(queries),scope='16 fixed holdout questions; both training arms always use CPU FP32'),indent=2))
del gpu_encoder;torch.cuda.empty_cache();os.environ['E1_ENCODER_DEVICE']='cpu'
print('EXACT_FP32_CPU_INDEX_READY' ,server.retriever.index.ntotal,flush=True)
uvicorn.run(server.app,host='10.123.4.11',port=18000)
