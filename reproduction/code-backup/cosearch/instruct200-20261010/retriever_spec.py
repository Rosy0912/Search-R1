"""Construct the native same-node GPU retrieval service command and scoped environment."""
import os
from pathlib import Path
BASE=Path('/data/user/jxu957/experiments/cosearch-continue300-x203-20261008')

def specification(root):
    root=Path(root)
    binary=BASE/'runtime/retriever-python310-candidate/bin/python3.10'
    server=BASE/'source/repo/reproduction/h20/retriever_local.py'
    assets=BASE/'source/assets-root/assets'
    required=[binary,server,assets/'wiki/e5_Flat.index',assets/'wiki/wiki-18.jsonl',assets/'e5-base-v2/config.json']
    for p in required:
        if not p.is_file():raise FileNotFoundError(str(p))
    env=dict(os.environ)
    for key in ('PYTHONHOME','LD_PRELOAD','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):
        env.pop(key,None)
    env.update(PYTHONPATH=':'.join([str(BASE/'runtime/gpu-faiss-alignment-candidate'),str(BASE/'source/env/lib/python3.10/site-packages'),str(BASE/'source/repo')]),
        LD_LIBRARY_PATH=str(BASE/'runtime/retriever-python310-candidate/lib'),
        SEARCH_R1_REPO=str(BASE/'source/repo'),HF_HOME=str(root/'cache/hf-retriever'),
        HF_DATASETS_CACHE=str(root/'cache/datasets-retriever'),TMPDIR=str(root/'tmp'),
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',MKL_NUM_THREADS='8')
    command=[str(binary),str(server),'--index_path',str(assets/'wiki/e5_Flat.index'),'--corpus_path',str(assets/'wiki/wiki-18.jsonl'),
        '--retriever_name','e5','--retriever_model',str(assets/'e5-base-v2'),'--topk','50','--device','cuda','--faiss_gpu','--threads','8','--host','127.0.0.1','--port','18086']
    return command,env
