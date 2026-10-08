"""Keep the upstream retrieval implementation; bind its HTTP server locally."""
import runpy
import os
from pathlib import Path
import torch  # Preload the project's CUDA/cuBLAS libraries before GPU FAISS.
import uvicorn

original_run = uvicorn.run
def local_run(app, **kwargs):
    kwargs['host'] = '127.0.0.1'
    return original_run(app, **kwargs)
uvicorn.run = local_run
repo = Path(os.environ.get('SEARCH_R1_REPO', str(Path(__file__).resolve().parents[2])))
runpy.run_path(str(repo / 'search_r1/search/retrieval_server.py'), run_name='__main__')
