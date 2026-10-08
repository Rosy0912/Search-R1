"""G1 retrieval mathematics, project-local localhost binding and GPU-thread fix."""
import os
from pathlib import Path
import runpy
import uvicorn

original_run=uvicorn.run

def local_run(app,**kwargs):
    kwargs['host']='127.0.0.1'
    kwargs['port']=18010
    return original_run(app,**kwargs)

uvicorn.run=local_run
base=Path(os.environ.get('SEARCH_G1_BASE','/data/search-g1'))
runpy.run_path(str(base/'repo/search_r1/search/retrieval_server.py'),run_name='__main__')
