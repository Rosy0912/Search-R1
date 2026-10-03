"""Use upstream retrieval math; bind this project's separate localhost port."""
import runpy
import torch
import uvicorn

original_run=uvicorn.run
def local_run(app,**kwargs):
    kwargs['host']='127.0.0.1'
    kwargs['port']=18000
    return original_run(app,**kwargs)
uvicorn.run=local_run
runpy.run_path('/data/search-r1/repo/search_r1/search/retrieval_server.py',run_name='__main__')
