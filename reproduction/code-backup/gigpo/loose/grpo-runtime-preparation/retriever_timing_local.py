"""Unchanged local retriever with detached per-request wall-time observations."""
import contextvars,json,os,runpy,time
from pathlib import Path
import torch  # Match reviewed torch-before-FAISS load order.
import uvicorn
stats=contextvars.ContextVar('retriever_request_stats',default=None)

def timed(label, function):
    def call(*args,**kwargs):
        current=stats.get();started=time.perf_counter()
        try:
            return function(*args,**kwargs)
        finally:
            if current is not None:
                current[label]=current.get(label,0.)+time.perf_counter()-started
                if label=='encoder_seconds':
                    query=args[0] if args else kwargs.get('query_list',[])
                    current['queries']+=1 if isinstance(query,str) else len(query)
    return call

class IndexObservation:
    def __init__(self,index):
        self.original=index
        self.search=timed('index_seconds',index.search)
    def __getattr__(self,name):
        return getattr(self.original,name)

class ObservationMiddleware:
    def __init__(self,app):self.app=app
    async def __call__(self,scope,receive,send):
        if scope['type']!='http' or scope.get('path')!='/retrieve':
            return await self.app(scope,receive,send)
        current={'started_unix_seconds':time.time(),'queries':0,'encoder_seconds':0.,'index_seconds':0.,'document_seconds':0.}
        token=stats.set(current);started=time.perf_counter()
        try:
            return await self.app(scope,receive,send)
        finally:
            current['server_http_seconds']=time.perf_counter()-started
            current['scope']='server ASGI app handling including request-body receive wait and response send; excludes client queue and pre-ASGI transport/network time; no extra CUDA synchronization'
            stats.reset(token)
            print('RETRIEVER_REQUEST_TIMING '+json.dumps(current),flush=True)

original_run=uvicorn.run

def local_run(app,**kwargs):
    routes=[r for r in app.routes if getattr(r,'path',None)=='/retrieve']
    assert len(routes)==1
    namespace=routes[0].endpoint.__globals__
    retriever=namespace['retriever']
    retriever.encoder.encode=timed('encoder_seconds',retriever.encoder.encode)
    retriever.index=IndexObservation(retriever.index)
    namespace['load_docs']=timed('document_seconds',namespace['load_docs'])
    app.add_middleware(ObservationMiddleware)
    kwargs['host']='127.0.0.1'
    return original_run(app,**kwargs)
uvicorn.run=local_run
repo=Path(os.environ['AGENT_ROOT'])/'repo'
runpy.run_path(str(repo/'search_r1/search/retrieval_server.py'),run_name='__main__')
