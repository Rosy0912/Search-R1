import pathlib,ast,types,asyncio,threading,time,json,hashlib,tempfile,shutil
from concurrent.futures import ThreadPoolExecutor
from typing import List,Optional
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
import httpx
r=pathlib.Path(__file__).resolve().parents[2];hybrid=r/'work/hybrid_retrieval_server.py';original=r/'work/tree-grpo/search_r1/search/retrieval_server.py';results=[]
class FakeRetriever:
 def __init__(self):self.active=0;self.peak=0;self.entries=[];self.lock=threading.Lock()
 def batch_search(self,query_list,num=None,return_score=False):
  with self.lock:self.active+=1;self.peak=max(self.peak,self.active);entry={'thread':threading.get_ident(),'start':time.perf_counter()};self.entries.append(entry)
  time.sleep(.20)
  results=[[{'id':str(i),'contents':q+':'+str(i)} for i in range(num)] for q in query_list];scores=[[float(i)/8 for i in range(num)] for q in query_list]
  with self.lock:self.active-=1;entry['end']=time.perf_counter()
  return (results,scores) if return_score else results

def build(use_wrapper):
 app=FastAPI();fake=FakeRetriever();ns={'BaseModel':BaseModel,'List':List,'Optional':Optional,'config':types.SimpleNamespace(retrieval_topk=3),'retriever':fake}
 tree=ast.parse(original.read_text());nodes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='QueryRequest'];endpoint=next(n for n in tree.body if isinstance(n,ast.AsyncFunctionDef) and n.name=='retrieve_endpoint');endpoint.decorator_list=[];nodes.append(endpoint);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(original),'exec'),ns);app.post('/retrieve')(ns['retrieve_endpoint']);old_openapi=app.openapi();app.openapi_schema=None
 rs=types.SimpleNamespace(app=app,retrieve_endpoint=ns['retrieve_endpoint'],QueryRequest=ns['QueryRequest']);poison=[];metadata={'pid':42,'mode':'fixture'}
 g={'rs':rs,'poison':poison,'metadata':metadata,'asyncio':asyncio,'ThreadPoolExecutor':ThreadPoolExecutor,'a':types.SimpleNamespace(run=pathlib.Path(tempfile.mkdtemp(prefix='cpu-health-fixture-'))),'json':json,'time':time}
 tree=ast.parse(hybrid.read_text());selected=[];in_wrapper=False
 for n in tree.body:
  if isinstance(n,ast.AsyncFunctionDef) and n.name=='provenance':selected.append(n)
  if use_wrapper:
   if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='original_endpoint' for t in n.targets):in_wrapper=True
   if in_wrapper:
    if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='print':break
    selected.append(n)
 exec(compile(ast.Module(body=selected,type_ignores=[]),str(hybrid),'exec'),g)
 return app,fake,poison,g,old_openapi

async def run():
 main=threading.get_ident()
 for use in (False,True):
  app,fake,poison,g,old_schema=build(use);transport=httpx.ASGITransport(app=app,raise_app_exceptions=False)
  async with httpx.AsyncClient(transport=transport,base_url='http://fixture') as client:
   started=time.perf_counter();one=asyncio.create_task(client.post('/retrieve',json={'queries':['a'],'topk':3,'return_scores':True}));two=asyncio.create_task(client.post('/retrieve',json={'queries':['b'],'topk':3,'return_scores':True}))
   await asyncio.sleep(.01);active=fake.active;t=time.perf_counter();health=await client.get('/provenance');health_total=time.perf_counter()-started;health_request=time.perf_counter()-t;first,second=await asyncio.gather(one,two)
   assert health.status_code==200 and first.status_code==second.status_code==200
   assert first.json()=={'result':[[{'document':{'id':str(i),'contents':'a:'+str(i)},'score':float(i)/8} for i in range(3)]]}
   assert second.json()=={'result':[[{'document':{'id':str(i),'contents':'b:'+str(i)},'score':float(i)/8} for i in range(3)]]}
   assert fake.peak==1 and fake.entries[1]['start']>=fake.entries[0]['end']
   if use:assert active==1 and health_total<.1 and len({e['thread'] for e in fake.entries})==1 and fake.entries[0]['thread']!=main
   else:assert active==0 and health_total>=.4 and all(e['thread']==main for e in fake.entries)
   schema=app.openapi();oldpath=old_schema['paths']['/retrieve']['post'];newpath=schema['paths']['/retrieve']['post']
   assert oldpath==newpath and old_schema['components']['schemas']==schema['components']['schemas']
   poison.append('simulated protocol fault');bad=await client.get('/provenance');assert bad.status_code==503
   if use:
    before=len(fake.entries);bad_query=await client.post('/retrieve',json={'queries':['c'],'return_scores':True});assert bad_query.status_code==503 and len(fake.entries)==before
   results.append({'wrapper':use,'health_completed_after_launch_seconds':health_total,'health_request_seconds':health_request,'retrieval_active_when_health_requested':active,'retrieval_threads':[e['thread'] for e in fake.entries],'max_parallel_retrieval':fake.peak,'original_response_exact':True,'request_response_schema_equal':True,'old_operation_id':oldpath['operationId'],'new_operation_id':newpath['operationId'],'old_description':oldpath.get('description'),'new_description':newpath.get('description'),'poison_health503':True})
  if use:g['cpu_executor'].shutdown(wait=True)
  shutil.rmtree(g['a'].run)
 for mode in ('cpu_exception','scaled_request_timeout'):
  app,fake,poison,g,_=build(True)
  if mode=='cpu_exception':
   def failure(**kwargs):raise RuntimeError('fixture CPU or index exception')
   fake.batch_search=failure
  else:
   async def scaled(future,timeout):
    assert timeout==1800
    return await asyncio.wait_for(future,timeout=.03)
   g['asyncio']=types.SimpleNamespace(get_running_loop=asyncio.get_running_loop,run=asyncio.run,wait_for=scaled)
  async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app,raise_app_exceptions=False),base_url='http://fixture') as client:
   begun=time.perf_counter();response=await client.post('/retrieve',json={'queries':['a'],'return_scores':True});elapsed=time.perf_counter()-begun
   assert response.status_code==500 and poison and (g['a'].run/'service-failure.json').is_file()
   health=await client.get('/provenance');nextquery=await client.post('/retrieve',json={'queries':['b'],'return_scores':True});assert health.status_code==nextquery.status_code==503
   results.append({'case':mode,'first_status':500,'health_status':503,'next_query_status':503,'failure_written':True,'response_wait_seconds':elapsed,'background_cpu_active_after_timeout':fake.active if mode=='scaled_request_timeout' else 'not applicable','timeout_fixture_scope':'production1800s argument verified, wait scaled to30ms; no1800s real wait'})
  g['cpu_executor'].shutdown(wait=True);shutil.rmtree(g['a'].run)
asyncio.run(run());(r/'work/aux-gate-review/cpu-health-concurrency-fixtures.json').write_text(json.dumps({'scope':'actual source AST CPU endpoint/wrapper, real FastAPI/ASGI HTTP client, fake blocking numeric retriever; no sockets/model/GPU','source_sha256':hashlib.sha256(hybrid.read_bytes()).hexdigest(),'results':results},indent=2)+'\n');print(json.dumps(results,indent=2))
