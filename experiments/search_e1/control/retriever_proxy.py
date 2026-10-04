"""HKUST-only relay preserving both training arms' existing URL."""
import json,urllib.request,urllib.error,os,time
from fastapi import FastAPI,Request,HTTPException
import uvicorn
app=FastAPI();target=os.environ.get('E1_RETRIEVER_TARGET','http://10.120.17.101:18001')
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
@app.get('/health')
def health():
 with opener.open(target+'/health',timeout=10) as r:return json.load(r)
@app.post('/retrieve')
async def retrieve(request:Request):
 import asyncio
 body=await request.body()
 def send():
  req=urllib.request.Request(target+'/retrieve',data=body,headers={'Content-Type':'application/json'})
  deadline=time.monotonic()+1800
  while True:
   try:
    with opener.open(req,timeout=min(1200,max(1,deadline-time.monotonic()))) as r:return json.load(r)
   except (urllib.error.URLError,TimeoutError,OSError) as error:
    # Retrieval is read-only: retry identical bytes against the same oracle.
    # Invalid payloads and backend computation failures must remain visible.
    if isinstance(error,urllib.error.HTTPError) and error.code not in (502,503,504):raise
    remaining=deadline-time.monotonic()
    if remaining<=0:raise
    print('RETRIEVER_TRANSPORT_RETRY',type(error).__name__,str(error),flush=True)
    time.sleep(min(5,remaining))
 return await asyncio.to_thread(send)
uvicorn.run(app,host=os.environ.get('E1_RETRIEVER_BIND','10.123.4.11'),port=int(os.environ.get('E1_RETRIEVER_PORT','18000')))
