"""HKUST-only relay preserving both training arms' existing URL."""
import json,urllib.request,os
from fastapi import FastAPI,Request,HTTPException
import uvicorn
app=FastAPI();target='http://10.120.17.101:18001'
@app.get('/health')
def health():
 with urllib.request.urlopen(target+'/health',timeout=10) as r:return json.load(r)
@app.post('/retrieve')
async def retrieve(request:Request):
 import asyncio
 body=await request.body()
 def send():
  req=urllib.request.Request(target+'/retrieve',data=body,headers={'Content-Type':'application/json'})
  with urllib.request.urlopen(req,timeout=1200) as r:return json.load(r)
 return await asyncio.to_thread(send)
uvicorn.run(app,host='10.123.4.11',port=18000)
