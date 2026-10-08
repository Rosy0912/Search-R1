"""Actual endpoint contract: CUDA-facing calls stay on the event-loop thread."""
import ast
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
import threading
import unittest
from typing import List,Optional
from fastapi import FastAPI
from pydantic import BaseModel

class RetrievalThreadTest(unittest.TestCase):
    def test_real_fastapi_concurrent_requests_use_initialization_thread(self):
        path=Path(__file__).resolve().parents[2]/'search_r1/search/retrieval_server.py'
        tree=ast.parse(path.read_text())
        nodes=[n for n in tree.body if getattr(n,'name',None) in ['QueryRequest','retrieve_endpoint']]
        created=threading.get_ident()
        calls=[]
        def search(**kwargs):
            self.assertEqual(threading.get_ident(),created)
            calls.append(kwargs)
            return [[{'id':'x','contents':'example'}]],[[0.75]]
        app=FastAPI()
        ns={'app':app,'BaseModel':BaseModel,'List':List,'Optional':Optional,
            'retriever':SimpleNamespace(batch_search=search),'config':SimpleNamespace(retrieval_topk=3)}
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),'exec'),ns)
        async def post():
            messages=[]
            body=json.dumps({'queries':['example'],'return_scores':True}).encode()
            async def receive():
                return {'type':'http.request','body':body,'more_body':False}
            async def send(message):
                messages.append(message)
            scope={'type':'http','asgi':{'version':'3.0'},'http_version':'1.1',
                   'method':'POST','scheme':'http','path':'/retrieve','raw_path':b'/retrieve',
                   'query_string':b'','root_path':'','headers':[(b'content-type',b'application/json')],
                   'client':('127.0.0.1',1234),'server':('127.0.0.1',8000)}
            await app(scope,receive,send)
            self.assertEqual(next(m['status'] for m in messages if m['type']=='http.response.start'),200)
            result=json.loads(b''.join(m.get('body',b'') for m in messages))
            self.assertEqual(result['result'][0][0]['score'],0.75)
        async def concurrent():
            await asyncio.gather(post(),post())
        asyncio.run(concurrent())
        self.assertEqual(len(calls),2)

    def test_async_endpoint_keeps_resource_thread_and_wire_contract(self):
        path=Path(__file__).resolve().parents[2]/'search_r1/search/retrieval_server.py'
        tree=ast.parse(path.read_text())
        endpoint=next(n for n in tree.body if getattr(n,'name',None)=='retrieve_endpoint')
        self.assertIsInstance(endpoint,ast.AsyncFunctionDef)
        endpoint.decorator_list=[]
        created=threading.get_ident()
        calls=[]
        def search(**kwargs):
            self.assertEqual(threading.get_ident(),created)
            calls.append(kwargs)
            return [[{'id':'x','contents':'example'}]],[[0.75]]
        ns={'QueryRequest':SimpleNamespace,'retriever':SimpleNamespace(batch_search=search),
            'config':SimpleNamespace(retrieval_topk=3)}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[endpoint],type_ignores=[])),str(path),'exec'),ns)
        request=SimpleNamespace(queries=['example'],topk=None,return_scores=True)
        result=asyncio.run(ns['retrieve_endpoint'](request))
        self.assertEqual(calls,[{'query_list':['example'],'num':3,'return_score':True}])
        self.assertEqual(result,{'result':[[{'document':{'id':'x','contents':'example'},'score':0.75}]]})
