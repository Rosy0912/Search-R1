import ast,asyncio,pathlib,types
import pytest
from verify_api import verify,check_payload
ROOT=pathlib.Path(__file__).parent

def endpoint(path):
 tree=ast.parse(path.read_text());f=next(n for n in tree.body if isinstance(n,ast.AsyncFunctionDef) and n.name=='retrieve_endpoint');f.decorator_list=[];f.args.args[0].annotation=None
 ns={};exec(compile(ast.Module(body=[f],type_ignores=[]),str(path),'exec'),ns);return ns
@pytest.mark.parametrize('count',[1,2,3,32])
@pytest.mark.parametrize('scored',[False,True])
def test_actual_endpoint_matches_original_g1(count,scored):
 docs=[[{'contents':f'doc {i}:{k}'} for k in range(3)] for i in range(count)];scores=[[.5,.4,.3] for _ in range(count)]
 class Retriever:
  def batch_search(self,query_list,num,return_score):
   assert query_list==[str(i) for i in range(count)] and num==3
   return (docs,scores) if return_score else docs
 ns=endpoint(ROOT/'retrieval_server.py');ns.update(retriever=Retriever(),config=types.SimpleNamespace(retrieval_topk=3));request=types.SimpleNamespace(queries=[str(i) for i in range(count)],topk=3,return_scores=scored)
 result=asyncio.run(ns['retrieve_endpoint'](request));check_payload(result,request.queries,scored)
 assert result['result']==([[{'document':d,'score':s} for d,s in zip(row,ss)] for row,ss in zip(docs,scores)] if scored else docs)
def test_no_gpu_retrieval_math_changed():
 base=ROOT/'original-native-server.py';old=ast.parse(base.read_text());new=ast.parse((ROOT/'retrieval_server.py').read_text())
 old.body=[n for n in old.body if not (isinstance(n,ast.AsyncFunctionDef) and n.name=='retrieve_endpoint')];new.body=[n for n in new.body if not (isinstance(n,ast.AsyncFunctionDef) and n.name=='retrieve_endpoint')]
 assert ast.dump(old)==ast.dump(new)
def test_old_actual_false_batch_bug_is_reproduced():
 ns=endpoint(ROOT/'original-native-server.py');ns.update(retriever=types.SimpleNamespace(batch_search=lambda **kw:[[],[],[]]),config=None)
 with pytest.raises(ValueError,match='too many values'):asyncio.run(ns['retrieve_endpoint'](types.SimpleNamespace(queries=['a','b','c'],topk=3,return_scores=False)))
def test_actual_protocol_check_records_both_requests():
 class Session:
  def post(self,url,json,timeout):
   docs=[[{'contents':'verified contents'}]*3 for _ in json['queries']]
   if json['return_scores']:docs=[[{'document':d,'score':.5} for d in row] for row in docs]
   return types.SimpleNamespace(status_code=200,json=lambda:{'result':docs})
 r=verify('http://127.0.0.1:18085/retrieve',Session());assert [x['request']['return_scores'] for x in r['records']]==[False,True]
def test_http500_never_accepted():
 s=types.SimpleNamespace(post=lambda *a,**k:types.SimpleNamespace(status_code=500))
 with pytest.raises(ValueError,match='HTTP 500'):verify('http://127.0.0.1:18085/retrieve',s)
def test_empty_docs_never_accepted():
 with pytest.raises(ValueError,match='Empty'):check_payload({'result':[[{'contents':''}]*3]},['q'],False)
def test_endpoint_body_is_exact_original_g1_branch():
 def body(path):
  tree=ast.parse(path.read_text());node=next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name=='retrieve_endpoint')
  return ast.dump(ast.Module(body=node.body,type_ignores=[]))
 assert body(ROOT/'retrieval_server.py')==body(ROOT/'original-g1-server.py')
