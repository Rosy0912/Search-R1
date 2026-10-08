import ast,pathlib,sys,types,hashlib
import pytest
sys.path.insert(0,str(pathlib.Path(__file__).parent))
from supervise import health_response

def test_retriever_docs_html_200_never_json_parsed():
 def forbidden():raise AssertionError('HTML must not be parsed as JSON')
 response=types.SimpleNamespace(status_code=200,headers={'content-type':'text/html'},json=forbidden)
 assert health_response(response,probe=False)['status_code']==200
 response.status_code=404;assert health_response(response,probe=False) is None

def test_probe_health_still_requires_actual_json_ok_true():
 response=types.SimpleNamespace(status_code=200,headers={},json=lambda:{'ok':False})
 assert health_response(response,probe=True) is None
 response.json=lambda:{'ok':True};assert health_response(response,probe=True)=={'ok':True}

def test_original_fastapi_app_has_docs_route_and_no_health():
 from fastapi import FastAPI
 source=pathlib.Path(__file__).parents[2]/'outputs/reference-search-r1/search_r1__search__retrieval_server.py'
 expected='f45f59caf9d0c4c5cceda74b7e6b8d825b5ad103b83fb8e0c3be9c9afabeb9ef'
 if not source.is_file():
  expected='b2eed2cbddbbe3c5bdc2d7f1e8b6f3738dc5629f3d42e0ba68690c82c9bcd052'
  source=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd/repo/search_r1/search/retrieval_server.py')
 assert hashlib.sha256(source.read_bytes()).hexdigest()==expected
 tree=ast.parse(source.read_text());declared=[]
 for function in tree.body:
  if isinstance(function,(ast.FunctionDef,ast.AsyncFunctionDef)):
   for decorator in function.decorator_list:
    if isinstance(decorator,ast.Call) and isinstance(decorator.func,ast.Attribute) and isinstance(decorator.func.value,ast.Name) and decorator.func.value.id=='app' and decorator.args:declared.append(decorator.args[0].value)
 assert '/retrieve' in declared and '/health' not in declared
 node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='app' for x in n.targets))
 namespace={'FastAPI':FastAPI};exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
 paths={r.path for r in namespace['app'].routes}
 assert '/docs' in paths and '/health' not in paths
