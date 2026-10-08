"""Actual transport configuration; never sends a network request."""
import ast,pathlib,urllib.request
from unittest import mock
import pytest
import requests
ROOT=pathlib.Path(__file__).parent
@pytest.mark.parametrize('filename',['check_h100_native_retriever.py','probe_h100_gpu_retrieval_http.py'])
def test_actual_direct_assignment_suppresses_proxy_env(filename):
 tree=ast.parse((ROOT/filename).read_text());node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='DIRECT' for x in n.targets));ns={'urllib':types_urllib()}
 with mock.patch.dict('os.environ',{'HTTP_PROXY':'http://forbidden.invalid:9','HTTPS_PROXY':'http://forbidden.invalid:9','ALL_PROXY':'http://forbidden.invalid:9','http_proxy':'http://forbidden.invalid:9','https_proxy':'http://forbidden.invalid:9','all_proxy':'http://forbidden.invalid:9'},clear=True):
  exec(compile(ast.Module(body=[node],type_ignores=[]),filename,'exec'),ns)
  assert not any(isinstance(h,urllib.request.ProxyHandler) and h.proxies for h in ns['DIRECT'].handlers)
 def no_network():raise AssertionError('network must never run in this fixture')

def types_urllib():
 import types
 return types.SimpleNamespace(request=urllib.request)

def test_actual_training_no_proxy_bypasses_requests_environment():
 text=(ROOT/'train_tree_stage_h100_native.sh').read_text();assert 'export NO_PROXY=127.0.0.1,localhost,10.32.0.11' in text and 'export no_proxy="$NO_PROXY"' in text
 setting='127.0.0.1,localhost,10.32.0.11'
 with mock.patch.dict('os.environ',{'HTTP_PROXY':'http://forbidden.invalid:9','HTTPS_PROXY':'http://forbidden.invalid:9','ALL_PROXY':'http://forbidden.invalid:9','NO_PROXY':setting,'no_proxy':setting},clear=True):
  s=requests.Session();config=s.merge_environment_settings('http://10.32.0.11:18082/retrieve',{},False,None,None);assert config['proxies']=={}
