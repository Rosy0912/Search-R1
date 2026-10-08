import json,os,pathlib,subprocess,sys,ast
import pytest
from allocator_policy import bootstrap,check_config,worker_environment,POLICY,NAME,VALUE
HERE=pathlib.Path(__file__).resolve().parent
def child(code,env):
 env=dict(env,PYTHONPATH=str(HERE),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='')
 return subprocess.run([sys.executable,'-c',code],env=env,capture_output=True,text=True)
def test_actual_early_bootstrap_before_torch_import():
 env=dict(os.environ);env.pop(NAME,None)
 r=child("from allocator_policy import bootstrap;import sys,os;bootstrap();assert 'torch' not in sys.modules;print(os.environ['PYTORCH_CUDA_ALLOC_CONF'])",env)
 assert r.returncode==0 and r.stdout.strip()==VALUE
 for name in ('supervise.py','train_entry.py'):
  tree=ast.parse((HERE/name).read_text());conditional=next(n for n in tree.body if isinstance(n,ast.If) and '__main__' in ast.unparse(n.test))
  assert ast.unparse(conditional.body[0])=='bootstrap()'
def test_wrong_allocator_rejected_before_import():
 r=child('from allocator_policy import bootstrap;bootstrap()',dict(os.environ,**{NAME:'max_split_size_mb:32'}))
 assert r.returncode!=0 and 'Unexpected CUDA allocator' in r.stderr
def test_late_bootstrap_rejected(monkeypatch):
 monkeypatch.setitem(sys.modules,'torch',object())
 with pytest.raises(RuntimeError,match='before importing Torch'):bootstrap()
def test_immutable_recipe_allocator_and_real_ray_environment(monkeypatch):
 from prepare import derive
 from supervise import verify_recipe
 from test_standard_r1 import prior
 c=derive(prior(),'profile2','/dataset/shaoshitong/search-r1-rl/topd/logs/r1-v8-test');assert verify_recipe(c)==2 and c['cuda_allocator']==POLICY
 assert worker_environment({NAME:VALUE,'HF_HOME':'own','UNRELATED':'no'})=={NAME:VALUE,'HF_HOME':'own'}
 c['cuda_allocator']['value']='expandable_segments:False'
 with pytest.raises(ValueError,match='immutable'):verify_recipe(c)
 with pytest.raises(RuntimeError,match='Ray'):worker_environment({})
def test_native_original_torch24_allocator_parser_no_cuda():
 import torch
 if torch.__version__.split('+')[0]!='2.4.0' or torch.version.cuda is None:pytest.skip('Actual CUDA-built native Torch2.4 CPU parser required')
 r=child("from allocator_policy import bootstrap;bootstrap();import torch;assert not torch.cuda.is_initialized();torch.cuda.memory._set_allocator_settings('expandable_segments:True');assert not torch.cuda.is_initialized();print('ORIGINAL_TORCH24_PARSER_NO_CUDA')",os.environ)
 assert r.returncode==0 and r.stdout.strip()=='ORIGINAL_TORCH24_PARSER_NO_CUDA'
def test_actual_ready_producer_and_parent_enforce_worker_env():
 producer=ast.parse((HERE/'observed_worker.py').read_text());checker=ast.parse((HERE/'resource_audit.py').read_text())
 assert any(isinstance(n,ast.keyword) and n.arg=='cuda_allocator_environment' and ast.unparse(n.value)=="os.environ.get('PYTORCH_CUDA_ALLOC_CONF')" for n in ast.walk(producer))
 fn=next(n for n in checker.body if isinstance(n,ast.FunctionDef) and n.name=='verify_ready')
 # Execute real checker through actual file parse; wrong env fails before later ownership gate.
 import types
 from resource_audit import verify_ready
 import tempfile
 with tempfile.TemporaryDirectory() as directory:
  root=pathlib.Path(directory)
  for role in ('actor','ref'):
   for rank in range(8):(root/(role+'-ready-rank%d.json'%rank)).write_text(json.dumps({'pid':1,'cuda_allocator_environment':'WRONG'}))
  tools=types.SimpleNamespace(process_identity=lambda pid:{})
  with pytest.raises(ValueError,match='allocator'):verify_ready(root,{'uuids':[], 'cuda_allocator':POLICY},0,tools,lambda:{},lambda *a:None)

@pytest.mark.parametrize('segments,valid',[([{'is_expandable':True,'allocated_size':1024}],True),([{'allocated_size':1024}],False),([{'is_expandable':False}],False)])
def test_original_snapshot_metadata_is_required_not_env_only(monkeypatch,segments,valid):
 import types
 from allocator_policy import evidence,check_evidence
 monkeypatch.setenv(NAME,VALUE)
 fake=types.SimpleNamespace(cuda=types.SimpleNamespace(memory=types.SimpleNamespace(_snapshot=lambda:{'segments':segments,'allocator_settings':{'expandable_segments':True}},get_allocator_backend=lambda:'native')))
 raw=evidence(fake);assert raw['snapshot_segments'] is segments
 if valid:assert check_evidence(raw) is raw
 else:
  with pytest.raises(ValueError):check_evidence(raw)

def test_allocator_parsed_false_not_approved_from_env_or_segments(monkeypatch):
 import types
 from allocator_policy import evidence,check_evidence
 monkeypatch.setenv(NAME,VALUE)
 fake=types.SimpleNamespace(cuda=types.SimpleNamespace(memory=types.SimpleNamespace(_snapshot=lambda:{'segments':[{'is_expandable':True}],'allocator_settings':{'expandable_segments':False}},get_allocator_backend=lambda:'native')))
 with pytest.raises(ValueError,match='parsed'):check_evidence(evidence(fake))
