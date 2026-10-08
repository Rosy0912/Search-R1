import ast,importlib.util,json,pathlib,threading,os
import pytest
HERE=pathlib.Path(__file__).parent
spec=importlib.util.spec_from_file_location('device_parent',HERE/'parent.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
def test_exclusive_receipt_does_not_replace_existing(tmp_path):
 f=tmp_path/'receipt.json';p.write(f,{'original':1})
 with pytest.raises(FileExistsError):p.write(f,{'changed':2})
 assert json.loads(f.read_text())=={'original':1}
def test_partial_publication_not_visible(tmp_path,monkeypatch):
 original=os.fsync;blocked=threading.Event();release=threading.Event()
 def delay(fd):blocked.set();release.wait(2);return original(fd)
 monkeypatch.setattr(p.os,'fsync',delay);target=tmp_path/'receipt.json';t=threading.Thread(target=p.write,args=(target,{'complete':True}));t.start();assert blocked.wait(2);assert not target.exists();release.set();t.join();assert json.loads(target.read_text())['complete']
def test_worker_dead_does_not_count_as_receipt(tmp_path):
 class Dead:
  def poll(self):return 1
 with pytest.raises(RuntimeError,match='exited'):p.wait_file(tmp_path/'absent',Dead())
def test_real_parent_source_all8_fds_and_finally_cleanup():
 tree=ast.parse((HERE/'parent.py').read_text());text=ast.unparse(tree)
 assert 'validate_inherited_gpu_leases(expected_count=8)' in text and 'lease_ownership(row)' in text
 spawn=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='spawn')
 assert 'rows' in ast.unparse(next(k.value for k in spawn.keywords if k.arg=='pass_fds'))
 main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main');region=next(n for n in main.body if isinstance(n,ast.Try) and n.finalbody)
 assert 'owned.cleanup()' in ast.unparse(ast.Module(body=region.finalbody,type_ignores=[]))
def test_single_worker_visibility_parent_all8_preflight():
 text=(HERE/'parent.py').read_text();assert "CUDA_VISIBLE_DEVICES=rows[0]['uuid']" in text
 entry=(HERE/'probe.sbatch').read_text();assert '--gres=gpu:8' in entry and '-m search_r1.baselines.preflight --' in entry
