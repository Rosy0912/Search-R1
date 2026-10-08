import importlib.util
import hashlib
from pathlib import Path
import pytest
P=Path(__file__).with_name('native_import_closure_producer.py')
s=importlib.util.spec_from_file_location('closure_producer',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
@pytest.fixture
def repo(tmp_path,monkeypatch):
    root=tmp_path/'repo';(root/'search_r1').mkdir(parents=True);(root/'verl/workers/rollout').mkdir(parents=True);(root/'reproduction/baselines').mkdir(parents=True)
    (root/'search_r1/__init__.py').write_text('')
    (root/'verl/__init__.py').write_text('')
    (root/'verl/workers/rollout/vllm_rollout.py').write_text('raise RuntimeError("MUST NOT IMPORT")\n')
    checker=root/'reproduction/baselines/e1_import_bootstrap.py';checker.write_text('raise RuntimeError("MUST NOT EXECUTE")\n')
    monkeypatch.setattr(m,'LIBRARIES',())
    return root,hashlib.sha256(checker.read_bytes()).hexdigest()
def test_all_sources_packages_lazy_no_import(repo):
    root,pin=repo;r=m.produce(root,pin)
    assert set(r)=={'modules'} and 'verl.workers.rollout.vllm_rollout' in r['modules']
    assert 'search_r1' in r['modules'] and 'search_r1.__init__' not in r['modules']
    assert 'reproduction.baselines.e1_import_bootstrap' in r['modules']
@pytest.mark.parametrize('pin',['','0'*64])
def test_wrong_external_checker_pin_rejects(repo,pin):
    with pytest.raises(ValueError):m.produce(repo[0],pin)
def test_source_symlink_rejects(repo,tmp_path):
    root,pin=repo;p=tmp_path/'foreign.py';p.write_text('');(root/'search_r1/x.py').symlink_to(p)
    with pytest.raises(ValueError):m.produce(root,pin)
def test_ambiguous_module_package_rejects(repo):
    root,pin=repo;(root/'search_r1/x').mkdir();(root/'search_r1/x/__init__.py').write_text('');(root/'search_r1/x.py').write_text('')
    with pytest.raises(ValueError):m.produce(root,pin)
def test_no_overwrite_reviewed_inventory(repo,tmp_path):
    r=m.produce(*repo);p=tmp_path/'closure.json';m.publish(r,p);old=p.read_bytes()
    with pytest.raises(FileExistsError):m.publish(r,p)
    assert p.read_bytes()==old
