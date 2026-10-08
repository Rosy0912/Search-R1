import importlib.util
from pathlib import Path
import sys
import json
import pytest
ROOT=Path(__file__).parents[1]
sys.path.insert(0,str(ROOT/'topd-32b-pipeline-ipc-r1-v2'))
spec=importlib.util.spec_from_file_location('owner_candidate',Path(__file__).with_name('owner.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_fresh_inventory_detects_actual_mutation(tmp_path):
    p=tmp_path/'weights';p.write_bytes(b'original frozen tensor inventory fixture')
    hashes={'weights':m.sha(p)};m.fresh_files(tmp_path,hashes)
    p.write_bytes(b'changed frozen tensor inventory fixture')
    with pytest.raises(RuntimeError,match='SHA mismatch'):m.fresh_files(tmp_path,hashes)


@pytest.mark.parametrize('kind',['missing','symlink','traversal','empty'])
def test_incomplete_or_nonlocal_inventory_rejected(tmp_path,kind):
    p=tmp_path/'real';p.write_bytes(b'actual')
    files={'real':m.sha(p)}
    if kind=='missing':files['missing']='a'*64
    elif kind=='symlink':(tmp_path/'linked').symlink_to(p);files={'linked':m.sha(p)}
    elif kind=='traversal':files={'../real':m.sha(p)}
    else:files={}
    with pytest.raises(RuntimeError):m.fresh_files(tmp_path,files)


@pytest.mark.parametrize('kind',['CPU_FIXTURE_NO_ALGORITHM','NATIVE_HARDWARE_PROBE_NO_TRAINING','GRPO500_TEACHER'])
def test_legacy_kind_refused_before_any_model_load(kind):
    with pytest.raises(RuntimeError,match='cannot start production'):
        m.owner_assets({'kind':kind},'/missing/repo','/missing/pipeline')


def test_production_missing_actual32b_proof_refused():
    with pytest.raises((KeyError,RuntimeError)):
        m.owner_assets({'kind':'NATIVE_32B_TOPD_SHARED_PIPELINE'},'/missing/repo','/missing/pipeline')


def test_canonical_size_and_sha_both_required(tmp_path):
    p=tmp_path/'model.safetensors';p.write_bytes(b'actual full byte fixture')
    canonical={'model.safetensors':{'size':p.stat().st_size,'sha256':m.sha(p)}}
    m.fresh_files(tmp_path,canonical)
    canonical['model.safetensors']['size']+=1
    with pytest.raises(RuntimeError,match='size mismatch'):m.fresh_files(tmp_path,canonical)

def test_actual_protocol_identity_success_and_commit_drift():
    from topd32b_shared_pipeline import backend
    manifest={key:'bound' for key in backend.IDENTITY};manifest['code_commit']='d23-actual';manifest['commit']='d23-actual'
    m.require_source_identity(backend,manifest,{'commit':'d23-actual'})
    manifest['commit']='other'
    with pytest.raises(RuntimeError,match='source identity'):m.require_source_identity(backend,manifest,{'commit':'d23-actual'})

def test_ipc_actual_v2_file_set_and_mutation(tmp_path):
    import import_gate as gate
    source=ROOT/'topd-32b-pipeline-ipc-r1-v2/topd32b_shared_pipeline'
    import shutil
    package=tmp_path/'package';shutil.copytree(source,package)
    gate.verify_package(package,gate.pins())
    (package/'backend.py').write_text((package/'backend.py').read_text()+'\n# changed\n')
    with pytest.raises(RuntimeError,match='SHA drift'):gate.verify_package(package,gate.pins())

def test_ipc_extra_module_and_symlink_refused(tmp_path):
    import import_gate as gate
    (tmp_path/'__init__.py').write_text('');records={'__init__.py':gate.sha(tmp_path/'__init__.py')}
    (tmp_path/'extra.py').write_text('import torch')
    with pytest.raises(RuntimeError,match='exact source'):gate.verify_package(tmp_path,records)
    (tmp_path/'extra.py').unlink();(tmp_path/'linked').symlink_to(tmp_path,target_is_directory=True)
    with pytest.raises(RuntimeError,match='symlink'):gate.verify_package(tmp_path,records)

def test_verified_v2_import_real_modules_without_cuda(tmp_path,monkeypatch):
    import import_gate as gate,shutil,torch
    source=ROOT/'topd-32b-pipeline-ipc-r1-v2/topd32b_shared_pipeline';package=tmp_path/'tools/topd-32b-pipeline-ipc-r1-v2/topd32b_shared_pipeline';shutil.copytree(source,package)
    monkeypatch.setattr(gate,'ROOT',tmp_path);monkeypatch.setenv('PYTHONDONTWRITEBYTECODE','1');monkeypatch.setattr(sys,'path',list(sys.path))
    for name in list(sys.modules):
        if name=='topd32b_shared_pipeline' or name.startswith('topd32b_shared_pipeline.'):monkeypatch.delitem(sys.modules,name)
    backend=gate.verified_ipc()
    assert Path(backend.__file__).resolve()==package/'backend.py' and not torch.cuda.is_initialized()
    # Import gate checks the actual complete package; no scoring/model load happens.
    for name in list(sys.modules):
        if name=='topd32b_shared_pipeline' or name.startswith('topd32b_shared_pipeline.'):del sys.modules[name]

def test_verified_v2_import_rejects_cached_foreign_module(tmp_path,monkeypatch):
    import import_gate as gate,shutil,types
    package=tmp_path/'tools/topd-32b-pipeline-ipc-r1-v2/topd32b_shared_pipeline';shutil.copytree(ROOT/'topd-32b-pipeline-ipc-r1-v2/topd32b_shared_pipeline',package)
    monkeypatch.setattr(gate,'ROOT',tmp_path);monkeypatch.setenv('PYTHONDONTWRITEBYTECODE','1')
    monkeypatch.setitem(sys.modules,'topd32b_shared_pipeline.backend',types.SimpleNamespace(__file__='/foreign/backend.py'))
    with pytest.raises(RuntimeError,match='foreign IPC'):gate.verified_ipc()


def context_source_condition(root,ref):
    import ast,owner
    tree=ast.parse(Path(owner.__file__).read_text())
    call=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='require' and len(n.args)>1 and isinstance(n.args[1],ast.Constant) and n.args[1].value=='Canonical V7 gate source drift before context phase')
    return eval(compile(ast.Expression(call.args[0]),'<actual-owner-context-gate>','eval'),dict(ROOT=root,gate_path=Path(ref['path']),gate_ref=ref,sha=lambda p:ref['actual_sha']))

def test_actual_context_gate_accepts_current_v7(tmp_path):
    h='fbc1507e5a1f1e6f16ba588763e58670f67436276191b8c19fbb4a765387097f'
    assert context_source_condition(tmp_path,dict(path=str(tmp_path/'tools/topd-32b-native-score-gate-r1-v2/score_gate.py'),sha256=h,actual_sha=h))

@pytest.mark.parametrize('drift',['old_path','wrong_bytes'])
def test_actual_context_gate_rejects_old_caller_or_current_byte_drift(tmp_path,drift):
    h='fbc1507e5a1f1e6f16ba588763e58670f67436276191b8c19fbb4a765387097f'
    ref=dict(path=str(tmp_path/'tools/topd-32b-native-score-gate-r1-v2/score_gate.py'),sha256=h,actual_sha=h)
    if drift=='old_path':ref['path']=str(tmp_path/'tools/topd-32b-native-score-gate-v2/score_gate.py')
    else:ref['actual_sha']='0'*64
    assert not context_source_condition(tmp_path,ref)
