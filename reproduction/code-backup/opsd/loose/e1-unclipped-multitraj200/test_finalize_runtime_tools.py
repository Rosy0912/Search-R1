"""Local synthetic packaging only; no source finalization, remote or training."""
import importlib.util
import json
import os
from pathlib import Path

import pytest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('candidate_finalize',HERE/'finalize_runtime_tools.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_local_exclusive_manifest_last_modes_and_exact_reuse(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'HERE',tmp_path)
    files={'runtime.py':b'pass\n','README.txt':b'CPU packaging fixture only'}
    manifest={'version':1,'tools':{name:m.sha(raw) for name,raw in files.items()}}
    result=m.publish('synthetic-unit',files,manifest)
    directory=Path(result['local_directory'])
    assert directory.stat().st_mode&0o777==0o555
    assert json.loads((directory/'manifest.json').read_bytes())==manifest
    assert all(p.stat().st_mode&0o777==0o444 for p in directory.iterdir())
    assert result==m.publish('synthetic-unit',files,manifest)


@pytest.mark.parametrize('name',['../escape','/absolute','manifest.json','..'])
def test_arbitrary_member_path_refused(tmp_path,monkeypatch,name):
    monkeypatch.setattr(m,'HERE',tmp_path)
    with pytest.raises(ValueError):m.publish('synthetic-unit',{name:b'never-written'},{'version':1})
    assert not list(tmp_path.iterdir())


def test_changed_bytes_or_symlink_existing_bundle_refused(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'HERE',tmp_path)
    files={'runtime.py':b'pass\n'};manifest={'version':1}
    result=m.publish('synthetic-unit',files,manifest);directory=Path(result['local_directory'])
    os.chmod(directory,0o755);p=directory/'runtime.py';os.chmod(p,0o644);p.write_bytes(b'changed')
    os.chmod(p,0o444);os.chmod(directory,0o555)
    with pytest.raises(ValueError):m.publish('synthetic-unit',files,manifest)
    os.chmod(directory,0o755);p.unlink();p.symlink_to(directory/'manifest.json');os.chmod(directory,0o555)
    with pytest.raises(ValueError):m.publish('synthetic-unit',files,manifest)


def test_candidate_unbound_source_refuses_runtime_preparation():
    with pytest.raises(ValueError,match='Unbound'):
        m.prepared_files(HERE/'first5-receipt-candidate',('receipt_payload.py',),{})
    assert 'search_r1/baselines/trajectory_archive.py' in m.SOURCE_FILES
    assert 'search_r1/baselines/rollout.py' in m.SOURCE_FILES
    assert len(m.SOURCE_FILES)==len(set(m.SOURCE_FILES))
