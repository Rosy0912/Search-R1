"""CPU/local fixture checks: never a source commit, actual GPU job or acceptance."""
import importlib.util
import json
import sys
from pathlib import Path
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('author200_tools_binding', HERE/'finalize_runtime_tools.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_blocked_source_or_path_is_not_finalized(tmp_path):
    path = tmp_path/'controller.py'
    path.write_text("SOURCE='BLOCKED_UNBOUND_SOURCE_COMMIT'\nNATIVE='BLOCKED_NATIVE_TOOLS_PATH'\n")
    with pytest.raises(ValueError, match='Unbound'):
        m.prepared_files(tmp_path, ('controller.py',),
                         {'BLOCKED_UNBOUND_SOURCE_COMMIT': 'a'*40})
    raw = m.prepared_files(tmp_path, ('controller.py',),
                          {'BLOCKED_UNBOUND_SOURCE_COMMIT': 'a'*40,
                           'BLOCKED_NATIVE_TOOLS_PATH': '/reviewed/fixture/path'})
    assert b'BLOCKED_' not in raw['controller.py']


def test_existing_immutable_bundle_can_only_be_reverified(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'HERE', tmp_path)
    files = {'fixture.py': b'VALUE=1\n'}
    manifest = {'scope': 'CPU_FIXTURE_ONLY', 'tools': {'fixture.py': m.sha(files['fixture.py'])}}
    first = m.publish('fixture', files, manifest)
    assert m.publish('fixture', files, manifest) == first
    directory = Path(first['local_directory'])
    (directory/'fixture.py').write_bytes(b'VALUE=2\n')
    with pytest.raises(ValueError, match='changed'):
        m.publish('fixture', files, manifest)
    assert (directory/'fixture.py').read_bytes() == b'VALUE=2\n'


def test_scheduler_implementation_is_explicitly_source_pinned():
    assert 'verl/utils/torch_functional.py' in m.SOURCE_FILES
    assert len(m.SOURCE_FILES) == len(set(m.SOURCE_FILES)) == 16
    assert len(m.FORMAL_FILES) == 8


def test_binding_output_cannot_contain_future_job_id():
    text = (HERE/'finalize_runtime_tools.py').read_text()
    assert 'job_id' not in text
    runner = (HERE/'first5-receipt-candidate/inspect_first5_cpu.py').read_text()
    assert "p.add_argument('--job-id',required=True)" in runner
    assert 'formal-3210' not in runner and 'job_3210' not in runner
    assert "(5,5,6,8,200,0)" in runner


def test_dirty_or_nonactual_source_is_rejected_before_bundle_creation(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['binder', '--repo', str(tmp_path), '--source', 'a'*40])
    monkeypatch.setattr(m, 'git', lambda repo, *args: 'a'*40 if args == ('rev-parse', 'HEAD') else ' M config.yaml')
    with pytest.raises(ValueError, match='Committed clean source'):
        m.main()
    assert list(tmp_path.iterdir()) == []
