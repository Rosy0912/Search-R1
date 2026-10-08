"""Binding producer CPU fixtures are toy files, never actual official32B evidence."""
from pathlib import Path
import json
import pytest
import eval64
import build_binding as b
from test_eval64 import inventory


@pytest.fixture
def setup(inventory, monkeypatch):
    root, model, _, binding = inventory
    (root / 'logs').mkdir()
    monkeypatch.setattr(b, 'ROOT', root)
    monkeypatch.setattr(b, 'COMMIT', binding['code_commit'])
    monkeypatch.setattr(b, 'SOURCES', binding['source_files'])
    monkeypatch.setattr(b, 'LAUNCHER_SHA', binding['owned_launcher_sha256'])
    return root, model, Path(binding['official_metadata_path']), Path(binding['data_path']), root / 'logs/binding.json'


def test_complete_real_tiny_safetensors_emit_binding_after_full_gate(setup):
    result = b.create_binding(*setup)
    path = setup[-1]
    assert path.is_file() and json.loads(path.read_text()) == result
    assert len([n for n in result['model_files'] if n.endswith('.safetensors')]) == 17
    assert eval64.asset_gate(path, setup[0])[1] == setup[1]
    assert not list(path.parent.glob('.candidate-binding-*'))


@pytest.mark.parametrize('fault', ['partial', 'short', 'tamper', 'transfer', 'official', 'source', 'launcher', 'data', 'outside'])
def test_bad_real_input_never_creates_final_binding(setup, fault, monkeypatch, tmp_path):
    root, model, metadata, data, output = setup
    shard = next(model.glob('*.safetensors'))
    if fault == 'partial': shard.unlink()
    if fault == 'short': shard.write_bytes(shard.read_bytes()[:-1])
    if fault == 'tamper':
        raw = bytearray(shard.read_bytes()); raw[-1] ^= 1; shard.write_bytes(raw)
    if fault == 'transfer': (model / 'inflight.part').write_bytes(b'partial')
    if fault == 'official': metadata.write_text('{}')
    if fault == 'source': (root / 'repo-terminal-fixed413/search_r1/baselines/bapo.py').write_text('wrong-answer scorer')
    if fault == 'launcher': (root / 'tools/topd-short-run.py').write_text('global-kill')
    if fault == 'data': data.write_bytes(b'other64')
    if fault == 'outside': output = tmp_path / 'outside.json'
    with pytest.raises(RuntimeError): b.create_binding(root, model, metadata, data, output)
    assert not output.exists()
    assert not list((root / 'logs').glob('.candidate-binding-*'))


def test_existing_binding_preserved_without_overwrite(setup):
    output = setup[-1]; output.write_text('previous actual receipt')
    with pytest.raises(RuntimeError, match='existing'): b.create_binding(*setup)
    assert output.read_text() == 'previous actual receipt'


def test_original_head_required_not_declaration_only(setup, monkeypatch):
    monkeypatch.setattr(b.subprocess, 'check_output', lambda *a, **kw: 'foreign-commit\n')
    with pytest.raises(RuntimeError, match='HEAD'): b.create_binding(*setup)
    assert not setup[-1].exists()


def test_runtime_asset_drift_during_fresh_gate_refuses_final(setup, monkeypatch):
    original = eval64.asset_gate
    def race(path, root):
        result = original(path, root)
        (root / 'assets-root/assets/wiki/wiki-18.jsonl').write_bytes(b'changed after hash')
        return result
    monkeypatch.setattr(eval64, 'asset_gate', race)
    with pytest.raises(RuntimeError, match='changed'): b.create_binding(*setup)
    assert not setup[-1].exists()
    assert not list(setup[-1].parent.glob('.candidate-binding-*'))


def test_candidate_header_failure_never_publishes(setup, monkeypatch):
    monkeypatch.setattr(eval64, 'asset_gate', lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('bad actual model header')))
    with pytest.raises(RuntimeError, match='header'): b.create_binding(*setup)
    assert not setup[-1].exists()
    assert not list(setup[-1].parent.glob('.candidate-binding-*'))


def test_concurrent_producer_winner_is_not_overwritten(setup, monkeypatch):
    original = eval64.asset_gate
    def winner(path, root):
        result = original(path, root)
        setup[-1].write_text('other verified producer owns this output')
        return result
    monkeypatch.setattr(eval64, 'asset_gate', winner)
    with pytest.raises(FileExistsError): b.create_binding(*setup)
    assert setup[-1].read_text() == 'other verified producer owns this output'
    assert not list(setup[-1].parent.glob('.candidate-binding-*'))


def test_source_head_changed_after_asset_gate_never_publishes(setup, monkeypatch):
    original = eval64.asset_gate
    def head_changes(path, root):
        result = original(path, root)
        monkeypatch.setattr(b.subprocess, 'check_output', lambda *a, **kw: 'changed-after-model-hash\n')
        return result
    monkeypatch.setattr(eval64, 'asset_gate', head_changes)
    with pytest.raises(RuntimeError, match='HEAD changed'): b.create_binding(*setup)
    assert not setup[-1].exists()
