"""Local supplement contract checks; actual proc lifecycle is native-only."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('native_transport_probe',HERE/'probe.py');probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)


def test_supplement_binds_exact_immutable_candidate_bytes():
    require=probe.require
    package=HERE.parent/'topd-official-resume-transfer'
    assert probe.sha(package/'candidate-v6.tar.gz')==probe.ARCHIVE_SHA
    assert probe.sha(package/'transfer.py')==probe.TRANSFER_SHA
    assert probe.sha(package/'official-metadata.json')==probe.METADATA_SHA


def test_supplement_preflight_outside_ownroot_fails_before_import_cuda(tmp_path,monkeypatch):
    monkeypatch.setattr(probe,'HERE',tmp_path)
    with pytest.raises(RuntimeError,match='Foreign ownROOT'):probe.preflight()


def test_record_is_exclusive_sha_bound_private(tmp_path):
    receipt=probe.write_record(tmp_path,'proof.json',dict(scope='CPU_FIXTURE_ONLY',pid=123))
    assert receipt['sha256']==probe.sha(tmp_path/'proof.json')
    assert (tmp_path/'proof.json').stat().st_mode&0o777==0o600
    assert json.loads((tmp_path/'proof.json').read_text())['pid']==123
    with pytest.raises(FileExistsError):probe.write_record(tmp_path,'proof.json',dict(pid=456))
    assert json.loads((tmp_path/'proof.json').read_text())['pid']==123


def test_tiny_probe_refuses_assets_or_foreign_directory_before_child(tmp_path,monkeypatch):
    def no_child(*a,**kw):raise AssertionError('No child should be spawned')
    monkeypatch.setattr(probe.subprocess,'Popen',no_child)
    with pytest.raises(RuntimeError,match='ownROOT/logs'):probe.tiny_probe(tmp_path,lambda *a,**kw:None)
