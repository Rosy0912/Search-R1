import importlib.util
from pathlib import Path
import sys
import json
import pytest
ROOT=Path(__file__).parents[1]
sys.path.insert(0,str(ROOT/'topd-32b-pipeline-ipc-candidate'))
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
