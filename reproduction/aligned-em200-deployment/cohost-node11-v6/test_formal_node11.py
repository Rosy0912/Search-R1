"""CPU binding fixtures only; no scheduler/assets/runtime execution."""
import importlib.util
import json
from pathlib import Path
import pytest

@pytest.fixture
def admission():
    p=Path(__file__).with_name('formal_admission.py');s=importlib.util.spec_from_file_location('node11_admission_fixture',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def candidate(m):
    return dict(source=m.SOURCE,root=str(m.ROOT),host='h100-11',scope='SHARED_ASSET_PATH_BINDINGS_NOT_GPU_ACCEPTANCE',model=str(m.MODEL),common=str(m.COMMON),retrieval_assets=str(m.RETRIEVAL),**{n:dict(path=str(m.ROOT/'inputs'/('baseline-aligned-'+n+'.json')),sha256='a'*64)for n in ('data','order','retrieval')})


def test_node11_shared_path_manifest_valid_without_claiming_hardware(admission,monkeypatch):
    b=candidate(admission);monkeypatch.setattr(admission,'raw_file',lambda p:json.dumps(b).encode());monkeypatch.setattr(admission,'sha',lambda p:'a'*64)
    assert admission.bindings()['scope']=='SHARED_ASSET_PATH_BINDINGS_NOT_GPU_ACCEPTANCE'


@pytest.mark.parametrize('change',[{'host':'h100-5'},{'scope':'PASS_INDEPENDENT_ALIGNED_HARDWARE_REVIEW'},{'source':'old'},{'model':'/dataset/local-unavailable'}])
def test_foreign_host_receipt_source_and_local_model_path_refused(admission,monkeypatch,change):
    b=candidate(admission);b.update(change);monkeypatch.setattr(admission,'raw_file',lambda p:json.dumps(b).encode());monkeypatch.setattr(admission,'sha',lambda p:'a'*64)
    with pytest.raises(ValueError):admission.bindings()


def test_derived_input_changed_sha_refused(admission,monkeypatch):
    b=candidate(admission);monkeypatch.setattr(admission,'raw_file',lambda p:json.dumps(b).encode());monkeypatch.setattr(admission,'sha',lambda p:'b'*64)
    with pytest.raises(ValueError,match='input pin changed'):admission.bindings()


def test_launcher_retains_shared_global_locks_and_no_idle_gate():
    b=Path(__file__).parent;worker=(b/'formal_worker.sh').read_text();submit=(b/'submit_formal.py').read_text();batch=(b/'formal.sbatch').read_text();driver=(b/'formal_driver.py').read_text()
    assert '$OLDROOT/scheduler-head/execution-global.lock' in worker
    assert '/bapo-baseline-aligned-em200-20261009/scheduler-head/submission-global.lock' in submit
    assert '--exclusive' not in batch and '--nodelist=h100-11' in batch
    assert '-m search_r1.baselines.preflight --' not in worker
    assert "validate_inherited_gpu_leases(os.environ['CUDA_VISIBLE_DEVICES'].split(','), require_guard=True)" in driver
    assert 'ppo_bapo_baseline_aligned_production' in driver and 'gpu_memory_utilization' not in driver
    assert worker.index('export BAPO_RUN_ID=')<worker.index('bootstrap.py')
