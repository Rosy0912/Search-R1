import ast,importlib.util,json,copy
from pathlib import Path
HERE=Path(__file__).parent

def test_atomic_worker_receipt(tmp_path):
    spec=importlib.util.spec_from_file_location('burst_worker',HERE/'worker.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    p=tmp_path/'receipt.json';m.write(p,dict(pid=37,devices=[dict(allocated=0,reserved=0)]))
    assert json.loads(p.read_text())['pid']==37 and not p.with_suffix('.tmp').exists()

def test_parent_module_import_is_cpu_only():
    tree=ast.parse((HERE/'probe.py').read_text())
    assert all(not isinstance(node,(ast.Import,ast.ImportFrom)) or all('torch' not in x.name for x in node.names) for node in tree.body)
    assert not any(isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr in ('from_pretrained','LLM','load_model') for node in ast.walk(tree))

def test_preflight_and_phase_release_scope():
    text=(HERE/'probe.sbatch').read_text()
    assert '-m search_r1.baselines.preflight --' in text and '--gres=gpu:8' in text
    source=(HERE/'worker.py').read_text()
    assert '(.25,2.,6.)' in source and 'Live CUDA pulse threads' in source
    assert source.index("assert not any(t.is_alive()") < source.index('torch._C._cuda_clearCublasWorkspaces()',source.index("assert not any(t.is_alive()"))


def actual_trace():
    ids={'20':dict(pid=20,starttime_ticks=7,cgroup=['0::/job'])}
    cap=dict(raw_snapshot=dict(identities=ids,gpus=[dict(uuid=str(i),sm=93) for i in range(8)]),uids={'20':0})
    low=copy.deepcopy(cap)
    for g in low['raw_snapshot']['gpus']:g['sm']=0
    return [dict(attempt=1,accepted=False,error='Device SM increased after upper-bound sample',retry_authorized=True,
                 foreign_sm_status='UNKNOWN_REJECTED_TRANSITION',retry_kind='FRESH_LOW_TO_HIGH_MEASUREMENT',captures=[low,copy.deepcopy(cap)]),
            dict(attempt=2,accepted=True,captures=[copy.deepcopy(cap),copy.deepcopy(cap)],fresh_measurement_uuids=['0'],
                 fresh_foreign_sm_upper_bounds={'0':0},fresh_foreign_sm_evidence={'0':dict(upper_bound=0,method='FRESH_SUPPORTED_NONZERO_PROCESS_WINDOW')})]

def coverage_fn():
    spec=importlib.util.spec_from_file_location('burst_probe',HERE/'probe.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m.coverage

def test_actual_raw_coverage_requires_rejected_then_accepted():
    assert coverage_fn()(actual_trace(),{str(i) for i in range(8)})['typed_transition_retries_observed']==1


def test_no_typed_branch_is_not_proof():
    import pytest
    with pytest.raises(RuntimeError):coverage_fn()(actual_trace()[1:],{str(i) for i in range(8)})


def test_between_attempt_pid_change_is_not_coverage():
    import pytest,copy
    trace=copy.deepcopy(actual_trace());trace[1]['captures'][0]['raw_snapshot']['identities']['30']={'pid':30}
    with pytest.raises(RuntimeError):coverage_fn()(trace,{str(i) for i in range(8)})


def test_missing_actual_all8_high_is_not_coverage():
    import pytest,copy
    trace=copy.deepcopy(actual_trace())
    for row in trace:
        for cap in row['captures']:cap['raw_snapshot']['gpus'][0]['sm']=0
    with pytest.raises(RuntimeError):coverage_fn()(trace,{str(i) for i in range(8)})


def test_toy_pulse_trigger_after_real_snapshot_same_object(tmp_path):
    spec=importlib.util.spec_from_file_location('burst_probe_trigger',HERE/'probe.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    original={'actual_device_sm':7};calls=[]
    def read():
        calls.append((tmp_path/'command-1').exists());return original
    callback=m.trigger_once_after_snapshot(read,tmp_path,1)
    assert callback() is original and calls==[False]
    assert (tmp_path/'command-1').exists()
    assert json.loads((tmp_path/'trigger-1.json').read_text())['actual_snapshot']==original
    assert callback() is original and calls==[False,True]
