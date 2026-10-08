"""CPU evidence contracts; synthetic samples do not establish hardware support."""
import os
import sys
from pathlib import Path
import pytest
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(os.environ['AGENT_ROOT'])/'repo' if os.environ.get('AGENT_ROOT') else HERE.parent/'topd'))
sys.path.insert(0, str(HERE))
import runtime_safety_candidate as c

F, O = {('u',1)}, {('u',2)}
def run(sm=80, status='SUCCESS', rows=None, start=1000000, end=2200000, now=2200000):
    rows = [dict(pid=2,timestamp_us=2000000,sm=80)] if rows is None else rows
    return c.sm_aggregate_bounds({'gpus':[dict(uuid='u',sm=sm)]},F,O,
        {'u':dict(status=status,start_us=start,end_us=end,samples=rows)},now)

def test_idle_notfound_device_bound_is_not_pid_zero():
    bound, ev = run(sm=7,status='NVML_ERROR_6',rows=[])
    assert bound == {'u':7}
    assert ev['u']['absent_pid_values']=='NOT_INFERRED'

def test_active_supported_missing_foreign_infers_only_nonzero_filter():
    bound, ev=run()
    assert bound=={'u':0}
    assert ev['u']['absent_interpretation']=='INFERRED_ZERO_NONZERO_FILTER_NOT_MEASURED'
    assert ev['u']['positive_own_pids']==[2]

def test_supported_foreign_nonzero_is_counted():
    bound,ev=run(rows=[dict(pid=2,timestamp_us=2000000,sm=80),dict(pid=1,timestamp_us=2000000,sm=16)])
    assert bound['u']==16
    assert ev['u']['foreign_observed_maxima']=={1:16}

@pytest.mark.parametrize('status',['NVML_ERROR_6','NVML_ERROR_3','UNKNOWN'])
def test_active_errors_never_infer_zero(status):
    with pytest.raises(Exception): run(status=status)

@pytest.mark.parametrize('rows',[
    [],[dict(pid=2,timestamp_us=999999,sm=80)],
    [dict(pid=2,timestamp_us=2200001,sm=80)],
    [dict(pid=2,timestamp_us=2000000,sm=0)],
    [dict(pid=3,timestamp_us=2000000,sm=80)],
    [dict(pid=2,timestamp_us=2000000,sm=float('nan'))],
    [dict(pid=2,timestamp_us=2000000,sm=80),dict(pid=1,timestamp_us=2100000,sm=1)]])
def test_invalid_positive_or_incomplete_period_refuses(rows):
    with pytest.raises(Exception): run(rows=rows)

def test_stale_query_refuses():
    with pytest.raises(Exception): run(now=5000000)

def test_cross_gpu_own_positive_cannot_prove_support():
    with pytest.raises(Exception):
        c.sm_aggregate_bounds({'gpus':[dict(uuid='other',sm=80)]},{('other',1)},O,
          {'other':dict(status='SUCCESS',start_us=1000000,end_us=2200000,
                       samples=[dict(pid=2,timestamp_us=2000000,sm=80)])},2200000)


def test_window_end_fresh_but_sample_old_refuses():
    with pytest.raises(Exception): run(now=4100000)


@pytest.mark.parametrize('initial_device_sm,final_device_sm,expected_bound',[(90,0,0),(7,2,7)])
def test_active_high_then_idle_does_not_charge_own_activity_to_foreign(monkeypatch,initial_device_sm,final_device_sm,expected_bound):
    import runtime_safety
    import search_r1.baselines.topd_local_gpu as backend
    import time
    now=int(time.time()*1000000)
    gpu=[dict(index=str(i),uuid='u'+str(i),used_mib=8000,total_mib=80000,sm=initial_device_sm) for i in range(8)]
    ids={str(i+1):dict(pid=i+1,starttime_ticks=1,cgroup=['0::/foreign']) for i in range(8)}
    ids['99']=dict(pid=99,starttime_ticks=1,cgroup=['0::/ours'])
    apps=[dict(uuid='u'+str(i),pid=i+1,used_mib=1000) for i in range(8)]
    base=dict(gpus=gpu,applications=apps,identities=ids)
    first=dict(gpus=gpu,applications=apps+[dict(uuid='u'+str(i),pid=99,used_mib=7000) for i in range(8)],identities=ids)
    final=dict(first,gpus=[dict(g,sm=final_device_sm) for g in gpu])
    seq=iter([first,final]);monkeypatch.setattr(c,'nvml_snapshot',lambda:next(seq))
    monkeypatch.setattr(c.subprocess,'check_output',lambda *a,**kw:'original pmon -')
    monkeypatch.setattr(backend,'lease_ownership',lambda r:None)
    monkeypatch.setenv('SEARCH_R1_GPU_PREFLIGHT','1')
    monkeypatch.setenv('SEARCH_R1_GPU_LEASES',c.json.dumps([dict(uuid='u'+str(i)) for i in range(8)]))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES',','.join(map(str,range(8))))
    windows={'u'+str(i):dict(status='SUCCESS',start_us=now-1200000,end_us=now,
                  samples=[dict(pid=99,timestamp_us=now-200000,sm=90)]) for i in range(8)}
    monkeypatch.setattr(c,'collect_process_windows',lambda *a:(windows,now))
    result=c.active_gate(dict(baseline=base,retriever_identity=ids['99'],cgroup='/ours'))
    assert set(result['foreign_sm_aggregate_upper_bounds'].values())=={expected_bound}
    assert result['foreign_sm_evidence']['u0']['initial_foreign_upper_bound']==expected_bound
