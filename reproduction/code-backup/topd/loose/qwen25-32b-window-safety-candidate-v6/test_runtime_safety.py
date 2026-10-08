"""CPU evidence contracts; synthetic samples do not establish hardware support."""
import os
import sys
from pathlib import Path
import pytest
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(os.environ['AGENT_ROOT'])/'repo-terminal-fixed413' if os.environ.get('AGENT_ROOT') else HERE.parent/'topd-terminal-fixed413'))
sys.path.insert(0, str(HERE))
sys.path.append(str(HERE.parent/'qwen25-32b-terminal-fix-refreeze'))
import runtime_safety as c

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
    assert ev['u']['absent_interpretation']=='NONZERO_FILTER_WINDOW_INFERENCE_NOT_MEASURED'
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
    [dict(pid=2,timestamp_us=2000000,sm=float('nan'))]])
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


def test_asynchronous_own_and_foreign_records_conservative_maxima():
    rows=[dict(pid=2,timestamp_us=1800000,sm=80),dict(pid=1,timestamp_us=1900000,sm=3),dict(pid=1,timestamp_us=2100000,sm=7)]
    bounds,ev=run(rows=rows)
    assert bounds=={'u':7} and ev['u']['foreign_observed_maxima']=={1:7}
    assert ev['u']['timestamp_span_us']==300000
    assert ev['u']['aggregate_scope']=='SUM_OF_PER_FOREIGN_PID_MAXIMA_IN_RETURNED_RECORDS_NOT_CONTINUOUS_BOUND'
    assert ev['u']['window']['samples'] is rows

def test_distinct_foreign_maxima_sum_breach_never_averaged():
    foreign=F|{('u',3)}
    rows=[dict(pid=2,timestamp_us=1800000,sm=80),dict(pid=1,timestamp_us=1900000,sm=9),dict(pid=3,timestamp_us=2100000,sm=8)]
    bounds,ev=c.sm_aggregate_bounds({'gpus':[dict(uuid='u',sm=80)]},foreign,O,
        {'u':dict(status='SUCCESS',start_us=1000000,end_us=2200000,samples=rows)},2200000)
    assert bounds['u']==17 and ev['u']['foreign_observed_maxima']=={1:9,3:8}

def test_final_clock_is_not_original_query_completion():
    bounds,ev=c.sm_aggregate_bounds({'gpus':[dict(uuid='u',sm=80)]},F,O,
        {'u':dict(status='SUCCESS',start_us=1000000,end_us=2200000,samples=[dict(pid=2,timestamp_us=2000000,sm=80)])},2400000,
        actual_query_completed_us=2200000)
    assert ev['u']['query_completed_us']==2200000 and ev['u']['evaluation_now_us']==2400000
    with pytest.raises(RuntimeError):
        c.sm_aggregate_bounds({'gpus':[dict(uuid='u',sm=80)]},F,O,
        {'u':dict(status='SUCCESS',start_us=1000000,end_us=2200000,samples=[dict(pid=2,timestamp_us=2000000,sm=80)])},3600000,
        actual_query_completed_us=2200000)

def test_actual3056_original_unmodified_windows_replay():
    fixture=c.json.loads((HERE/'actual3056-window.json').read_text());rec=fixture['record']
    foreign={tuple(v) for v in rec['foreign']};own={(g['uuid'],pid) for g in rec['snapshot']['gpus'] for pid in fixture['own_pids']}
    bound,ev=c.sm_aggregate_bounds(rec['snapshot'],foreign,own,rec['windows'],rec['query_completed_us'])
    high=[g['uuid'] for g in rec['snapshot']['gpus'] if g['sm']>15]
    assert high and all(bound[u]==0 for u in high)
    assert any(ev[u]['timestamp_span_us']>0 for u in high)
    assert all(ev[u]['window'] is rec['windows'][u] for u in high)
    assert fixture['scope']=='ACTUAL_3056_FAILED_C7_RAW_REPLAY_NOT_NEW_GPU_PROOF'


@pytest.mark.parametrize('bad',['nan','inf','negative','too_high','future_query','early_query'])
def test_invalid_numeric_or_query_provenance_rejected(bad):
    sample=dict(pid=2,timestamp_us=2000000,sm=80)
    query=2200000
    if bad in ('nan','inf'):sample['sm']=float(bad)
    elif bad=='negative':sample['sm']=-1
    elif bad=='too_high':sample['sm']=101
    elif bad=='future_query':query=2200001
    else:query=2199999
    with pytest.raises(RuntimeError):
        c.sm_aggregate_bounds({'gpus':[dict(uuid='u',sm=80)]},F,O,
        {'u':dict(status='SUCCESS',start_us=1000000,end_us=2200000,samples=[sample])},2200000,
        actual_query_completed_us=query)


@pytest.mark.parametrize('scenario',['foreign_breach','initial14_final_low','final_stale'])
def test_real_active_gate_asynchronous_bounds_and_current_final_clock(monkeypatch,scenario):
    import time
    import search_r1.baselines.topd_local_gpu as backend
    now=int(time.time()*1000000)
    gpu=[dict(index=str(i),uuid='u'+str(i),used_mib=9000,total_mib=80000,sm=90) for i in range(8)]
    ids={str(pid):dict(pid=pid,starttime_ticks=1,cgroup=['0::/foreign']) for pid in (1,3)}
    ids['99']=dict(pid=99,starttime_ticks=1,cgroup=['0::/ours'])
    apps=[dict(uuid=g['uuid'],pid=pid,used_mib=1000) for g in gpu for pid in (1,3)]
    base=dict(gpus=gpu,applications=apps,identities=ids)
    first=dict(gpus=gpu,applications=apps+[dict(uuid=g['uuid'],pid=99,used_mib=7000) for g in gpu],identities=ids)
    final=dict(first,gpus=[dict(g,sm=0 if scenario=='initial14_final_low' else 90) for g in gpu])
    seq=iter([first,final]);monkeypatch.setattr(c,'nvml_snapshot',lambda:next(seq))
    monkeypatch.setattr(c.subprocess,'check_output',lambda *a,**kw:'raw pmon -')
    monkeypatch.setattr(backend,'lease_ownership',lambda r:None)
    monkeypatch.setenv('SEARCH_R1_GPU_PREFLIGHT','1');monkeypatch.setenv('SEARCH_R1_GPU_LEASES',c.json.dumps([dict(uuid=g['uuid']) for g in gpu]))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES',','.join(map(str,range(8))))
    value=10 if scenario=='foreign_breach' else 7
    windows={g['uuid']:dict(status='SUCCESS',start_us=now-1200000,end_us=now,
        samples=[dict(pid=99,timestamp_us=now-100000,sm=90),dict(pid=1,timestamp_us=now-200000,sm=value),dict(pid=3,timestamp_us=now-300000,sm=value)]) for g in gpu}
    monkeypatch.setattr(c,'collect_process_windows',lambda *a:(windows,now))
    if scenario=='final_stale':monkeypatch.setattr(time,'time',lambda:(now+2000000)/1e6)
    runtime=dict(baseline=base,retriever_identity=ids['99'],cgroup='/ours')
    if scenario!='initial14_final_low':
        with pytest.raises(RuntimeError,match='Foreign aggregate SM|Stale/future'):c.active_gate(runtime)
    else:
        record=c.active_gate(runtime)
        assert set(record['foreign_sm_aggregate_upper_bounds'].values())=={14}
        assert record['foreign_sm_evidence']['u0']['initial_foreign_upper_bound']==14
        assert record['foreign_sm_evidence']['u0']['final_foreign_upper_bound']==0


def test_actual_collector_wait_one_second_complete_return(monkeypatch):
    import time,types
    clock=[10.0];waits=[];cursors=[];finalized=[]
    def sleep(seconds):waits.append(seconds);clock[0]+=seconds
    monkeypatch.setattr(time,'time',lambda:clock[0]);monkeypatch.setattr(time,'sleep',sleep)
    sample=types.SimpleNamespace(pid=2,timeStamp=10010000,smUtil=80)
    def get(handle,cursor):cursors.append(cursor);return [sample]
    nv=types.SimpleNamespace(nvmlInit=lambda:None,nvmlShutdown=lambda:finalized.append(True),
        nvmlDeviceGetHandleByUUID=lambda uid:uid,nvmlDeviceGetProcessUtilization=get,NVMLError=Exception)
    monkeypatch.setitem(sys.modules,'pynvml',nv)
    windows,query=c.collect_process_windows({'gpus':[dict(uuid='u',sm=100)]},F)
    assert waits==[1.0] and cursors==[10000000] and finalized==[True]
    assert windows['u']==dict(start_us=10000000,end_us=11000000,query_start_us=11000000,query_end_us=11000000,status='SUCCESS',samples=[dict(pid=2,timestamp_us=10010000,sm=80.)])
    assert query==11000000
    # Actual elapsed source clock, not an amended timestamp or filtered old row.
    bounds,_=c.sm_aggregate_bounds({'gpus':[dict(uuid='u',sm=100)]},F,O,windows,query+388409,actual_query_completed_us=query)
    assert bounds['u']==0


def test_old_3071_query_fresh_final_stale_still_rejected():
    rows=[dict(pid=2,timestamp_us=10010281,sm=100)]
    w={'u':dict(status='SUCCESS',start_us=10000000,end_us=11220000,samples=rows)}
    c.sm_aggregate_bounds({'gpus':[dict(uuid='u',sm=100)]},F,O,w,11220000)
    with pytest.raises(RuntimeError,match='Stale/future'):
        c.sm_aggregate_bounds({'gpus':[dict(uuid='u',sm=100)]},F,O,w,11220000+388409,actual_query_completed_us=11220000)
