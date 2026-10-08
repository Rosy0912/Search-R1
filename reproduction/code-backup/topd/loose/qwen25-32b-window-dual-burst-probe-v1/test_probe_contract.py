import ast,importlib.util,json,copy
from pathlib import Path
import pytest
HERE=Path(__file__).parent

def module(name,file):
 spec=importlib.util.spec_from_file_location(name,HERE/file);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def trace():
 ids={str(p):dict(pid=p,starttime_ticks=p,cgroup=['0::/foreign' if p==10 else '0::/job']) for p in (10,20,30)}
 cap=dict(raw_snapshot=dict(identities=ids,gpus=[dict(uuid=str(i),sm=93) for i in range(8)],applications=[dict(uuid=str(i),pid=p) for i in range(8) for p in (10,20,30)]),uids={str(p):0 for p in (10,20,30)},uid_errors={})
 w=dict(status='SUCCESS',start_us=1000000,end_us=2200000,samples=[dict(pid=20,sm=55,timestamp_us=1800000),dict(pid=30,sm=45,timestamp_us=2000000),dict(pid=10,sm=7,timestamp_us=2100000)])
 ev=dict(upper_bound=7,method='FRESH_SUPPORTED_NONZERO_PROCESS_WINDOW',positive_own_pids=[20,30],window=w,query_completed_us=2200000,evaluation_now_us=2300000,timestamp_min_us=1800000,timestamp_max_us=2100000,timestamp_span_us=300000,foreign_observed_maxima={10:7},aggregate_scope='SUM_OF_PER_FOREIGN_PID_MAXIMA_IN_RETURNED_RECORDS_NOT_CONTINUOUS_BOUND',absent_interpretation='NONZERO_FILTER_WINDOW_INFERENCE_NOT_MEASURED')
 return [dict(event='PHASE_STARTED',phase=3,accepted=False),dict(accepted=True,captures=[copy.deepcopy(cap),copy.deepcopy(cap)],observed_foreign_sm_upper_bounds={'0':7},observed_foreign_sm_evidence={'0':ev})]

def test_real_unequal_record_scope_without_semantic_retry_requirement():
 r=module('window_cover','probe.py').coverage(trace(),{str(i) for i in range(8)},{20,30})
 assert len(r['actual_high_unequal_dual_positive_windows'])==1
 assert r['scope']=='OBSERVED_RETURNED_PROCESS_RECORD_MAXIMA_NOT_CONTINUOUS_SM_GUARANTEE'

@pytest.mark.parametrize('fault',['not_accepted','same_ts','single_owner','low_tail','no_all8_high','foreign_breach','forged_max','uid','changed','stale','unknown','span','unsupported','end_after_query','short_phase_only'])
def test_incomplete_or_false_window_coverage_refused(fault):
 t=trace();row=t[1];ev=row['observed_foreign_sm_evidence']['0'];w=ev['window']
 if fault=='not_accepted':row['accepted']=False
 elif fault=='same_ts':
  for r in w['samples']:r['timestamp_us']=2000000
 elif fault=='single_owner':w['samples'][1]['sm']=0
 elif fault=='low_tail':ev['method']='DEVICE_TOTAL_AGGREGATE_UPPER_BOUND'
 elif fault=='no_all8_high':
  for cap in row['captures']:cap['raw_snapshot']['gpus'][0]['sm']=0
 elif fault=='foreign_breach':w['samples'][2]['sm']=16
 elif fault=='forged_max':ev['foreign_observed_maxima']={10:1}
 elif fault=='uid':row['captures'][1]['uids']['20']=1
 elif fault=='changed':row['captures'][1]['raw_snapshot']['identities']['40']={'pid':40}
 elif fault=='stale':ev['evaluation_now_us']=4000000
 elif fault=='unknown':w['samples'][2]['pid']=11
 elif fault=='span':ev['timestamp_span_us']=1
 elif fault=='unsupported':w['status']='NVML_ERROR_3'
 elif fault=='end_after_query':ev['query_completed_us']=2199999
 else:t[0]['phase']=1
 with pytest.raises((RuntimeError,KeyError)):module('window_bad','probe.py').coverage(t,{str(i) for i in range(8)},{20,30})

def test_actual_reading_returns_same_object_then_two_burst_commands(tmp_path):
 for n in ('owner-0','owner-1'):(tmp_path/n).mkdir()
 sample={'actual':7};seen=[]
 def read():seen.append((tmp_path/'owner-0/command-1').exists());return sample
 fn=module('window_trigger','probe.py').trigger_once_after_snapshot(read,tmp_path,1)
 assert fn() is sample and fn() is sample and seen==[False,True]
 assert all((tmp_path/n/'command-1').exists() for n in ('owner-0','owner-1'))
 assert json.loads((tmp_path/'trigger-1.json').read_text())['actual_snapshot']==sample

def test_atomic_worker_json_and_parent_no_model_load(tmp_path):
 m=module('window_worker','worker.py');m.write(tmp_path/'r.json',dict(pid=4));assert json.loads((tmp_path/'r.json').read_text())['pid']==4
 tree=ast.parse((HERE/'probe.py').read_text())
 assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('from_pretrained','LLM','load_model') for n in ast.walk(tree))


@pytest.mark.parametrize('fault',['clean','source_drift','cached_shadow'])
def test_explicit_frozen_cpu_utility_import_contract(tmp_path,monkeypatch,fault):
    import sys,types,hashlib
    path=tmp_path/'eval64.py';path.write_text('def require(value,message):\n    if not value:raise RuntimeError(message)\n')
    digest=hashlib.sha256(path.read_bytes()).hexdigest();probe=module('utility_contract','probe.py')
    monkeypatch.delitem(sys.modules,'eval64',raising=False)
    if fault=='source_drift':path.write_text('raise RuntimeError("must never execute drifted source")')
    elif fault=='cached_shadow':monkeypatch.setitem(sys.modules,'eval64',types.SimpleNamespace(__file__='/foreign/eval64.py'))
    if fault=='clean':
        imported=probe.verified_eval64(path,digest)
        assert Path(imported.__file__).resolve()==path.resolve()
        with pytest.raises(RuntimeError,match='real helper'):imported.require(False,'real helper')
    else:
        with pytest.raises(RuntimeError):probe.verified_eval64(path,digest)
