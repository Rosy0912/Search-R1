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
 return [dict(event='PHASE_STARTED',phase=1,accepted=False),dict(accepted=True,captures=[copy.deepcopy(cap),copy.deepcopy(cap)],observed_foreign_sm_upper_bounds={'0':7},observed_foreign_sm_evidence={'0':ev})]+tail_trace(cap)

def test_real_unequal_record_scope_without_semantic_retry_requirement():
 r=module('window_cover','probe.py').coverage(trace(),{str(i) for i in range(8)},{20,30},require_no_sample=False)
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
 else:t[0]['phase']=2
 with pytest.raises((RuntimeError,KeyError)):module('window_bad','probe.py').coverage(t,{str(i) for i in range(8)},{20,30},require_no_sample=False)

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


def tail_trace(cap):
 first=dict(copy.deepcopy(cap),epoch=2.2)
 post=dict(copy.deepcopy(cap),epoch=2.3,snapshot_role='POST_FAILURE_IDENTITY_CLOSE_NOT_ORIGINAL_FINAL')
 w=dict(status='SUCCESS',start_us=1000000,end_us=2200000,samples=[dict(pid=20,sm=0,timestamp_us=2100000)])
 rejected=dict(error='No positive known-own sample proves active process sampler',accepted=False,retry_authorized=True,retry_kind='FRESH_SUPPORTED_ZERO_OWN_MEASUREMENT',foreign_sm_status='UNKNOWN_REJECTED_TRANSITION',attempt=1,epoch=2.4,captures=[first,post],original_windows=[dict(query_completed_us=2200000,epoch=2.2,snapshot=copy.deepcopy(cap['raw_snapshot']),foreign=[[str(i),10] for i in range(8)],windows={str(i):copy.deepcopy(w) for i in range(8)})])
 fresh=dict(validator='PINNED_MULTI_PID_WINDOW_FULL_SM_GATE',observed_foreign_sm_evidence={str(i):dict(method='DEVICE_TOTAL_AGGREGATE_UPPER_BOUND',upper_bound=0,initial_foreign_upper_bound=0,final_foreign_upper_bound=0,initial_evidence=dict(method='DEVICE_TOTAL_AGGREGATE_UPPER_BOUND',upper_bound=0)) for i in range(8)},accepted=True,attempt=2,epoch=2.5,captures=[copy.deepcopy(first),copy.deepcopy(post)],fresh_measurement_uuids=[str(i) for i in range(8)],observed_foreign_sm_upper_bounds={str(i):0 for i in range(8)})
 for c in fresh['captures']:
  for gpu in c['raw_snapshot']['gpus']:gpu['sm']=0
 return [dict(event='PHASE_STARTED',phase=2,accepted=False),rejected,fresh]

@pytest.mark.parametrize('fault',['no_rejection','not_authorized','no_fresh','identity','unsupported','foreign','stale','false_zero','missingtarget'])
def test_actual_zero_tail_coverage_cannot_be_skipped(fault):
 t=trace();r=t[-2];f=t[-1]
 if fault=='no_rejection':r['error']='other'
 elif fault=='not_authorized':r['retry_authorized']=False
 elif fault=='no_fresh':f['accepted']=False
 elif fault=='identity':f['captures'][0]['uids']['20']=5
 elif fault=='unsupported':r['original_windows'][0]['windows']['7']['status']='NVML_ERROR_3'
 elif fault=='foreign':r['original_windows'][0]['windows']['7']['samples'].append(dict(pid=10,sm=16,timestamp_us=2100000))
 elif fault=='stale':r['original_windows'][0]['windows']['7']['samples'][0]['timestamp_us']=500000
 elif fault=='false_zero':
  for w in r['original_windows'][0]['windows'].values():w['samples'][0]['sm']=1
 else:f['fresh_measurement_uuids']=[]
 with pytest.raises(RuntimeError):module('tail_negative','probe.py').coverage(t,{str(i) for i in range(8)},{20,30},require_no_sample=False)


def test_fresh_supported_initial_high_final_low_is_legal():
 t=trace();fresh=t[-1];fresh['captures'][0]['raw_snapshot']['gpus'][0]['sm']=93
 source=copy.deepcopy(t[1]['observed_foreign_sm_evidence']['0'])
 ev=fresh['observed_foreign_sm_evidence']['0'];ev.update(initial_evidence=source,initial_foreign_upper_bound=7,upper_bound=7)
 fresh['observed_foreign_sm_upper_bounds']['0']=7
 assert module('high_to_low','probe.py').coverage(t,{str(i) for i in range(8)},{20,30},require_no_sample=False)['actual_zero_own_remeasurements']
 source['window']['samples'][0]['sm']=0;source['window']['samples'][1]['sm']=0
 with pytest.raises(RuntimeError):module('high_to_low_bad','probe.py').coverage(t,{str(i) for i in range(8)},{20,30},require_no_sample=False)

def test_parent_worker_duration_order_matches_actual_done_contract():
 for name in ('worker.py','probe.py'):
  tree=ast.parse((HERE/name).read_text())
  tuples=[tuple(n.value for n in node.elts) for node in ast.walk(tree) if isinstance(node,ast.Tuple) and all(isinstance(n,ast.Constant) and isinstance(n.value,(float,int)) for n in node.elts)]
  assert (16.,.25,2.) in tuples


def test_forged_supported_branch_on_actually_low_capture_refused():
 t=trace();fresh=t[-1];ev=fresh['observed_foreign_sm_evidence']['0']
 ev['initial_evidence']=copy.deepcopy(t[1]['observed_foreign_sm_evidence']['0'])
 ev.update(initial_foreign_upper_bound=7,upper_bound=7)
 fresh['observed_foreign_sm_upper_bounds']['0']=7
 with pytest.raises(RuntimeError,match='unsupported'):module('false_supported_low','probe.py').coverage(t,{str(i) for i in range(8)},{20,30},require_no_sample=False)

@pytest.mark.parametrize('case',['good','no_actual_notfound','not_supported','unexpected_rows','foreign_late','no_fresh','false_initial_zero','changed'])
def test_actual_notfound_unknown_then_fresh_coverage(case):
 t=trace();r=t[-2];f=t[-1]
 r.update(retry_kind='FRESH_NO_SAMPLE_AFTER_CURSOR_MEASUREMENT',foreign_sm_status='UNKNOWN_NO_SAMPLE_AFTER_CURSOR',no_sample_after_cursor_uuids=['0'])
 w=r['original_windows'][0]['windows']['0'];w.update(status='NVML_ERROR_6',samples=[])
 if case=='no_actual_notfound':w.update(status='SUCCESS',samples=[dict(pid=20,sm=0,timestamp_us=2100000)])
 elif case=='not_supported':w['status']='NVML_ERROR_3'
 elif case=='unexpected_rows':w['samples']=[dict(pid=20,sm=0,timestamp_us=2100000)]
 elif case=='foreign_late':r['original_windows'][0]['windows']['7']['samples'].append(dict(pid=10,sm=16,timestamp_us=2100000))
 elif case=='no_fresh':f['accepted']=False
 elif case=='false_initial_zero':r['foreign_sm_status']='MEASURED_ZERO'
 elif case=='changed':f['captures'][1]['raw_snapshot']['identities'].pop('30')
 m=module('notfound_cover','probe.py')
 if case=='good':
  facts=m.coverage(t,{str(i) for i in range(8)},{20,30})
  assert facts['actual_not_found_cursor_remeasurement_covered'] is True
  assert facts['actual_zero_own_remeasurements'][0]['rejected']['accepted'] is False
 else:
  with pytest.raises(RuntimeError):m.coverage(t,{str(i) for i in range(8)},{20,30})


def test_post_actual_done_calls_real_full_gate_without_altered_snapshot():
 tree=ast.parse((HERE/'probe.py').read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
 loop=next(n for n in ast.walk(main) if isinstance(n,ast.For) and isinstance(n.iter,ast.Call) and isinstance(n.iter.func,ast.Name) and n.iter.func.id=='enumerate')
 text=ast.unparse(loop)
 assert text.index("'Both owners all8 release failed'") < text.index("'WORKERS_DONE_POST_COMPLETION_GATE'") < text.rindex('records.append(guard.active_gate(runtime))')
 assert 'actual_done_pids' in text
