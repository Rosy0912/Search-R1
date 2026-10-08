import ast,importlib.util,json,copy
from pathlib import Path
import pytest
HERE=Path(__file__).parent

def module(name,file):
 spec=importlib.util.spec_from_file_location(name,HERE/file);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def trace():
 ids={'20':dict(pid=20,starttime_ticks=7,cgroup=['0::/job']),'30':dict(pid=30,starttime_ticks=8,cgroup=['0::/job'])}
 cap=dict(raw_snapshot=dict(identities=ids,gpus=[dict(uuid=str(i),sm=93) for i in range(8)]),uids={'20':0,'30':0})
 closure=copy.deepcopy(cap);closure['capture_scope']='POST_FAILURE_IDENTITY_CLOSE_NOT_C7_FINAL'
 return [dict(attempt=1,accepted=False,error='Process records span different completed sampling periods',retry_authorized=True,
   foreign_sm_status='UNKNOWN_REJECTED_MIXED_PERIOD',retry_kind='FRESH_MIXED_PERIOD_MEASUREMENT',captures=[copy.deepcopy(cap),closure],
   original_windows=[dict(windows={'0':dict(status='SUCCESS',samples=[dict(pid=20,sm=55,timestamp_us=1),dict(pid=30,sm=45,timestamp_us=2)])})]),
   dict(attempt=2,accepted=True,captures=[copy.deepcopy(cap),copy.deepcopy(cap)],fresh_measurement_uuids=['0'],fresh_foreign_sm_upper_bounds={'0':0},
    fresh_foreign_sm_evidence={'0':dict(upper_bound=0,method='FRESH_SUPPORTED_NONZERO_PROCESS_WINDOW',positive_own_pids=[20,30],window=dict(status='SUCCESS',samples=[dict(pid=20,sm=55,timestamp_us=3),dict(pid=30,sm=45,timestamp_us=3)]))})]

def test_both_true_original_pid_mixed_then_fresh_evidence_required():
 r=module('dual_cover','probe.py').coverage(trace(),{str(i) for i in range(8)},{20,30})
 assert r['typed_mixed_retries_observed']==1 and len(r['dual_positive_mixed_windows'])==1

@pytest.mark.parametrize('kind',['low_high_only','single_positive','same_period','no_closure','changed','no_high','badbound','no_success','low_tail_only','single_fresh_owner','fresh_mixed'])
def test_missing_real_mixed_or_dangerous_acceptance_never_coverage(kind):
 t=trace()
 if kind=='low_high_only':t[0]['error']='Device SM increased after upper-bound sample'
 elif kind=='single_positive':t[0]['original_windows'][0]['windows']['0']['samples'][1]['sm']=0
 elif kind=='same_period':t[0]['original_windows'][0]['windows']['0']['samples'][1]['timestamp_us']=1
 elif kind=='no_closure':t[0]['captures'][1].pop('capture_scope')
 elif kind=='changed':t[1]['captures'][0]['raw_snapshot']['identities']['40']={'pid':40}
 elif kind=='no_high':
  for r in t:
   for c in r['captures']:c['raw_snapshot']['gpus'][0]['sm']=0
 elif kind=='badbound':t[1]['fresh_foreign_sm_upper_bounds']['0']=16
 elif kind=='low_tail_only':t[1]['fresh_foreign_sm_evidence']['0']['method']='DEVICE_TOTAL_AGGREGATE_UPPER_BOUND'
 elif kind=='single_fresh_owner':t[1]['fresh_foreign_sm_evidence']['0']['positive_own_pids']=[20]
 elif kind=='fresh_mixed':t[1]['fresh_foreign_sm_evidence']['0']['window']['samples'][1]['timestamp_us']=4
 else:t.pop()
 with pytest.raises((RuntimeError,KeyError)):module('dual_bad','probe.py').coverage(t,{str(i) for i in range(8)},{20,30})

def test_actual_reading_returns_same_object_then_two_burst_commands(tmp_path):
 for n in ('owner-0','owner-1'):(tmp_path/n).mkdir()
 sample={'actual':7};seen=[]
 def read():seen.append((tmp_path/'owner-0/command-1').exists());return sample
 fn=module('dual_trigger','probe.py').trigger_once_after_snapshot(read,tmp_path,1)
 assert fn() is sample and fn() is sample and seen==[False,True]
 assert all((tmp_path/n/'command-1').exists() for n in ('owner-0','owner-1'))
 assert json.loads((tmp_path/'trigger-1.json').read_text())['actual_snapshot']==sample

def test_atomic_worker_json_and_parent_no_model_load(tmp_path):
 m=module('dual_worker','worker.py');m.write(tmp_path/'r.json',dict(pid=4));assert json.loads((tmp_path/'r.json').read_text())['pid']==4
 tree=ast.parse((HERE/'probe.py').read_text())
 assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('from_pretrained','LLM','load_model') for n in ast.walk(tree))
