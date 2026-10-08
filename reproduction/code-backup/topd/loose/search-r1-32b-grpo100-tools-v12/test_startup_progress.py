import copy,json,pathlib
import pytest
import retrieval_startup as startup
HERE=pathlib.Path(__file__).resolve().parent
RAW=json.loads((HERE/'actual3176-startup.fixture.jsonl').read_text().splitlines()[0])
HISTORY=RAW['raw_retry_history']
SERVICE=HISTORY[1]['raw_failed_measurement']['initial']['identities']['3117022']
UUIDS=[g['uuid'] for g in HISTORY[1]['raw_failed_measurement']['initial']['gpus']]
def error(raw):return startup.KnownOwnAttachTransition('actual qualification requires next fullfresh',copy.deepcopy(raw))
def test_actual3176_six_unknowns_require_seventh_complete_measurement():
 calls=[]
 def call():
  calls.append(len(calls))
  if len(calls)<=6:raise error(HISTORY[len(calls)-1]['raw_failed_measurement'])
  return {'scope':'CPU_SCHEDULE_ONLY_COMPLETE_ORIGINAL_SAMPLE_CALL_REQUIRED'}
 with pytest.raises(startup.KnownOwnAttachTransition):startup.bounded_sample(call,pause=0)
 calls.clear();result=startup.bounded_sample(call,pause=0,progress_service=SERVICE,progress_uuids=UUIDS)
 assert len(calls)==7 and len(result['known_own_attach_retry_history'])==6
 assert sum(h['retry_kind']=='INITIALIZATION_ONLY_VERIFIED_UUID_PROGRESS' for h in result['known_own_attach_retry_history'])==4
 assert result['known_own_attach_retry_history'][-1]['raw_failed_measurement']==HISTORY[-1]['raw_failed_measurement']
@pytest.mark.parametrize('change',['foreign_bound','identity','uid','cgroup','lease','nonmonotonic','foreignmemory'])
def test_original_qualification_progress_rejects_actual_negative(change):
 raw=copy.deepcopy(HISTORY[1]['raw_failed_measurement']);ids=list(UUIDS)
 if change=='foreign_bound':raw['initial_bounds'][UUIDS[0]]=16.
 if change=='identity':raw['final']['identities']['3117022']['starttime_ticks']+=1
 if change=='uid':raw['final']['identities']['3117022']['uid']=999
 if change=='cgroup':raw['final']['identities']['3117022']['cgroup']=['1:freezer:/foreign']
 if change=='lease':ids[0]='FOREIGN'
 if change=='nonmonotonic':raw['final_own_pid_uuids']=raw['initial_own_pid_uuids']
 if change=='foreignmemory':raw['final']['applications'][0]['used_mib']=80000.
 with pytest.raises(RuntimeError):startup.ownership_progress(error(raw),SERVICE,ids)
def test_no_progress_zero_unknown_keeps_original_six_budget():
 count=[]
 def call():count.append(1);raise error(HISTORY[3]['raw_failed_measurement'])
 with pytest.raises(startup.KnownOwnAttachTransition):startup.bounded_sample(call,pause=0,progress_service=SERVICE,progress_uuids=UUIDS)
 assert len(count)==6
def test_duplicate_progress_rejected_without_acceptance():
 count=[]
 def call():count.append(1);raise error(HISTORY[1]['raw_failed_measurement'])
 with pytest.raises(RuntimeError,match='regressed|repeated'):startup.bounded_sample(call,pause=0,progress_service=SERVICE,progress_uuids=UUIDS)
 assert len(count)==2
def test_fixed_thirty_second_budget_never_resets(monkeypatch):
 clock=[0.];monkeypatch.setattr(startup.time,'monotonic',lambda:clock[0]);calls=[]
 def call():
  calls.append(1);clock[0]+=20
  raise error(HISTORY[1 if len(calls)==1 else 2]['raw_failed_measurement'])
 with pytest.raises(startup.KnownOwnAttachTransition):startup.bounded_sample(call,pause=0,progress_service=SERVICE,progress_uuids=UUIDS)
 assert len(calls)==2 and clock[0]==40
def test_unknown_nvml_or_foreign_exception_never_gets_progress_retry():
 count=[]
 def call():count.append(1);raise RuntimeError('Unsupported NVML empty / foreign process')
 with pytest.raises(RuntimeError,match='Unsupported'):startup.bounded_sample(call,pause=0,progress_service=SERVICE,progress_uuids=UUIDS)
 assert len(count)==1

def test_actual_guardian_progress_only_in_starting_branch(tmp_path,monkeypatch):
 import threading,types
 from window_guardian import Guardian
 g=Guardian.__new__(Guardian);g.lock=threading.RLock();g.runtime={'retriever_identity':SERVICE,'cgroup':'job'};g.starting_retriever=True;g.evidence=tmp_path;g.current_phase='retriever-starting';g.baseline={};g.safety=object();g.guard=types.SimpleNamespace(lease_reader=lambda:UUIDS);seen=[]
 def wrapper(call,**kwargs):seen.append(kwargs);return {'scope':'CPU_CALLER_WIRING_ONLY'}
 monkeypatch.setattr(startup,'bounded_sample',wrapper)
 g.sample();assert seen==[{'progress_service':SERVICE,'progress_uuids':UUIDS}]
 g.starting_retriever=False;g.guard.active_gate=lambda runtime,pre_model: {'scope':'ORIGINAL_ACTIVE_GATE'}
 assert g.sample()['scope']=='ORIGINAL_ACTIVE_GATE' and len(seen)==1
