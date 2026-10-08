import ast,copy,pathlib,sys,types,os,time
import pytest
sys.path.insert(0,str(pathlib.Path(__file__).parent))
import retrieval_startup as r
SOURCE=pathlib.Path(__file__).parents[1]/'qwen25-32b-window-safety-candidate-v4/runtime_safety.py'
def setup(monkeypatch,partial=False):
 require_source=SOURCE.parent.parent/'qwen25-32b-terminal-fix-refreeze/eval64.py'
 require_node=next(n for n in ast.parse(require_source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='require')
 req={};exec(compile(ast.Module(body=[require_node],type_ignores=[]),'original-eval-require','exec'),req)
 need=req['require'];ns={'require':need,'math':__import__('math')}
 node=next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='sm_aggregate_bounds');exec(compile(ast.Module(body=[node],type_ignores=[]),'original-v4-sm','exec'),ns)
 foreign={'pid':3,'starttime_ticks':7,'cgroup':['0::/foreign']};service={'pid':5,'starttime_ticks':9,'cgroup':['0::/own']}
 gpus=[dict(index=str(i),uuid='uuid'+str(i),used_mib=10.,total_mib=100.,sm=6.) for i in range(8)]
 apps=[dict(uuid='uuid0',pid=3,used_mib=1.)]
 baseline=dict(gpus=copy.deepcopy(gpus),applications=copy.deepcopy(apps),identities={'3':foreign})
 if partial:apps.append(dict(uuid='uuid0',pid=5,used_mib=4.))
 snap=dict(gpus=gpus,applications=apps,identities={'3':foreign,**({'5':service} if partial else {})})
 safety=types.SimpleNamespace(require=need,nvml_snapshot=lambda:copy.deepcopy(snap),pid_identity=lambda p:service if p==5 else foreign,collect_process_windows=lambda a,b:({},int(time.time()*1e6)),sm_aggregate_bounds=ns['sm_aggregate_bounds'])
 monkeypatch.setenv('CUDA_VISIBLE_DEVICES',','.join(str(i) for i in range(8)))
 return safety,baseline,snap,service,{g['uuid'] for g in gpus}
def sample(s,b,service,u):return r.sample(s,b,u,service,'/own',lease_reader=lambda:u,descendant_check=lambda p,root:p==root,uid_read=lambda p:os.getuid())
@pytest.mark.parametrize('partial',[False,True])
def test_original_sm_with_real_none_or_partial_retriever_not_fake_all8(monkeypatch,partial):
 s,b,snap,service,u=setup(monkeypatch,partial);out=sample(s,b,service,u)
 assert out['scope']=='STARTING_RETRIEVER_PARTIAL_RESIDENCY_NOT_ALL8_READY'
 assert len(out['own_pid_uuids'])==(1 if partial else 0) and out['foreign_sm_aggregate_upper_bounds']=={'uuid0':6.}

def test_actual_foreign_memory_or_identity_drift_reject(monkeypatch):
 s,b,snap,service,u=setup(monkeypatch);snap['applications'][0]['used_mib']=16.
 with pytest.raises(RuntimeError,match='memory exceeds'):sample(s,b,service,u)
 snap['applications'][0]['used_mib']=1.;snap['identities']['3']={'pid':3,'starttime_ticks':8,'cgroup':['0::/foreign']}
 with pytest.raises(RuntimeError,match='Foreign PID'):sample(s,b,service,u)

def test_unknown_gpu_pid_or_uid_never_startup_retry(monkeypatch):
 s,b,snap,service,u=setup(monkeypatch,True);snap['applications'].append(dict(uuid='uuid1',pid=8,used_mib=1));snap['identities']['8']=dict(pid=8,starttime_ticks=8,cgroup=['0::/other'])
 with pytest.raises(RuntimeError,match='Unknown/foreign'):sample(s,b,service,u)

def test_known_service_attach_requires_full_fresh_repeat(monkeypatch):
 s,b,snap,service,u=setup(monkeypatch)
 snapshots=[copy.deepcopy(snap)]
 attached=copy.deepcopy(snap);attached['applications'].append(dict(uuid='uuid0',pid=5,used_mib=4.));attached['identities']['5']=service
 snapshots.append(attached);s.nvml_snapshot=lambda:copy.deepcopy(snapshots.pop(0))
 with pytest.raises(r.KnownOwnAttachTransition):sample(s,b,service,u)
 calls=[]
 def retry():
  calls.append(1)
  if len(calls)==1:raise r.KnownOwnAttachTransition('real own attach')
  return {'raw_snapshot':attached}
 assert len(r.bounded_sample(retry,pause=0)['known_own_attach_retry_history'])==1 and len(calls)==2
 def fatal():raise RuntimeError('Foreign aggregate SM exceeds15% or unknown')
 with pytest.raises(RuntimeError,match='Foreign aggregate'):r.bounded_sample(fatal,pause=0)

def test_high_foreign_sm_uses_original_supported_window_and_rejects_breach(monkeypatch):
 s,b,snap,service,u=setup(monkeypatch,True);snap['gpus'][0]['sm']=90.
 def collect(a,foreign):
  now=int(time.time()*1e6);return {'uuid0':dict(status='SUCCESS',start_us=now-1200000,end_us=now,samples=[dict(pid=5,sm=70.,timestamp_us=now-100000),dict(pid=3,sm=16.,timestamp_us=now-200000)])},now
 s.collect_process_windows=collect
 with pytest.raises(RuntimeError,match='Foreign aggregate SM'):sample(s,b,service,u)

def test_actual_guardian_starts_before_http_health():
 src=pathlib.Path(r.__file__).with_name('supervise.py').read_text();block=src[src.index('guardian.prepare_retriever_start'):src.index("initial=root/'initial'")]
 assert block.index('prepare_retriever_start')<block.index("spawn('retriever'")<block.index('guardian.start()')<block.index('healthy(')<block.index('resident_retriever(')


def test_original_runtimeerror_unknown_sample_new_owned_pid_gets_actual_fresh_retry(monkeypatch):
 s,b,snap,service,u=setup(monkeypatch)
 snap['gpus'][0]['sm']=80.
 initial=copy.deepcopy(snap);final=copy.deepcopy(snap);final['applications'].append(dict(uuid='uuid0',pid=5,used_mib=4.));final['identities']['5']=service
 reads=[initial,final];s.nvml_snapshot=lambda:reads.pop(0)
 def collect(a,foreign):
  now=int(time.time()*1e6);return {'uuid0':dict(status='SUCCESS',start_us=now-1200000,end_us=now,samples=[dict(pid=3,sm=6.,timestamp_us=now-200000),dict(pid=5,sm=70.,timestamp_us=now-100000)])},now
 s.collect_process_windows=collect
 with pytest.raises(r.KnownOwnAttachTransition,match='original process window'):sample(s,b,service,u)
 assert reads==[]

def test_completed_measurement_over_budget_not_accepted(monkeypatch):
 clock=iter([0.,31.]);monkeypatch.setattr(r.time,'monotonic',lambda:next(clock))
 with pytest.raises(TimeoutError,match='30s call budget'):r.bounded_sample(lambda:{'actual':'sample'},seconds=30)


@pytest.mark.parametrize('bad',['FOREIGN16','STALE','UNSUPPORTED'])
def test_original_unknown_newown_exception_does_not_hide_bad_window(monkeypatch,bad):
 s,b,snap,service,u=setup(monkeypatch);snap['gpus'][0]['sm']=80.
 initial=copy.deepcopy(snap);final=copy.deepcopy(snap);final['applications'].append(dict(uuid='uuid0',pid=5,used_mib=4.));final['identities']['5']=service
 reads=[initial,final];s.nvml_snapshot=lambda:reads.pop(0)
 def collect(a,foreign):
  now=int(time.time()*1e6);return {'uuid0':dict(status='NVML_ERROR_3' if bad=='UNSUPPORTED' else 'SUCCESS',start_us=now-1200000,end_us=now,samples=[dict(pid=5,sm=70.,timestamp_us=now-100000),dict(pid=3,sm=16. if bad=='FOREIGN16' else 6.,timestamp_us=now-3000000 if bad=='STALE' else now-200000)])},now
 s.collect_process_windows=collect
 with pytest.raises(RuntimeError) as e:sample(s,b,service,u)
 assert not isinstance(e.value,r.KnownOwnAttachTransition)

@pytest.mark.parametrize('foreign_sm',[6.,16.])
def test_partial_low_to_high_requires_real_supported_new_window(monkeypatch,foreign_sm):
 s,b,snap,service,u=setup(monkeypatch,True);low=copy.deepcopy(snap);high=copy.deepcopy(snap);high['gpus'][0]['sm']=80.
 reads=[low,high,high];s.nvml_snapshot=lambda:copy.deepcopy(reads.pop(0));calls=[]
 def collect(snapshot,foreign):
  now=int(time.time()*1e6);calls.append(snapshot['gpus'][0]['sm'])
  if snapshot['gpus'][0]['sm']<=15:return {},now
  return {'uuid0':dict(status='SUCCESS',start_us=now-1200000,end_us=now,samples=[dict(pid=5,sm=70.,timestamp_us=now-100000),dict(pid=3,sm=foreign_sm,timestamp_us=now-200000)])},now
 s.collect_process_windows=collect
 if foreign_sm<=15:
  with pytest.raises(r.KnownOwnAttachTransition) as e:sample(s,b,service,u)
  assert e.value.evidence['windows']['uuid0']['status']=='SUCCESS'
 else:
  with pytest.raises(RuntimeError) as e:sample(s,b,service,u)
  assert not isinstance(e.value,r.KnownOwnAttachTransition)
 assert calls==[6.,80.]

@pytest.mark.parametrize('kind',['ZERO_OWN','NOT_FOUND'])
def test_actual_zero_owned_original_helper_requires_new_complete_measurement(monkeypatch,kind):
 s,b,snap,service,u=setup(monkeypatch,True);snap['gpus'][0]['sm']=90.;calls=[]
 def collect(a,foreign):
  now=int(time.time()*1e6);calls.append(1)
  status='NVML_ERROR_6' if kind=='NOT_FOUND' and len(calls)==1 else 'SUCCESS'
  rows=[] if status!='SUCCESS' else [dict(pid=5,sm=0. if len(calls)==1 else 70.,timestamp_us=now-100000),dict(pid=3,sm=6.,timestamp_us=now-200000)]
  return {'uuid0':dict(status=status,start_us=now-1200000,end_us=now,samples=rows)},now
 s.collect_process_windows=collect
 out=r.bounded_sample(lambda:sample(s,b,service,u),pause=0)
 assert len(calls)==2 and out['foreign_sm_aggregate_upper_bounds']=={'uuid0':6.}
 history=out['known_own_attach_retry_history'];assert len(history)==1
 raw=history[0]['raw_failed_measurement'];assert raw['eligibility']['zero_own_targets']==['uuid0']
 assert raw['eligibility']['not_found_unknown_targets']==(['uuid0'] if kind=='NOT_FOUND' else [])
 assert out['windows']['uuid0']['status']=='SUCCESS'

@pytest.mark.parametrize('bad',['FOREIGN16','STALE','UNKNOWN_PID','UNSUPPORTED','EMPTY'])
def test_zero_own_retry_does_not_hide_original_fatal_window(monkeypatch,bad):
 s,b,snap,service,u=setup(monkeypatch,True);snap['gpus'][0]['sm']=90.
 def collect(a,foreign):
  now=int(time.time()*1e6)
  rows=[dict(pid=5,sm=0.,timestamp_us=now-100000),dict(pid=99 if bad=='UNKNOWN_PID' else 3,sm=16. if bad=='FOREIGN16' else 6.,timestamp_us=now-2100000 if bad=='STALE' else now-200000)]
  return {'uuid0':dict(status='NVML_ERROR_3' if bad=='UNSUPPORTED' else 'SUCCESS',start_us=now-1200000,end_us=now,samples=[] if bad=='EMPTY' else rows)},now
 s.collect_process_windows=collect
 with pytest.raises(RuntimeError) as e:r.bounded_sample(lambda:sample(s,b,service,u),pause=0)
 assert not isinstance(e.value,r.KnownOwnAttachTransition)
 assert e.value.retry_history==[]

@pytest.mark.parametrize('bad',['FOREIGN16','STALE'])
def test_earlier_zero_own_card_never_hides_later_card_foreign_violation(monkeypatch,bad):
 s,b,snap,service,u=setup(monkeypatch,True);snap['gpus'][0]['sm']=90.;snap['gpus'][1]['sm']=90.
 for obj in (b,snap):obj['applications'].append(dict(uuid='uuid1',pid=3,used_mib=1.))
 snap['applications'].append(dict(uuid='uuid1',pid=5,used_mib=4.))
 def collect(a,foreign):
  now=int(time.time()*1e6);windows={}
  for i in range(2):
   windows['uuid'+str(i)]=dict(status='SUCCESS',start_us=now-1200000,end_us=now,samples=[dict(pid=5,sm=0. if i==0 else 70.,timestamp_us=now-100000),dict(pid=3,sm=16. if i==1 and bad=='FOREIGN16' else 6.,timestamp_us=now-2100000 if i==1 and bad=='STALE' else now-200000)])
  return windows,now
 s.collect_process_windows=collect
 with pytest.raises(RuntimeError) as e:r.bounded_sample(lambda:sample(s,b,service,u),pause=0)
 assert not isinstance(e.value,r.KnownOwnAttachTransition)
 assert len(e.value.measurement_evidence['windows'])==2

def test_raw_original_windows_persisted_before_actual_helper_rejects(monkeypatch):
 s,b,snap,service,u=setup(monkeypatch,True);snap['gpus'][0]['sm']=90.;raw=[]
 def collect(a,foreign):
  now=int(time.time()*1e6);return {'uuid0':dict(status='NVML_ERROR_3',start_us=now-1200000,end_us=now,samples=[])},now
 s.collect_process_windows=collect
 with pytest.raises(RuntimeError):r.sample(s,b,u,service,'/own',lease_reader=lambda:u,descendant_check=lambda p,root:p==root,uid_read=lambda p:os.getuid(),raw_writer=raw.append)
 assert raw[0]['stage']=='initial-before-ownership-validation'
 assert raw[1]['actual']['windows']['uuid0']['status']=='NVML_ERROR_3'

def test_first_owned_cleanup_record_persisted_before_later_record_overwrite(tmp_path):
 import json,window_guardian
 guardian=window_guardian.Guardian.__new__(window_guardian.Guardian)
 guardian.evidence=tmp_path;guardian.owned=types.SimpleNamespace(cleanup_record={'signals':[{'pid':5,'signal':'TERM'}],'remaining':[]})
 guardian.record_cleanup()
 guardian.owned.cleanup_record={'signals':[],'remaining':[]}
 actual=json.loads((tmp_path/'owned-cleanup-history.jsonl').read_text())
 assert actual['actual']['signals']==[{'pid':5,'signal':'TERM'}]

@pytest.mark.parametrize('kind',['ZERO_OWN','NOT_FOUND'])
def test_actual_low_to_high_new_qualification_zero_owned_retries_full_sample(monkeypatch,kind):
 s,b,snap,service,u=setup(monkeypatch,True);high=copy.deepcopy(snap);high['gpus'][0]['sm']=90.;reads=[copy.deepcopy(snap),high,high,high,high]
 s.nvml_snapshot=lambda:copy.deepcopy(reads.pop(0));calls=[]
 def collect(snapshot,foreign):
  now=int(time.time()*1e6);calls.append(1)
  if snapshot['gpus'][0]['sm']<=15:return {},now
  status='NVML_ERROR_6' if kind=='NOT_FOUND' and len(calls)==2 else 'SUCCESS'
  rows=[] if status!='SUCCESS' else [dict(pid=5,sm=0. if len(calls)==2 else 70.,timestamp_us=now-100000),dict(pid=3,sm=6.,timestamp_us=now-200000)]
  return {'uuid0':dict(status=status,start_us=now-1200000,end_us=now,samples=rows)},now
 s.collect_process_windows=collect
 out=r.bounded_sample(lambda:sample(s,b,service,u),pause=0)
 assert len(calls)==3 and out['foreign_sm_aggregate_upper_bounds']=={'uuid0':6.}
 assert out['known_own_attach_retry_history'][0]['raw_failed_measurement']['eligibility']['zero_own_targets']==['uuid0']
