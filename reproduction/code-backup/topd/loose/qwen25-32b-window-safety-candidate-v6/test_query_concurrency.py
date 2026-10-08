"""Read-only query scheduling contracts. No CUDA or runtime freshness approval."""
import sys,threading,time,types
from concurrent.futures import ThreadPoolExecutor
import pytest
import runtime_safety as c


def test_snapshot_both_original_queries_really_overlap(monkeypatch):
 barrier=threading.Barrier(2);calls=[]
 def query(argv,**kwargs):
  calls.append((argv,kwargs));barrier.wait(timeout=2)
  return '0, u, 20, 100, 80\n' if any('--query-gpu=' in a for a in argv) else 'u, 2, 10\n'
 monkeypatch.setattr(c.subprocess,'check_output',query);monkeypatch.setattr(c,'pid_identity',lambda pid:dict(pid=pid))
 result=c.nvml_snapshot()
 assert len(calls)==2 and all(k==dict(text=True,timeout=10) for _,k in calls)
 assert result==dict(gpus=[dict(index='0',uuid='u',used_mib=20.,total_mib=100.,sm=80.)],applications=[dict(uuid='u',pid=2,used_mib=10.)],identities={'2':dict(pid=2)})


def test_snapshot_failure_still_joins_sibling(monkeypatch):
 barrier=threading.Barrier(2);finished=threading.Event()
 def query(argv,**kwargs):
  barrier.wait(timeout=2)
  if any('--query-gpu=' in a for a in argv):raise RuntimeError('actual GPU query failed')
  time.sleep(.03);finished.set();return ''
 monkeypatch.setattr(c.subprocess,'check_output',query)
 with pytest.raises(RuntimeError,match='actual GPU query failed'):c.nvml_snapshot()
 assert finished.is_set()


def install_nvml(monkeypatch,*,fail=None):
 barrier=threading.Barrier(8);lock=threading.Lock();completed=[];state=dict(initialized=False,shutdown=False,waited=False)
 class NvError(Exception):
  def __init__(self,value):self.value=value
 def init():state['initialized']=True
 def handle(uuid):assert state['initialized'] and not state['shutdown'];return uuid
 def utilization(uuid,start):
  assert state['waited'];barrier.wait(timeout=2)
  try:
   if uuid==fail:raise RuntimeError('unexpected query failure')
   return [types.SimpleNamespace(pid=2,timeStamp=int(time.time()*1e6),smUtil=50)]
  finally:
   with lock:completed.append(uuid)
 def shutdown():
  assert len(completed)==8 and not state['shutdown'];state['shutdown']=True
 fake=types.SimpleNamespace(nvmlInit=init,nvmlShutdown=shutdown,nvmlDeviceGetHandleByUUID=handle,nvmlDeviceGetProcessUtilization=utilization,NVMLError=NvError)
 monkeypatch.setitem(sys.modules,'pynvml',fake)
 def wait(seconds):assert seconds==1.0;state['waited']=True
 monkeypatch.setattr(time,'sleep',wait)
 snapshot=dict(gpus=[dict(uuid=str(i),sm=80) for i in range(8)])
 return snapshot,{(str(i),1) for i in range(8)},state,completed


def test_all_eight_nvml_queries_overlap_join_and_keep_actual_clocks(monkeypatch):
 snapshot,foreign,state,completed=install_nvml(monkeypatch)
 windows,now=c.collect_process_windows(snapshot,foreign)
 assert state['shutdown'] and len(completed)==8 and len(windows)==8
 assert len({w['start_us'] for w in windows.values()})==1
 assert all(w['status']=='SUCCESS' and w['start_us']<=w['samples'][0]['timestamp_us']<=w['end_us']<=now and w['query_start_us']<=w['query_end_us']==w['end_us'] for w in windows.values())
 assert now<=int(time.time()*1e6)


def test_unexpected_nvml_failure_joins_every_worker_before_shutdown(monkeypatch):
 snapshot,foreign,state,completed=install_nvml(monkeypatch,fail='0')
 with pytest.raises(RuntimeError,match='unexpected query failure'):c.collect_process_windows(snapshot,foreign)
 assert state['shutdown'] and len(completed)==8


def test_non_target_no_nvml_context_or_thread(monkeypatch):
 monkeypatch.setitem(sys.modules,'pynvml',types.SimpleNamespace(nvmlInit=lambda:pytest.fail('must not initialize')))
 windows,now=c.collect_process_windows(dict(gpus=[dict(uuid='u',sm=15)]),{('u',1)})
 assert windows=={} and isinstance(now,int)


def test_actual3204_window_remains_stale_at_original_error_clock():
 import pathlib,json
 r=json.loads((pathlib.Path(__file__).parent/'actual3204-last-window.json').read_text())['original_record']
 w=r['original_windows'][0];foreign={tuple(x) for x in w['foreign']}
 snapshot=r['captures'][-1]['raw_snapshot']
 own={(x['uuid'],x['pid']) for x in snapshot['applications']}-foreign
 # Actual query clock passed, error-log clock exceeds1.5s: no earlier clock is substituted.
 c.sm_aggregate_bounds(w['snapshot'],foreign,own,w['windows'],w['query_completed_us'])
 with pytest.raises(RuntimeError,match='Stale/future process sample'):
  c.sm_aggregate_bounds(snapshot,foreign,own,w['windows'],int(r['epoch']*1e6),actual_query_completed_us=w['query_completed_us'])
