"""No process/GPU/system signalling. Exercise frozen code with actual certificate fixture."""
import hashlib,importlib.util,json,os,signal
from pathlib import Path
from unittest.mock import patch
base=Path(__file__).resolve().parent
watch=base/'tools/resource_watch.py'
spec=importlib.util.spec_from_file_location('bound_watch',watch); w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
snapshot=base.parent/'evidence/recovery3231-formal3246-fault-snapshot.json'
d=json.loads(snapshot.read_text());stops=[e for e in d['events'] if e['event']=='owned_freeze_signal']
class Clock:
 def __init__(self):self.t=0.
 def __call__(self):return self.t
 def sleep(self,x):self.t+=x

def case(scan_cost):
 clock=Clock();state={e['pid']:dict(e['certificate']) for e in stops};logs=[];counts={'views':0,'signals':0}
 def inventory(*args):
  counts['views']+=1;clock.t+=scan_cost
  return {p:dict(r) for p,r in state.items()}
 def send(fd,sig,*args):
  assert sig==signal.SIGSTOP;counts['signals']+=1;state[fd]['state']='T'
 with patch.dict(os.environ,SLURM_JOB_ID='3246',SEARCH_R1_GPU_LEASES=json.dumps([{'uuid':u} for u in stops[0]['lease_uuids']])):
  f=w.OwnedFreezer(stops[0]['run'],188479,logs.append,clock,clock.sleep,inventory,lambda pid,flags:pid,send,lambda fd:None)
  error=None
  try:f.freeze()
  except ValueError as e:error=str(e)
 return dict(scan_cost_seconds=scan_cost,elapsed_mock_seconds=clock.t,**counts,error=error,all_301_mock_epochs_stopped=all(r['state']=='T' for r in state.values()),stable_events=sum(e['event']=='owned_freeze_stable' for e in logs))
slow=case(.15);fast=case(.001)
assert slow['error']=='Own process freeze failed to reach stable stopped snapshot'
assert slow['all_301_mock_epochs_stopped'] and slow['signals']==301 and slow['views']==303 and slow['stable_events']==0
assert fast['error'] is None and fast['stable_events']==1
print(json.dumps(dict(scope='CPU deterministic fake clocks, fake inventory and fake pidfd send; zero real signals/GPU/process manipulation',watcher_sha256=hashlib.sha256(watch.read_bytes()).hexdigest(),actual_snapshot_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),slow=slow,fast=fast),indent=2))
