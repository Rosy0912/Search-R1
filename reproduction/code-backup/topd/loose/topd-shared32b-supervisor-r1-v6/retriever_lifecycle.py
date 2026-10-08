"""Exact reviewed partial-startup resource policy; actual docs HTML readiness."""
import hashlib,json,os,threading,time
from pathlib import Path
from contracts import require,sha

# Complete independently reviewed/native3116 source, no G1 learning callback.
STARTUP_SHA='1f401e123b97e240f1040beb9ee02fbcb525d86fa90b9494951f231f9003ecf4'

class RetrieverLifecycle:
 def __init__(self,safety,guard,baseline,uuids,cgroup,run):
  source=Path(__file__).with_name('retrieval_startup.py')
  require(sha(source)==STARTUP_SHA,'Reviewed partial startup source drift')
  from dependencies import load
  self.partial=load('topd_partial_retrieval_startup',dict(path=str(source),sha256=STARTUP_SHA))
  self.safety,self.guard,self.baseline,self.uuids=safety,guard,baseline,uuids
  self.runtime=dict(baseline=baseline,retriever_identity=None,cgroup=cgroup)
  self.run=Path(run);self.lock=threading.RLock();self.starting=True
 def raw(self,record):
  with (self.run/'retriever-startup-raw.jsonl').open('a') as f:f.write(json.dumps(record,sort_keys=True,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
 def attach(self,pid):
  self.runtime['retriever_identity']=self.safety.pid_identity(pid)
 def sample(self):
  with self.lock:
   if not self.starting:return self.guard.active_gate(self.runtime,pre_model=False)
   try:
    record=self.partial.bounded_sample(lambda:self.partial.sample(self.safety,self.baseline,self.uuids,self.runtime['retriever_identity'],self.runtime['cgroup'],lease_reader=self.guard.lease_reader,raw_writer=self.raw),progress_service=self.runtime['retriever_identity'],progress_uuids=self.guard.lease_reader())
   except BaseException as exc:
    self.raw(dict(stage='FAILED_STARTUP_NOT_ACCEPTED',epoch=time.time(),error=repr(exc),raw_failed_measurement=getattr(exc,'measurement_evidence',None),raw_retry_history=getattr(exc,'retry_history',[])));raise
   self.raw(dict(stage='PARTIAL_FULL_MEASUREMENT_ACCEPTED',epoch=time.time(),actual=record,source_sha256=STARTUP_SHA));return record
 def full(self,record):
  with self.lock:
   require(self.starting and {u for u,p in record['own_pid_uuids']}==set(self.uuids),'Actual all8 retriever residency required')
   require(self.safety.pid_identity(self.runtime['retriever_identity']['pid'])==self.runtime['retriever_identity'],'Actual retriever changed at residency transition')
   # Must pass complete original WindowV5 before closing partial mode.
   result=self.guard.active_gate(self.runtime,pre_model=False)
   self.starting=False
   self.raw(dict(stage='ALL8_RESIDENT_COMPLETE_WINDOW_V5_ACCEPTED',epoch=time.time(),actual=result));return result

def docs_health(session):
 response=session.get('http://127.0.0.1:18085/docs',timeout=5)
 if response.status_code!=200:return None
 require('text/html' in response.headers.get('content-type','').lower(),'Retriever docs readiness is actual HTML')
 return dict(url='http://127.0.0.1:18085/docs',status=200,body_sha256=hashlib.sha256(response.content).hexdigest(),content_type=response.headers.get('content-type'))

def wait_ready(stage,service,launcher,errors,timeout=600):
 import requests
 deadline=time.monotonic()+timeout;session=requests.Session();session.trust_env=False
 try:
  while True:
   require(not errors,'Owned startup guardian failed '+repr(errors));require(service.poll() is None,'Owned retriever exited before readiness')
   require(time.monotonic()<deadline,'Actual retriever readiness/idle timeout')
   try:health=docs_health(session)
   except requests.exceptions.RequestException:health=None
   if health is not None:
    record=stage.sample()
    if {u for u,p in record['own_pid_uuids']}==set(stage.uuids):
     try:actual_fds=launcher.inherited_lease_fds()
     except RuntimeError as exc:
      if 'exceeds the 15% idle thresholds' not in str(exc):raise
      stage.raw(dict(stage='INITIAL_PUBLIC_IDLE_NOT_YET_MET',epoch=time.time(),error=str(exc)))
     else:
      require(not errors,'Owned startup guardian failed before full residency')
      full=stage.full(record);stage.raw(dict(stage='ACTUAL_DOCS_HTML_ALL8_RESIDENT_IDLE_READY',epoch=time.time(),health=health,lease_fds=list(actual_fds)))
      return full
   time.sleep(1)
 finally:session.close()
