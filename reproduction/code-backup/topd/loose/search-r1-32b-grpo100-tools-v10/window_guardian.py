"""Actual WindowV5 policy; owns no GPU and preserves every raw measurement."""
import hashlib,importlib.util,json,os,pathlib,sys,threading,time
RUNTIME='9e41464d49054a1765fed29f5839f9fd8c568f0040aedbf6cf36b3c096adb3c6'
GUARD='65165ac7c8f5c9042ba254fd36d26e222096e6764fa83c167061e25530d5ad39'
MANIFEST='c8506109114ac45d2f18ac7f4eaf7c1f4d411ce8c0fabfabb0e78d82f689dac0'
OLD_GUARD='d340c77f54c8c744816da416c91016e59bd5a21bd42262b2ae672a3432f6588b'
EVAL='24d144fdac16b0983a11e3bd99d111f44d0bc312dc98b74847f3d16524076bee'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def load(name,path,expected):
 p=pathlib.Path(path).resolve()
 if sha(p)!=expected:raise ValueError('Guardian source drift '+str(p))
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def validate_policy_review(root,policy):
 root=pathlib.Path(root).resolve()
 def read(ref):
  p=pathlib.Path(ref['path']).resolve()
  if root not in p.parents or sha(p)!=ref['sha256']:raise ValueError('Actual telemetry evidence drift')
  return json.loads(p.read_text())
 review=read(policy['actual_independent_native_review'])
 if review.get('schema')!='WINDOW_V5_CPU_REPLAY_POLICY_REVIEW_V1' or review.get('status')!='CPU_REPLAY_POLICY_PASS_NOT_GPU_TELEMETRY_APPROVAL' or review.get('approval_scope')!='SOURCE_AND_NATIVE_CPU_CONTRACTS_ONLY' or review.get('actual_gpu_telemetry_approval') is not False or review.get('runtime_source_sha256')!=RUNTIME or review.get('adapter_source_sha256')!=GUARD:raise ValueError('WindowV5 CPU review identities/scope differ')
 if review.get('policy')!={'ordinary_attempt_cap':6,'explicit_low_high_attempt_cap':9,'call_seconds':30,'startup_seconds':120}:raise ValueError('WindowV5 exact measurement budgets differ')
 for name in ('source_manifest','native_cpu_result','native_cpu_log','independent_source_review','inherited_runtime_observation_ref'):
  ref=review[name];p=pathlib.Path(ref['path']).resolve()
  if root not in p.parents or sha(p)!=ref['sha256']:raise ValueError('Actual telemetry raw drift '+name)
 if review['source_manifest']['sha256']!=MANIFEST:raise ValueError('WindowV5 reviewed manifest differs')
 old=read(review['inherited_runtime_observation_ref'])
 if old.get('status')!='OBSERVED_PATHS_PASS_NOT_FOUND_BRANCH_UNOBSERVED' or old.get('source_sha256')!=RUNTIME or old.get('adapter_sha256')!=OLD_GUARD or old.get('monitor_policy_source_initial_allocation_eligible') is not True or old.get('approval_scope')!='MONITOR_POLICY_SOURCE_AND_OBSERVED_PATHS_ONLY_INITIAL_ALLOCATION_NOT_FORMAL100_NOT_TEACHER_NOT_TOPD':raise ValueError('Inherited unchanged-runtime observation differs')
 for ref in old['raw_refs']:
  p=pathlib.Path(ref['path']).resolve()
  if root/'logs' not in p.parents or sha(p)!=ref['sha256']:raise ValueError('Inherited actual telemetry raw drift')
 return review
class Guardian:
 def __init__(self,root,policy,evidence,scope,owned):
  self.root,self.policy,self.evidence,self.scope,self.owned=pathlib.Path(root),policy,pathlib.Path(evidence),scope,owned
  if policy['runtime_sha256']!=RUNTIME or policy['guard_sha256']!=GUARD:raise ValueError('Only fixed WindowV5 policy')
  validate_policy_review(self.root,policy)
  evalpath=self.root/'tools/qwen25-32b-terminal-fix-refreeze/eval64.py'
  if 'eval64' in sys.modules and pathlib.Path(sys.modules['eval64'].__file__).resolve()!=evalpath:raise ValueError('eval64 shadow')
  load('eval64',evalpath,EVAL)
  # Policy imports original same-node lease_ownership by verified repo413 path;
  # child algorithm PYTHONPATH is separately scoped to unchanged G1 candidate.
  sys.path.insert(0,str(self.root/'repo-terminal-fixed413'))
  directory=pathlib.Path(policy['directory']).resolve()
  if directory!=self.root/'tools/qwen25-32b-window-safety-candidate-v5' or sha(directory/'files.json')!=policy['manifest_sha256'] or policy['manifest_sha256']!=MANIFEST:raise ValueError('Exact V5 directory/manifest required')
  self.safety=load('g1_window_runtime',directory/'runtime_safety.py',RUNTIME);self.adapter=load('g1_window_adapter',directory/'startup_guard.py',GUARD)
  self.baseline=self.safety.nvml_snapshot();self.runtime=None;self.guard=None;self.current_phase=None;self.ready=False;self.lock=threading.RLock();self.errors=[];self.stop=threading.Event();self.thread=None;self.starting_retriever=False;self.exit_ledger=None;self.controlled_exit=False
 def begin_phase(self,label):
  """Only owner invokes before controlled CUDA launch; raw prior exit must pass."""
  with self.lock:
   if getattr(self,"exit_ledger",None):self.exit_ledger.close();self.exit_ledger=None
   self.controlled_exit=False
   if self.guard:
    self.guard.close_startup();self.safety.nvml_snapshot=self.guard.original_snapshot;self.safety.collect_process_windows=self.guard.original_collect;self.safety._startup_guard_spy=False
   self.current_phase=label;self.ready=False
   self.guard=self.adapter.StartupGuard(self.safety,expected_path=pathlib.Path(self.policy['directory'])/'runtime_safety.py',trace_path=self.evidence/('window-'+label+'.jsonl'),low_high_attempts=9)
   self.guard.ready_check=lambda:self.ready
 def prepare_retriever_start(self,cgroup):
  self.runtime=dict(baseline=self.baseline,retriever_identity=None,cgroup=cgroup)
  self.begin_phase('retriever-starting');self.starting_retriever=True;self.sample()
 def attach_retriever(self,pid):
  with self.lock:self.runtime['retriever_identity']=self.safety.pid_identity(pid)
 def resident_retriever(self,pid,cgroup):
  self.runtime=dict(baseline=self.baseline,retriever_identity=self.safety.pid_identity(pid),cgroup=cgroup);self.starting_retriever=False
  self.begin_phase('retriever-resident')
  self.sample()
 def sample(self):
  with self.lock:
   if self.runtime is None:return
   if self.starting_retriever:
    from retrieval_startup import sample,bounded_sample
    def persist_raw(record):
     with open(self.evidence/'retriever-startup-raw-measurements.jsonl','a') as f:f.write(json.dumps(record,sort_keys=True,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
    try:record=bounded_sample(lambda:sample(self.safety,self.baseline,self.guard.lease_reader(),self.runtime['retriever_identity'],self.runtime['cgroup'],lease_reader=self.guard.lease_reader,raw_writer=persist_raw),progress_service=self.runtime['retriever_identity'],progress_uuids=self.guard.lease_reader())
    except BaseException as failure:
     with open(self.evidence/'retriever-startup-failures.jsonl','a') as f:f.write(json.dumps(dict(epoch=time.time(),error=str(failure),raw_retry_history=getattr(failure,'retry_history',[]),raw_failed_measurement=getattr(failure,'measurement_evidence',None)),sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
     raise
   else:record=self.guard.active_gate(self.runtime,pre_model=False)
   with open(self.evidence/'foreign-dual15-active.jsonl','a') as f:f.write(json.dumps(dict(epoch=time.time(),phase=self.current_phase,source_runtime_sha256=RUNTIME,source_guard_sha256=GUARD,actual=record),sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
   return record
 def close_ready(self,stage_pid=None):
  with self.lock:
   self.ready=True;self.guard.close_startup()
 def record_cleanup(self):
  with open(self.evidence/'owned-cleanup-history.jsonl','a') as f:
   f.write(json.dumps(dict(epoch=time.time(),scope='GUARDIAN_FIRST_FAILURE_OWNED_CLEANUP',actual=getattr(self.owned,'cleanup_record',None)),sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
 def start(self):
  def run():
   while not self.stop.is_set():
    try:
     if hasattr(self,'phase_poll'):self.phase_poll()
     self.sample()
    except BaseException as e:
     self.errors.append(repr(e));self.stop.set()
     # Fail means revoke the ongoing owned CUDA work, not wait24h for fit.
     try:self.owned.cleanup()
     except BaseException as cleanup:self.errors.append('cleanup:'+repr(cleanup))
     finally:
      try:self.record_cleanup()
      except BaseException as persist:self.errors.append('cleanup-history:'+repr(persist))
    self.stop.wait(1)
  self.thread=threading.Thread(target=run,name='r1-actual-foreign-dual15');self.thread.start()
 def check(self):
  if self.errors:raise RuntimeError('Actual foreign dual15 failed '+repr(self.errors))
 def finish(self):
  self.stop.set()
  if self.thread:self.thread.join(45)
  if self.thread and self.thread.is_alive():raise RuntimeError('Guardian did not terminate')
  self.check()
  if self.runtime:self.sample()
 def close(self):
  if getattr(self,"exit_ledger",None):self.exit_ledger.close();self.exit_ledger=None
  if self.guard:
   self.guard.close_startup();self.safety.nvml_snapshot=self.guard.original_snapshot;self.safety.collect_process_windows=self.guard.original_collect;self.safety._startup_guard_spy=False
