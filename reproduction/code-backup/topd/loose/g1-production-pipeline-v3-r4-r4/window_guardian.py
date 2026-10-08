"""Actual WindowV4 policy; owns no GPU and preserves every raw measurement."""
import hashlib,importlib.util,json,os,pathlib,sys,threading,time
RUNTIME='9e41464d49054a1765fed29f5839f9fd8c568f0040aedbf6cf36b3c096adb3c6'
GUARD='d340c77f54c8c744816da416c91016e59bd5a21bd42262b2ae672a3432f6588b'
EVAL='24d144fdac16b0983a11e3bd99d111f44d0bc312dc98b74847f3d16524076bee'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def load(name,path,expected):
 p=pathlib.Path(path).resolve()
 if sha(p)!=expected:raise ValueError('Guardian source drift '+str(p))
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
class Guardian:
 def __init__(self,root,policy,evidence,scope,owned):
  self.root,self.policy,self.evidence,self.scope,self.owned=pathlib.Path(root),policy,pathlib.Path(evidence),scope,owned
  if policy['runtime_sha256']!=RUNTIME or policy['guard_sha256']!=GUARD:raise ValueError('Only fixed WindowV4 policy')
  native=policy.get('actual_independent_native_review')
  if not native or sha(native['path'])!=native['sha256']:raise ValueError('WindowV4 actual native review pending')
  review=json.loads(pathlib.Path(native['path']).read_text())
  if review.get('status')!='OBSERVED_PATHS_PASS_NOT_FOUND_BRANCH_UNOBSERVED' or review.get('source_sha256')!=RUNTIME or review.get('adapter_sha256')!=GUARD or review.get('monitor_policy_source_initial_allocation_eligible') is not True or review.get('approval_scope')!='MONITOR_POLICY_SOURCE_AND_OBSERVED_PATHS_ONLY_INITIAL_ALLOCATION_NOT_FORMAL100_NOT_TEACHER_NOT_TOPD':raise ValueError('WindowV4 actual review identities differ')
  for r in review['raw_refs']:
   if not (self.root/'logs') in pathlib.Path(r['path']).resolve().parents or sha(r['path'])!=r['sha256']:raise ValueError('Actual telemetry raw drift')
  evalpath=self.root/'tools/qwen25-32b-terminal-fix-refreeze/eval64.py'
  if 'eval64' in sys.modules and pathlib.Path(sys.modules['eval64'].__file__).resolve()!=evalpath:raise ValueError('eval64 shadow')
  load('eval64',evalpath,EVAL)
  # Policy imports original same-node lease_ownership by verified repo413 path;
  # child algorithm PYTHONPATH is separately scoped to unchanged G1 candidate.
  sys.path.insert(0,str(self.root/'repo-terminal-fixed413'))
  directory=pathlib.Path(policy['directory']).resolve()
  if directory!=self.root/'tools/qwen25-32b-window-safety-candidate-v4' or sha(directory/'files.json')!=policy['manifest_sha256'] or policy['manifest_sha256']!='9835bc499bcb10da1959ae0e3bce88cf7a7966ad51ed21dc80fbe03368eaba8c':raise ValueError('Exact V4 directory/manifest required')
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
   self.guard=self.adapter.StartupGuard(self.safety,expected_path=pathlib.Path(self.policy['directory'])/'runtime_safety.py',trace_path=self.evidence/('window-'+label+'.jsonl'))
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
    try:record=bounded_sample(lambda:sample(self.safety,self.baseline,self.guard.lease_reader(),self.runtime['retriever_identity'],self.runtime['cgroup'],lease_reader=self.guard.lease_reader,raw_writer=persist_raw))
    except BaseException as failure:
     with open(self.evidence/'retriever-startup-failures.jsonl','a') as f:f.write(json.dumps(dict(epoch=time.time(),error=str(failure),raw_retry_history=getattr(failure,'retry_history',[]),raw_failed_measurement=getattr(failure,'measurement_evidence',None)),sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
     raise
   elif self.controlled_exit:
    from controlled_exit import exit_gate
    def raw(record):
     with open(self.evidence/'controlled-exit-raw.jsonl','a') as f:f.write(json.dumps(dict(epoch=time.time(),phase=self.current_phase,actual=record),sort_keys=True,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
    record=exit_gate(self.guard,self.safety,self.runtime,self.exit_ledger,raw)
   else:record=self.guard.active_gate(self.runtime,pre_model=False)
   with open(self.evidence/'foreign-dual15-active.jsonl','a') as f:f.write(json.dumps(dict(epoch=time.time(),phase=self.current_phase,source_runtime_sha256=RUNTIME,source_guard_sha256=GUARD,actual=record),sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
   return record
 def close_ready(self,stage_pid=None):
  with self.lock:
   from controlled_exit import ReadyLedger
   if self.exit_ledger:self.exit_ledger.close()
   self.exit_ledger=ReadyLedger(self.safety,self.runtime,self.guard.lease_reader(),self.safety.nvml_snapshot(),stage_pid=stage_pid)
   with open(self.evidence/'ready-owned-exit-ledger.jsonl','a') as f:f.write(json.dumps(dict(epoch=time.time(),phase=self.current_phase,actual=self.exit_ledger.evidence()),sort_keys=True,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
   self.ready=True;self.guard.close_startup()
 def begin_controlled_exit(self,label):
  with self.lock:
   if not self.ready or self.exit_ledger is None:raise RuntimeError('Explicit controlled EXIT requires actual READY ledger')
   ledger=self.exit_ledger;self.exit_ledger=None
   self.begin_phase(label)
   self.exit_ledger=ledger;ledger.active=True;self.controlled_exit=True
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
  self.thread=threading.Thread(target=run,name='g1-actual-foreign-dual15');self.thread.start()
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

def poll_phase_controls(guardian,config,root,tools):
 from pipeline import atomic,validate_peer
 control=pathlib.Path(root)/'phase-control'
 for request in sorted(control.glob('*.request.json')):
  ack=request.with_name(request.name.replace('.request.json','.ack.json'))
  if ack.exists():continue
  need=lambda x,msg: None if x else (_ for _ in ()).throw(ValueError(msg))
  need(not request.is_symlink(),'Phase request alias');r=json.loads(request.read_text())
  need(r['operation'] in ('BEGIN','READY','EXIT') and r['phase'] in ('A','B') and r['outer_step'] in (0,),'Unknown supervised stage')
  need(str(r['job'])==config['job'] and tools.process_identity(r['pid'])['starttime_ticks']==r['starttime_ticks'],'Phase requester identity drift')
  validate_peer(r['pid'],os.getuid(),os.getpid(),config['cgroup'])
  cp=pathlib.Path(r['config_path']).resolve();directory=pathlib.Path(r['directory']).resolve();need(pathlib.Path(config['evidence_root']).resolve() in cp.parents and pathlib.Path(config['evidence_root']).resolve() in directory.parents,'Phase evidence foreign')
  need(sha(cp)==r['config_sha256'],'Phase config bytes changed');c=json.loads(cp.read_text());need(c['job']==config['job'] and c['uuids']==config['uuids'] and c['outer_step']==r['outer_step'],'Phase config allocation mismatch')
  cmd=pathlib.Path('/proc/%d/cmdline'%r['pid']).read_bytes().split(b'\0');expected='owner.py' if r['operation']=='BEGIN' else 'stage_worker.py';need(str(pathlib.Path(config['producer'])/expected).encode() in cmd,'Requester not frozen producer entry')
  state=getattr(guardian,'protocol_state',{})
  key=str(r['outer_step']);current=state.get(key)
  if r['operation']=='BEGIN':
   need(current==(None if r['phase']=='A' else 'A_EXIT'),'Phase ordering mismatch')
   if r['phase']=='B':
    b=json.loads((directory/'phase-A-owner-barrier.json').read_text());need(b['returncode']==0 and not b['own_processes_remaining'] and not b['own_gpu_records_remaining'] and b['baseline_preserved'] and b['original_global_gpu_baseline_preserved'] and b['config_sha256']==r['config_sha256'],'Actual A exit absent')
   guardian.begin_phase('outer%s-%s-start'%(key,r['phase']));guardian.sample();state[key]=r['phase']+'_BEGIN'
  elif r['operation']=='READY':
   need(current==r['phase']+'_BEGIN','Unordered stage READY')
   snapshot=guardian.safety.nvml_snapshot();leased=set(config['uuids']);own=set()
   for row in snapshot['applications']:
    ident=snapshot['identities'][str(row['pid'])]
    member=any(x.split(':',2)[-1]==guardian.runtime['cgroup'] or x.split(':',2)[-1].startswith(guardian.runtime['cgroup']+'/') for x in ident['cgroup'])
    if member and row['pid']!=guardian.runtime['retriever_identity']['pid']:own.add(row['uuid'])
   need(own==leased,'Actual stage all8 model GPU coverage absent');guardian.sample();guardian.close_ready(stage_pid=r['pid']);state[key]=r['phase']+'_READY'
  else:
   need(current==r['phase']+'_READY','Unordered stage EXIT');guardian.sample();guardian.begin_controlled_exit('outer%s-%s-controlled-exit'%(key,r['phase']));state[key]=r['phase']+'_EXIT'
  guardian.protocol_state=state
  atomic(ack,dict(status='AUTHORIZED_TRANSITION',request_sha256=sha(request),runtime_sha256=RUNTIME,guard_sha256=GUARD,actual_epoch=time.time(),phase_state=state[key]))
