"""Read-only lock proof: metadata, CPU parser and own temporary FLOCK fixtures."""
import ast,fcntl,hashlib,importlib.util,json,os,subprocess,sys,tempfile,time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path.cwd();HERE=ROOT/'work/tree-probability-probe-owned/v2/lock-child';SOURCE=ROOT/'work/tree-probability-probe-owned/resource_adapter_v2.py';EXPECTED='29822f98f8749cdbede4f62c91a0745d241a71c4bd16ce7e52e2f58126c3e273'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();assert sha(SOURCE)==EXPECTED
spec=importlib.util.spec_from_file_location('lock_child_adapter',SOURCE);A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)
checks=[]
def ck(n,b,d=None):
 assert b,(n,d);checks.append(dict(name=n,passed=True,detail=d))
def refuse(n,call,phrase):
 try:call()
 except ValueError as e:ck(n,phrase in str(e),str(e))
 else:raise AssertionError(n+' unexpectedly accepted')
envp=ROOT/'outputs/tree-instruct-300/PPO_PROBE_ADAPTER_V2_FDINFO_CPU_ENV_RAW.json';env=json.loads(envp.read_bytes())
ck('actual_ENV_capture_pins_current_adapter',env['adapter_sha256']==EXPECTED and env['status']=='ACTUAL_H1009_CPU_FDINFO_OFD_OWNER_DISTINGUISHED')
ck('actual_ENV_capture_own_toy_scope',env['actual_adapter_owner_passed'] and env['actual_adapter_foreign_holder_refused'] and env['somebody_holds_lock_after'] and env['parent_same_inode_fd_has_no_flock'] and env['model_forwards']==env['optimizer_writes']==env['gpu_calls']==0 and env['cuda_imported'] is False and 'synthetic temporary file' in env['scope'])
st=SimpleNamespace(st_dev=os.makedev(env['device_major'],env['device_minor']),st_ino=env['inode'])
with patch.object(A.os,'major',return_value=env['device_major']),patch.object(A.os,'minor',return_value=env['device_minor']):
 ck('actual_before_fdinfo_owner_parse_passes',A.fdinfo_exclusive_owner(env['before'],st))
 refuse('actual_after_no_FLOCK_refuses_foreign_global_holder',lambda:A.fdinfo_exclusive_owner(env['after'],st),'no unique kernel FLOCK')
failedp=ROOT/'outputs/tree-instruct-300/PPO_PROBE_ADAPTER_V2_FDINFO_CPU_RAW.json';failederr=ROOT/'outputs/tree-instruct-300/PPO_PROBE_ADAPTER_V2_FDINFO_CPU.err'
ck('non_ENV_failure_remains_failure_not_success',failedp.read_bytes()==b'' and 'ModuleNotFoundError: No module named' in failederr.read_text() and 'omegaconf' in failederr.read_text())
text='lock:\t1: FLOCK ADVISORY WRITE 77 fd:01:123 0 EOF\n';stat=SimpleNamespace(st_dev=os.makedev(253,1),st_ino=123)
with patch.object(A.os,'major',return_value=253),patch.object(A.os,'minor',return_value=1):
 ck('printed_PID_is_not_required_to_equal_supervisor',A.fdinfo_exclusive_owner(text,stat))
 for n,t in [('blocked_waiter','lock: 1: -> FLOCK ADVISORY WRITE 77 fd:01:123 0 EOF\n'),('partial_lock',text.replace('0 EOF','1 EOF')),('shared_lock',text.replace('WRITE','READ')),('wrong_inode',text.replace(':123',':124')),('OFDLCK_not_FLOCK',text.replace('FLOCK','OFDLCK')),('duplicate_records',text+text)]:refuse('kernel_parser_refuses_'+n,lambda v=t:A.fdinfo_exclusive_owner(v,stat),'')

# Real local FLOCK tests prove same-OFD inheritance and foreign-holder state;
# local host has no Linux /proc fdinfo. Only own tiny files/processes are used.
with tempfile.TemporaryDirectory(prefix='tree-lock-child-') as tmp:
 lock=Path(tmp)/'own.lock';lock.touch()
 with lock.open('r+') as parent:
  fcntl.flock(parent.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
  code="import fcntl,json,os,sys; fd=int(sys.argv[1]); fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB); print(json.dumps({'inherited_same_OFD_idempotent':True,'pid':os.getpid()}))"
  cp=subprocess.run([sys.executable,'-c',code,str(parent.fileno())],pass_fds=(parent.fileno(),),capture_output=True,text=True,timeout=10)
  v=json.loads(cp.stdout);ck('real_exec_inherited_same_OFD_lock_idempotent',cp.returncode==0 and v['inherited_same_OFD_idempotent'] and v['pid']!=os.getpid())
  with lock.open('r+') as own_other:
   try:fcntl.flock(own_other.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:ck('different_OFD_same_inode_blocked_while_parent_owns',True)
   else:raise AssertionError('different FD bypassed own lock')
  fcntl.flock(parent.fileno(),fcntl.LOCK_UN)
  code="import fcntl,json,sys; f=open(sys.argv[1],'r+'); fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB); print(json.dumps({'foreign_owns':True}),flush=True); sys.stdin.readline(); fcntl.flock(f.fileno(),fcntl.LOCK_UN)"
  child=subprocess.Popen([sys.executable,'-c',code,str(lock)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
  try:
   observed=json.loads(child.stdout.readline());ck('real_foreign_new_OFD_owns_after_parent_unlock',observed['foreign_owns'])
   with lock.open('r+') as own_other:
    try:fcntl.flock(own_other.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:ck('someone_held_global_probe_still_blocks_after_parent_unlock',True)
    else:raise AssertionError('foreign holder not blocking')
  finally:
   child.communicate('release\n',timeout=10)
   assert child.returncode==0

# Exercise original check with fabricated /proc/stat/fdinfo and a REAL local lock.
# The single lease fixture narrows this to ownership; full8 release is root scope.
with tempfile.TemporaryDirectory(prefix='tree-lock-adapter-check-') as tmp:
 lock=Path(tmp)/'own.lock';lock.touch();guard=Path(tmp)/'guard.json';uuids=[str(x) for x in range(8)]
 guard.write_text(json.dumps(dict(job_id='99999',epoch=100.,gpus=[dict(uuid=x,foreign_memory_fraction=.01,foreign_sm_upper=0.) for x in uuids]))+'\n')
 with lock.open('r+') as parent:
  fcntl.flock(parent.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);actual=lock.stat();other=Path(tmp)/'other';other.touch()
  kernel=f'lock: 1: FLOCK ADVISORY WRITE 77 {os.major(actual.st_dev):x}:{os.minor(actual.st_dev):x}:{actual.st_ino} 0 EOF\n'
  mode={'info':kernel,'changed_fd':False,'wrong_expected_inode':False,'start':1234,'zombie':False,'fd_reads':0,'fd_stats':0}
  class FD:
   def stat(self):
    mode['fd_stats']+=1
    return other.stat() if mode['changed_fd'] and mode['fd_stats']>1 else actual
  class FakeLeaseDir:
   def __truediv__(self,name):
    assert name=='0.lock';return other if mode['wrong_expected_inode'] else lock
  class FakeTmp:
   def __truediv__(self,name):
    assert name=='search-r1-gpu-leases-'+str(os.getuid());return FakeLeaseDir()
  def fake_path(*parts):
   if parts==('/tmp',):return FakeTmp()
   value='/'.join(str(x).strip('/') for x in parts)
   if value in ['proc/101/cgroup','proc/102/cgroup']:return SimpleNamespace(read_text=lambda:'0::/fixture/job_99999/own\n')
   if value==f'proc/101/fd/{parent.fileno()}':return FD()
   if value==f'proc/101/fdinfo/{parent.fileno()}':
    def read():mode['fd_reads']+=1;return mode['info']
    return SimpleNamespace(read_text=read)
   if value.endswith('search-r1-gpu-leases-'+str(os.getuid())+'/0.lock'):return other if mode['wrong_expected_inode'] else lock
   return Path(*parts)
  def fake_process(pid):return ('Z' if mode['zombie'] else 'S',mode['start']) if pid==101 else ('S',5678)
  state=dict(job_id='99999',supervisor=dict(pid=101,start_ticks=1234),guard=dict(pid=102,start_ticks=5678),guard_path=str(guard),gpu_uuids=uuids,leases=[dict(uuid='0',fd=parent.fileno())])
  with patch.object(A,'Path',fake_path),patch.object(A,'local',return_value=guard),patch.object(A,'process',side_effect=fake_process),patch.object(A,'ACTIVE',state),patch.object(A,'PROTECTED',{}),patch.object(A.socket,'gethostname',return_value='h100-9'),patch.object(A.time,'time',return_value=105.),patch.dict(os.environ,{'SLURM_JOB_ID':'99999'}):
   result=A.check('tiny-owner-check');ck('actual_check_control_accepts_owned_FD_different_printed_PID',result['job_id']=='99999' and mode['fd_reads']==1)
   mode.update(start=9999,fd_reads=0,fd_stats=0);refuse('check_PID_start_change_refused_before_fdinfo',lambda:A.check('tiny'),'supervisor exited or changed identity');ck('start_refusal_no_fdinfo_consumption',mode['fd_reads']==0)
   mode.update(start=1234,zombie=True);refuse('check_zombie_supervisor_refused',lambda:A.check('tiny'),'supervisor exited or changed identity')
   mode.update(zombie=False,wrong_expected_inode=True,fd_reads=0,fd_stats=0);refuse('check_FD_expected_inode_mismatch_refused',lambda:A.check('tiny'),'Supervisor lease FD changed');ck('inode_refusal_before_fdinfo',mode['fd_reads']==0)
   mode.update(wrong_expected_inode=False,changed_fd=True,fd_reads=0,fd_stats=0);refuse('check_FD_replacement_during_ownership_read_refused',lambda:A.check('tiny'),'Parent FD changed during kernel ownership check')
   mode.update(changed_fd=False,info='',fd_reads=0,fd_stats=0)
   # Parent unlock + separate own OFD now holds this same inode. The v1 global
   # lock probe would pass; v2 must refuse its parent-FD ownership first.
   fcntl.flock(parent.fileno(),fcntl.LOCK_UN)
   with lock.open('r+') as foreign:
    fcntl.flock(foreign.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    refuse('actual_check_foreign_same_inode_not_parent_OFD_refused',lambda:A.check('tiny'),'no unique kernel FLOCK')
    fcntl.flock(foreign.fileno(),fcntl.LOCK_UN)

# Original process parsing uses stat's trailing comm delimiter and field22 start.
fields=['S']+['0']*18+['123456']+['0']*3
with patch.object(A,'Path',return_value=SimpleNamespace(read_text=lambda:'101 (toy name ) with parens) '+' '.join(fields))):
 ck('process_stat_PID_start_field22_and_comm_parentheses',A.process(101)==('S',123456))
 for v in [True,0,-1,'101']:refuse('process_refuses_invalid_PID_'+repr(v),lambda n=v:A.process(n),'Invalid process identity')
result=dict(status='ADAPTER_V2_KERNEL_FLOCK_OFD_SOURCE_CPU_REVIEW_PASSED',adapter_sha256=EXPECTED,checks=checks,check_count=len(checks),own_real_local_flock_exec_toy=True,actual_h1009_saved_toy_metadata_verified=True,non_ENV_failure_not_promoted=True,fdinfo_proc_control_fixture_only=True,printed_PID_need_not_equal_supervisor=True,actual_GPU_leases_checked=False,actual_background_PIDs_checked=False,actual_PT_assets_loaded=False,actual_GPU_or_remote_calls=0,actual_model_forwards=0,actual_optimizer_writes=0,scope='Source lock subreview, saved Linux own-file toy metadata, parsing/control fixtures, real local own temporary FLOCK/exec tests only. No real lease/PID/proc state or actual runtime release accepted.')
(HERE/'LOCK_CHILD_CPU_PROOF.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n');print(json.dumps(dict(status=result['status'],check_count=len(checks))))
