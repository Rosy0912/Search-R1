"""User-authorized stop of exact original B subtree, retaining complete3 and partial4."""
import hashlib,json,os,pathlib,signal,time,ctypes,platform
assert platform.machine()=='x86_64'
libc=ctypes.CDLL(None,use_errno=True)
def pidfd_open(pid):
 fd=libc.syscall(434,ctypes.c_int(pid),ctypes.c_uint(0))
 if fd<0:raise OSError(ctypes.get_errno(),'pidfd_open')
 return fd
def pidfd_send(fd,sig):
 result=libc.syscall(424,ctypes.c_int(fd),ctypes.c_int(sig),ctypes.c_void_p(),ctypes.c_uint(0))
 if result<0:raise OSError(ctypes.get_errno(),'pidfd_send_signal')
ROOT=pathlib.Path('/weka_3/shaoshitong/recursive-bran-b-4gpu-hkust5-20261011')
EXPECTED={401223:651299641,401228:651299742,411651:651318130,412273:651321123,412683:651321851,412684:651321852,412685:651321852}
def read(pid):
 p=pathlib.Path('/proc/%d'%pid);s=(p/'stat').read_text().rsplit(')',1)[1].split();uid=int(next(l for l in (p/'status').read_text().splitlines() if l.startswith('Uid:')).split()[1])
 return dict(pid=pid,uid=uid,tick=int(s[19]),state=s[0],ppid=int(s[1]))
def exact(pid,tick):
 x=read(pid);assert x['uid']==1001 and x['tick']==tick;return x
assert os.getuid()==1001
for pid,tick in EXPECTED.items():exact(pid,tick)
manifest=ROOT/'checkpoints/training_state/global_step_3/manifest.json';raw=manifest.read_bytes();m=json.loads(raw)
assert m['status']=='COMPLETE_SAME_WORLD_STATE_SAVED' and m['identity']['completed_step']==3
assert m['identity']['code_commit']=='3da904591125f117b1eae1ba09546ebe793cb2d4'
assert all(w['actual_total_optimizer_calls']==60 and w['observed_adam_steps']==[60] and w['scheduler_last_epoch']==3 and w['full_worker_state_finite'] for w in m['worker_states'])
assert not (ROOT/'checkpoints/training_state/global_step_4/manifest.json').exists()
assert not any((ROOT/'output/rank-audit/policy-safety').glob('step-000004-*'))
# Freeze the actual driver before verifying that no fourth update was issued.
fd=pidfd_open(411651);pidfd_send(fd,signal.SIGSTOP);os.close(fd)
time.sleep(.3)
try:
 assert exact(411651,651318130)['state'] in ('T','t')
 assert not (ROOT/'checkpoints/training_state/global_step_4/manifest.json').exists()
 assert not any((ROOT/'output/rank-audit/policy-safety').glob('step-000004-*'))
 allproc={}
 for p in pathlib.Path('/proc').iterdir():
  if p.name.isdigit():
   try:allproc[int(p.name)]=read(int(p.name))
   except (FileNotFoundError,ProcessLookupError,PermissionError):pass
 owned={401223}
 changed=True
 while changed:
  new={pid for pid,x in allproc.items() if x['ppid'] in owned};changed=not new.issubset(owned);owned|=new
 assert set(EXPECTED).issubset(owned) and all(allproc[p]['uid']==1001 for p in owned)
 records=[allproc[p] for p in sorted(owned)]
 receipt=dict(protocol='recursive-b-speed-parent-terminal-v1',time=time.time(),status='STOP_REQUEST_ISSUED',
   original_controller=dict(pid=401228,uid=1001,tick=651299742),owned_processes=records,
   parent_completed_step=3,parent_manifest_sha256=hashlib.sha256(raw).hexdigest(),
   stop_scope='Latest human: stop this B, adapt acceleration-handoff, repair and resume same complete state',
   current_partial_outer=4,partial_has_no_committed_update=True,sampling_rng_parent_not_recorded=True,
   controller_rc_known=False,no_training_rc0_inferred=True)
 target=ROOT/'receipts/speed-stop-from3-request.json'
 with target.open('x') as f:json.dump(receipt,f,sort_keys=True);f.flush();os.fsync(f.fileno())
 for pid in [401228]+[p for p in sorted(owned,reverse=True) if p not in (401223,401228)]+[401223]:
  try:
   exact(pid,allproc[pid]['tick']);fd=pidfd_open(pid);pidfd_send(fd,signal.SIGTERM)
   if pid==411651:pidfd_send(fd,signal.SIGCONT)
   os.close(fd)
  except (FileNotFoundError,ProcessLookupError):pass
except BaseException:
 try:
  exact(411651,651318130);fd=pidfd_open(411651);pidfd_send(fd,signal.SIGCONT);os.close(fd)
 except (FileNotFoundError,ProcessLookupError):pass
 raise
print(json.dumps(dict(status='EXACT_OWNED_SUBTREE_SIGTERM_SENT',owned_count=len(records),checkpoint=3,request_path=str(target))))
