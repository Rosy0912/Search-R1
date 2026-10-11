"""Stop only the original recursive-A owner tree after its pinned fourth state."""
import hashlib,json,os,pathlib,signal,time
B=pathlib.Path('/data/shaoshitong/recursive-bran-a-8gpu-20261011')
ROOT,TICK,UID=3360882,953222439,1005
M=B/'checkpoints/training_state/global_step_4/manifest.json'
SHA='4d781f6920ce8ea8d56bff7c488ea960aa5b60e7b99a7635d19961bbfa06816f'
def read(pid):
 p=pathlib.Path('/proc',str(pid));s=(p/'stat').read_text().split(') ',1)[1].split()
 return dict(pid=pid,uid=p.stat().st_uid,ppid=int(s[1]),tick=int(s[19]),state=s[0],argv=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace'))
assert os.getuid()==UID
raw=M.read_bytes();assert hashlib.sha256(raw).hexdigest()==SHA
m=json.loads(raw);assert m['status']=='COMPLETE_SAME_WORLD_STATE_SAVED'
assert m['identity']['guard_continue_accounting']['total_optimizer_calls_per_rank']=={str(r):80 for r in range(8)}
owner=read(ROOT);assert owner['uid']==UID and owner['tick']==TICK
rows={}
for p in pathlib.Path('/proc').iterdir():
 if p.name.isdigit():
  try: rows[int(p.name)]=read(int(p.name))
  except (FileNotFoundError,ProcessLookupError,PermissionError):pass
scope={ROOT}
while True:
 new={p for p,r in rows.items() if r['ppid'] in scope}
 if new<=scope:break
 scope|=new
assert 2239529 not in scope and all(rows[p]['uid']==UID for p in scope)
handles={}
for p in scope:
 if rows[p]['state']=='Z':continue
 try:
  fd=os.pidfd_open(p);r=read(p)
  assert r['uid']==UID and r['tick']==rows[p]['tick'];handles[p]=fd
 except (FileNotFoundError,ProcessLookupError):pass
receipt=B/'receipts/sparse-schedule/cache-resume4-owned-stop.json'
assert not receipt.exists()
proof=dict(time=time.time(),scope=[rows[p] for p in sorted(scope)],checkpoint=str(M),sha256=SHA,reason='human explicit stop/resume for same-policy vLLM weight reuse',signals=[])
receipt.write_text(json.dumps(proof,indent=2))
# Freeze the authenticated owner first so no new child/update starts during teardown.
signal.pidfd_send_signal(handles[ROOT],signal.SIGSTOP)
for p,fd in handles.items():
 try:signal.pidfd_send_signal(fd,signal.SIGTERM);proof['signals'].append([p,'TERM'])
 except ProcessLookupError:pass
try:signal.pidfd_send_signal(handles[ROOT],signal.SIGCONT)
except ProcessLookupError:pass
time.sleep(8)
for p,fd in handles.items():
 try:
  r=read(p)
  if r['state']!='Z':signal.pidfd_send_signal(fd,signal.SIGKILL);proof['signals'].append([p,'KILL'])
 except (FileNotFoundError,ProcessLookupError):pass
 finally:os.close(fd)
proof['finished_at']=time.time();proof['remaining']=[]
for p in scope:
 try:
  r=read(p)
  if r['tick']==rows[p]['tick'] and r['state']!='Z':proof['remaining'].append(r)
 except (FileNotFoundError,ProcessLookupError):pass
assert hashlib.sha256(M.read_bytes()).hexdigest()==SHA
receipt.write_text(json.dumps(proof,indent=2));assert not proof['remaining']
print(json.dumps(dict(stopped=True,owned_processes=len(scope),checkpoint_sha256=SHA)))
