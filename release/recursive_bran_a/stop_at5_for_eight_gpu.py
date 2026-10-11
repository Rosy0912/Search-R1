"""One-shot authenticated boundary stop for authorized eight-shard retrieval."""
import hashlib,json,os,signal,subprocess,time
from pathlib import Path
B=Path('/data/shaoshitong/recursive-bran-a-8gpu-20261011')
ROOT,TICK,UID=3920441,953928259,1005
RUN='3bdfeb78c2a7480980f190a5c1c96bdf'
M=B/'checkpoints/training_state/global_step_5/manifest.json'
R=B/'receipts/sparse-schedule/eight-gpu-boundary5-stop.json'
def read(pid):
 p=Path('/proc',str(pid));s=(p/'stat').read_text().rsplit(') ',1)[1].split()
 return dict(pid=pid,uid=p.stat().st_uid,ppid=int(s[1]),tick=int(s[19]),state=s[0])
assert os.getuid()==UID and not R.exists()
raw=M.read_bytes();m=json.loads(raw);i=m['identity'];g=i['guard_continue_accounting']
assert m['status']=='COMPLETE_SAME_WORLD_STATE_SAVED' and i['completed_step']==5
assert i['job_id']=='hkust-direct-'+RUN and i['code_commit']=='d5542fe5fad12abd672926e67fbfd9df53bb7187'
counts=g['total_optimizer_calls_per_rank'];assert len(set(counts.values()))==1 and 80<=next(iter(counts.values()))<=100
assert len(m['worker_states'])==8 and all(w['full_worker_state_finite'] for w in m['worker_states'])
for w in m['worker_states']:
 p=Path(w['path']);assert p.is_file() and p.stat().st_size==w.get('bytes',p.stat().st_size)
owner=read(ROOT);assert owner['uid']==UID and owner['tick']==TICK
start=json.loads((B/('receipts/timed-guard-start-'+RUN+'.json')).read_text());assert start['pid']==ROOT and start['run_id']==RUN
rows={}
for p in Path('/proc').iterdir():
 if p.name.isdigit():
  try:rows[int(p.name)]=read(int(p.name))
  except (FileNotFoundError,PermissionError,ProcessLookupError):pass
scope={ROOT}
while True:
 children={pid for pid,r in rows.items() if r['ppid'] in scope}
 if children<=scope:break
 scope|=children
assert 2239529 not in scope and all(rows[x]['uid']==UID for x in scope)
apps={int(x) for x in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).split()}
assert apps<=scope|{2239529}
handles={}
for pid in scope:
 if rows[pid]['state']=='Z':continue
 try:
  fd=os.pidfd_open(pid);assert read(pid)['tick']==rows[pid]['tick'];handles[pid]=fd
 except (FileNotFoundError,ProcessLookupError):pass
proof=dict(time=time.time(),reason='human eight-GPU retrieval/search-only performance handoff after complete5; preserve later partials',
 checkpoint=str(M),checkpoint_sha256=hashlib.sha256(raw).hexdigest(),parent_identity=i,scope=[rows[x] for x in sorted(scope)],signals=[])
with R.open('x') as f:json.dump(proof,f,indent=2)
for pid,fd in handles.items():
 try:signal.pidfd_send_signal(fd,signal.SIGSTOP)
 except ProcessLookupError:pass
for pid,fd in handles.items():
 try:signal.pidfd_send_signal(fd,signal.SIGTERM);signal.pidfd_send_signal(fd,signal.SIGCONT);proof['signals'].append([pid,'TERM'])
 except ProcessLookupError:pass
time.sleep(5)
for pid,fd in handles.items():
 try:
  if read(pid)['tick']==rows[pid]['tick'] and read(pid)['state']!='Z':signal.pidfd_send_signal(fd,signal.SIGKILL);proof['signals'].append([pid,'KILL'])
 except (FileNotFoundError,ProcessLookupError):pass
 finally:os.close(fd)
for _ in range(20):
 remaining=[]
 for pid in scope:
  try:
   r=read(pid)
   if r['tick']==rows[pid]['tick'] and r['state']!='Z':remaining.append(r)
  except (FileNotFoundError,ProcessLookupError):pass
 if not remaining:break
 time.sleep(1)
assert hashlib.sha256(M.read_bytes()).hexdigest()==proof['checkpoint_sha256']
proof.update(finished_at=time.time(),remaining=remaining,owned_release=not remaining)
R.write_text(json.dumps(proof,indent=2));assert not remaining
print(json.dumps(proof))
