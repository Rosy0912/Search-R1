"""Human-authorized exact PR100 replacement; pidfd-bound stop, then narrow outputs removal."""
import json,os,signal,time,shutil
from pathlib import Path
B=Path('/data/shaoshitong/search-e1-opsd-pr100-8gpu-20261010')
N=Path('/data/shaoshitong/search-e1-opsd-j-minus-p-bf16-20261010')
def scan():
 out={}
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   raw=(p/'stat').read_text();s=raw[raw.rfind(')')+2:].split();uid=int((p/'status').read_text().split('Uid:')[1].split()[0]);cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
   out[int(p.name)]={'pid':int(p.name),'ppid':int(s[1]),'tick':int(s[19]),'state':s[0],'uid':uid,'cmd':cmd}
  except (FileNotFoundError,ProcessLookupError,PermissionError):pass
 return out
allp=scan();roots={pid for pid,v in allp.items() if v['uid']==1005 and str(B)+'/' in v['cmd'] and any(n in v['cmd'] for n in ('/supervisor.py','/controller.py','/worker.py','/trainer.py','/memory_observer.py'))}
assert any('/supervisor.py' in allp[p]['cmd'] for p in roots),'No exact current supervisor; inspect rather than broad kill'
own=set(roots)
while True:
 more={p for p,v in allp.items() if v['ppid'] in own and v['uid']==1005}
 if more<=own:break
 own|=more
bound={p:allp[p] for p in own};fds={}
for pid,v in bound.items():
 try:
  fd=os.pidfd_open(pid);now=scan().get(pid)
  if now and (now['uid'],now['tick'])==(v['uid'],v['tick']):fds[pid]=fd
  else:os.close(fd)
 except ProcessLookupError:pass
for pid,fd in fds.items():
 if '/supervisor.py' in bound[pid]['cmd']:signal.pidfd_send_signal(fd,signal.SIGSTOP)
for pid,fd in fds.items():
 try:signal.pidfd_send_signal(fd,signal.SIGTERM)
 except ProcessLookupError:pass
for pid,fd in fds.items():
 if '/supervisor.py' in bound[pid]['cmd']:
  try:signal.pidfd_send_signal(fd,signal.SIGCONT)
  except ProcessLookupError:pass
for _ in range(20):
 current=scan();live=[p for p,v in bound.items() if p in current and current[p]['tick']==v['tick'] and current[p]['state']!='Z']
 if not live:break
 time.sleep(.5)
for pid in live:
 try:signal.pidfd_send_signal(fds[pid],signal.SIGKILL)
 except ProcessLookupError:pass
time.sleep(1);current=scan();survivors=[p for p,v in bound.items() if p in current and current[p]['tick']==v['tick'] and current[p]['state']!='Z'];assert not survivors,survivors
assert not [p for p,v in current.items() if v['uid']==1005 and str(B)+'/' in v['cmd'] and any(n in v['cmd'] for n in ('/supervisor.py','/controller.py','/worker.py','/trainer.py','/memory_observer.py'))],'New old-run processes; deletion blocked'
targets=[B/'runs',B/'logs',B/'periodic-recovery-e72eb3c'/'pipeline',*sorted((B/'periodic-recovery-e72eb3c').glob('eval-*'))]
deleted=[]
for p in targets:
 assert p.parent.resolve().is_relative_to(B.resolve()) and not p.is_symlink(),str(p)
 if p.exists():shutil.rmtree(p);deleted.append(str(p))
N.joinpath('receipts').mkdir(parents=True,exist_ok=True)
proof={'human_scope':'Stop exact old PR100 including its active evaluation; delete outputs/CKPT/LOG, preserve sources/source100/other groups/oracle','time':time.time(),'bound_processes':list(bound.values()),'survivors':survivors,'deleted':deleted,'all_deleted_absent':all(not Path(p).exists() for p in deleted),'new_training_started':False}
p=N/'receipts'/'old-pr-stop-delete.json';p.write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
