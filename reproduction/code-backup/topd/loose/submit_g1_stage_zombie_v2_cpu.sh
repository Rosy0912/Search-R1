python3 - <<'PY'
import pathlib,hashlib,json,tarfile,fcntl,subprocess,os,time
r=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd');a=r/'tools/g1-stage-zombie-native-stress-v2.tar.gz';target=r/'tools/g1-stage-zombie-native-stress-v2'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(a)=='b51121a12b2b5e5fef74421a978536ee3f4fae4f80453673fce228a692e7bd89'
with tarfile.open(a) as t:
 members=t.getmembers();assert len(members)==5
 for m in members:
  parts=pathlib.PurePosixPath(m.name).parts;assert m.isfile() and len(parts)==2 and parts[0]==target.name and parts[1] not in ('.','..')
 if not target.exists():t.extractall(r/'tools')
assert not target.is_symlink() and sha(target/'files.json')=='638d6143020454feb526c4c08efe1ffd250c1681028b0a1e9922f0882ee24b9c'
f=json.loads((target/'files.json').read_text())['files'];assert {p.name for p in target.iterdir()}==set(f)|{'files.json'}
for n,v in f.items():assert not (target/n).is_symlink() and (target/n).stat().st_size==v['size'] and sha(target/n)==v['sha256']
record=r/'logs/g1-stage-zombie-native-stress-v2-submission.json'
def persist(v):
 tmp=record.with_suffix('.tmp')
 with tmp.open('w') as f:json.dump(v,f,indent=2);f.flush();os.fsync(f.fileno())
 os.replace(tmp,record);fd=os.open(record.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
with record.with_suffix('.lock').open('a') as lock:
 fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
 if record.exists():
  v=json.loads(record.read_text())
  assert v['status']=='SUBMITTED','Unknown submission: refuse retry; inspect actual queue '+json.dumps(v)
 else:
  queue=subprocess.check_output(['squeue','-h','-o','%i|%j'],text=True)
  assert not any(line.split('|',1)[-1]=='g1-stage-zombie-cpu-probe' for line in queue.splitlines()),queue
  v={'status':'UNKNOWN_BEFORE_SUBMISSION','epoch':time.time(),'archive_sha256':sha(a),'manifest_sha256':sha(target/'files.json'),'entry_sha256':sha(target/'native.sbatch'),'scope':'ONE_CPU_NO_GRES_NO_MODEL_NO_CALIBRATION'};persist(v)
  q=subprocess.run(['sbatch','--parsable','--output',str(r/'logs/g1-stage-zombie-native-stress-%j.log'),str(target/'native.sbatch')],capture_output=True,text=True);v.update(returncode=q.returncode,stdout=q.stdout,stderr=q.stderr);persist(v);assert q.returncode==0,v
  jid=q.stdout.strip().split(';')[0];assert jid.isdigit(),v;v.update(status='SUBMITTED',job_id=int(jid));persist(v)
print(json.dumps(v));print(subprocess.check_output(['scontrol','show','job',str(v['job_id']),'-o'],text=True))
PY
