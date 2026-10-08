import subprocess,pathlib
remote=r'''python3 - <<'INNER'
import hashlib,json,pathlib,subprocess,tarfile,time,fcntl,os
r=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd');archive=r/'tools/qwen25-32b-window-dual-burst-v3.tar.gz'
window_archive=r/'tools/qwen25-32b-window-safety-v3.tar.gz'
assert hashlib.sha256(window_archive.read_bytes()).hexdigest()=='9f56393a567689a2619124ed30c9f5a3fd9c910a0ee5c0e9b51f75dfe25992b0'
window_target=r/'tools/qwen25-32b-window-safety-candidate-v3';window_target.mkdir(exist_ok=True)
with tarfile.open(window_archive) as t:
 for m in t.getmembers():
  parts=pathlib.PurePosixPath(m.name).parts
  assert m.isfile() and len(parts)==2 and parts[0]==window_target.name and parts[1] not in ('.','..')
  p=window_target/parts[1];b=t.extractfile(m).read();assert not p.is_symlink()
  if p.exists():assert p.read_bytes()==b
  else:p.write_bytes(b)
assert hashlib.sha256((window_target/'files.json').read_bytes()).hexdigest()=='349ff8477c2e7fa6417f17eb783dd8a8b4c06c487c1eeedd858cbed6f5ea4e8e'
for n,h in json.loads((window_target/'files.json').read_text()).items():assert hashlib.sha256((window_target/n).read_bytes()).hexdigest()==h
assert hashlib.sha256(archive.read_bytes()).hexdigest()=='be3001b78b4431268a140c167d9836612c6aca41a45c6c2e9f202f46efcb1baa'
target=r/'tools/qwen25-32b-window-dual-burst-probe-v3';target.mkdir(exist_ok=True)
with tarfile.open(archive) as t:
 for m in t.getmembers():
  parts=pathlib.PurePosixPath(m.name).parts
  assert m.isfile() and len(parts)==2 and parts[0]==target.name and parts[1] not in ('.','..')
  p=target/parts[1];b=t.extractfile(m).read();assert not p.is_symlink()
  if p.exists():assert p.read_bytes()==b
  else:p.write_bytes(b)
for n,h in json.loads((target/'files.json').read_text()).items():assert hashlib.sha256((target/n).read_bytes()).hexdigest()==h
entry=target/'native-cpu.sbatch';assert hashlib.sha256(entry.read_bytes()).hexdigest()=='f11fe24cb82b4841db6cf3cf53798e3a27b902950f816e50b18d9c852a913e1c'
record=r/'logs/topd-window-dual-v3-burst-cpu-submission.json'
def persist(d):
 tmp=record.with_suffix('.tmp')
 with tmp.open('w') as f:json.dump(d,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,record)
 fd=os.open(str(record.parent),os.O_RDONLY);os.fsync(fd);os.close(fd)
with record.with_suffix('.lock').open('a') as lock:
 fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
 if record.exists():
  d=json.loads(record.read_text());assert d['status']=='SUBMITTED',d
 else:
  q=subprocess.check_output(['squeue','-h','-o','%i|%j'],text=True);assert not any(x.split('|',1)[-1]=='topd-window-dual-burst-cpu' for x in q.splitlines())
  d=dict(status='UNKNOWN_BEFORE_SUBMISSION',epoch=time.time(),entry=str(entry),partition_override='gpu',scope='CPU_CONTRACT_NO_GPU_NO_QUALITY');persist(d)
  res=subprocess.run(['sbatch','--partition=gpu','--parsable',str(entry)],text=True,capture_output=True);d.update(returncode=res.returncode,stdout=res.stdout,stderr=res.stderr);persist(d);assert res.returncode==0,d
  jid=res.stdout.strip().split(';')[0];assert jid.isdigit();d.update(status='SUBMITTED',job_id=int(jid));persist(d)
print(json.dumps(d))
INNER'''
p=subprocess.run(['python3','work/remote_topd.py'],input=remote,text=True,capture_output=True);print(p.stdout,end='');print(p.stderr,end='')
if not p.returncode:pathlib.Path('outputs/topd-window-dual-v3-burst-cpu-submission.json').write_text(p.stdout)
raise SystemExit(p.returncode)
