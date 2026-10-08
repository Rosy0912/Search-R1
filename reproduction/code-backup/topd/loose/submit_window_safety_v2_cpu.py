import pathlib, subprocess

remote = r'''python3 - <<'INNER'
import json,pathlib,hashlib,tarfile,os,fcntl,subprocess,time
r=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd')
archive=r/'tools/qwen25-32b-window-safety-v2.tar.gz'
assert hashlib.sha256(archive.read_bytes()).hexdigest()=='71910615b1c2518f2482a91047872f174bd307c139faf9bcfa85db6b86f72439'
target=r/'tools/qwen25-32b-window-safety-candidate-v2';target.mkdir(exist_ok=True)
with tarfile.open(archive) as t:
 for m in t.getmembers():
  parts=pathlib.PurePosixPath(m.name).parts
  assert m.isfile() and len(parts)==2 and parts[0]==target.name and parts[1] not in ('.','..')
  p=target/parts[1];b=t.extractfile(m).read();assert not p.is_symlink()
  if p.exists():assert p.read_bytes()==b
  else:p.write_bytes(b)
manifest=target/'files.json'
assert hashlib.sha256(manifest.read_bytes()).hexdigest()=='5c921921dafbb95144243f8a90310d17ea719d3557daba0d16abe83e45e6d589'
files=json.loads(manifest.read_text())
for n,h in files.items():
 p=target/n;assert target in p.resolve().parents and not p.is_symlink() and hashlib.sha256(p.read_bytes()).hexdigest()==h
assert {p.name for p in target.iterdir()}==set(files)|{'files.json'}
entry=target/'native-cpu.sbatch'
assert hashlib.sha256(entry.read_bytes()).hexdigest()=='8bd42449afe1ab76f77629918e1ea3ab321802774aa948941b624412f110a784'
record=r/'logs/topd-window-safety-v2-cpu-submission.json'
def persist(d):
 tmp=record.with_suffix('.tmp')
 with tmp.open('w') as f:json.dump(d,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,record);fd=os.open(str(record.parent),os.O_RDONLY);os.fsync(fd);os.close(fd)
with record.with_suffix('.lock').open('a') as lock:
 fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
 if record.exists():
  d=json.loads(record.read_text());assert d['status']=='SUBMITTED',d
 else:
  q=subprocess.check_output(['squeue','-h','-o','%i|%j'],text=True);assert not any(x.split('|',1)[-1]=='topd-window-safety-cpu' for x in q.splitlines())
  d=dict(status='UNKNOWN_BEFORE_SUBMISSION',scope='CPU_ONLY_NOT_GPU_OR_TRAINING',epoch=time.time(),entry=str(entry));persist(d)
  res=subprocess.run(['sbatch','--parsable',str(entry)],text=True,capture_output=True);d.update(returncode=res.returncode,stdout=res.stdout,stderr=res.stderr);persist(d);assert res.returncode==0,d
  jid=res.stdout.strip().split(';')[0];assert jid.isdigit();d.update(status='SUBMITTED',job_id=int(jid));persist(d)
print(json.dumps(d))
INNER'''
p=subprocess.run(['python3','work/remote_topd.py'],input=remote,text=True,capture_output=True)
print(p.stdout,end='');print(p.stderr,end='')
if not p.returncode:pathlib.Path('outputs/topd-window-safety-v2-native-cpu-submission.json').write_text(p.stdout)
raise SystemExit(p.returncode)
