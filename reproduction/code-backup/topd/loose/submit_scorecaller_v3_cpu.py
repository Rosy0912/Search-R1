import subprocess,pathlib
remote=r'''python3 - <<'INNER'
import hashlib,json,pathlib,subprocess,tarfile,time,fcntl,os
r=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd');archive=r/'tools/topd-32b-native-score-caller-v3.tar.gz'
assert hashlib.sha256(archive.read_bytes()).hexdigest()=='0ca9c90475f55c3f19a61315a1c2c02056d95ae0ce88191bcc425cdfcb9aeb56'
target=r/'tools/topd-32b-native-score-gate-v3';target.mkdir(exist_ok=True)
with tarfile.open(archive) as t:
 for m in t.getmembers():
  parts=pathlib.PurePosixPath(m.name).parts
  assert m.isfile() and len(parts)==2 and parts[0]==target.name and parts[1] not in ('.','..')
  p=target/parts[1];b=t.extractfile(m).read();assert not p.is_symlink()
  if p.exists():assert p.read_bytes()==b
  else:p.write_bytes(b)
for n,h in json.loads((target/'tool-files.json').read_text()).items():assert hashlib.sha256((target/n).read_bytes()).hexdigest()==h
entry=target/'native-cpu.sbatch';assert hashlib.sha256(entry.read_bytes()).hexdigest()=='ac960446242eeb03d592b57ce2062f85945e13ce3cce30df96e5d436a4b99c1e'
record=r/'logs/topd32b-score-caller-v3-cpu-submission.json'
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
  q=subprocess.check_output(['squeue','-h','-o','%i|%j'],text=True);assert not any(x.split('|',1)[-1]=='topd32b-score-caller-v3-cpu' for x in q.splitlines())
  d=dict(status='UNKNOWN_BEFORE_SUBMISSION',epoch=time.time(),entry=str(entry),partition_override='gpu',scope='CPU_CONTRACT_NO_GPU_NO_QUALITY');persist(d)
  res=subprocess.run(['sbatch','--partition=gpu','--job-name=topd32b-score-caller-v3-cpu','--output='+str(r/'logs/topd32b-score-caller-v3-cpu-%j.log'),'--parsable',str(entry)],text=True,capture_output=True);d.update(returncode=res.returncode,stdout=res.stdout,stderr=res.stderr);persist(d);assert res.returncode==0,d
  jid=res.stdout.strip().split(';')[0];assert jid.isdigit();d.update(status='SUBMITTED',job_id=int(jid));persist(d)
print(json.dumps(d))
INNER'''
p=subprocess.run(['python3','work/remote_topd.py'],input=remote,text=True,capture_output=True);print(p.stdout,end='');print(p.stderr,end='')
if not p.returncode:pathlib.Path('outputs/topd32b-score-caller-v3-cpu-submission.json').write_text(p.stdout)
raise SystemExit(p.returncode)
