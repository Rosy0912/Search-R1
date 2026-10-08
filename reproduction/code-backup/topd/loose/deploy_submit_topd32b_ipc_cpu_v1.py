"""One authorized CPU-only IPC contract deployment/submission, unknown-safe."""
import subprocess
from pathlib import Path
remote=r'''python3 - <<'INNER'
import fcntl,hashlib,json,os,pathlib,subprocess,tarfile,time
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd')
archive=root/'tools/topd-32b-pipeline-ipc-native-cpu-v1.tar.gz'
assert hashlib.sha256(archive.read_bytes()).hexdigest()=='9d80010e1abf15ca00d2216665b1e2d185580bf96f36de4c66594a53be481a52'
target=root/'tools/topd-32b-pipeline-ipc-native-cpu-v1'
assert not target.is_symlink();target.mkdir(exist_ok=True)
with tarfile.open(archive) as tar:
 members=tar.getmembers();assert len(members)==7 and all(m.isfile() for m in members)
 manifest=json.loads(tar.extractfile('files.json').read())
 expected=set(manifest['files'])|{'files.json'}
 assert {m.name for m in members}==expected
 for member in members:
  relative=pathlib.PurePosixPath(member.name);assert not relative.is_absolute() and '..' not in relative.parts
  p=target/member.name;p.parent.mkdir(exist_ok=True)
  assert not p.is_symlink() and all(not ancestor.is_symlink() for ancestor in p.parents if ancestor!=root)
  data=tar.extractfile(member).read()
  if p.exists():assert p.read_bytes()==data,'Existing deployed bytes differ: '+str(p)
  else:
   with p.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 for name,digest in manifest['files'].items():
  actual=hashlib.sha256((target/name).read_bytes()).hexdigest();assert actual==digest
  print('ACTUAL_DEPLOY_SHA256',actual,str(target/name),flush=True)
assert manifest['expected_native_tests']==18
entry=target/'native-cpu.sbatch';assert hashlib.sha256(entry.read_bytes()).hexdigest()=='5ef7de83d65d89c8ee85f1cb5beaf5ea1d250bd96f17d90cc1dddbfd31786f42'
record=root/'logs/topd32b-ipc-native-cpu-v1-submission.json'
with (root/'logs/topd32b-ipc-native-cpu-v1-submission.lock').open('a') as lock:
 fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
 if record.exists():
  state=json.loads(record.read_text());assert state['status']=='SUBMITTED','UNKNOWN requires read-only recovery; never resubmit'
 else:
  queue=subprocess.check_output(['squeue','-h','-o','%i|%j'],text=True,timeout=20)
  assert not any(x.split('|',1)[-1]=='topd32b-ipc-cpu-v1' for x in queue.splitlines())
  state=dict(status='UNKNOWN_BEFORE_SUBMISSION',epoch=time.time(),entry=str(entry),entry_sha256=hashlib.sha256(entry.read_bytes()).hexdigest(),archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),scope=manifest['scope'])
  with record.open('x') as f:json.dump(state,f);f.write('\n');f.flush();os.fsync(f.fileno())
  directory=os.open(str(record.parent),os.O_DIRECTORY);os.fsync(directory);os.close(directory)
  result=subprocess.run(['sbatch','--parsable',str(entry)],capture_output=True,text=True,timeout=30)
  assert result.returncode==0,(result.stdout,result.stderr)
  job=result.stdout.strip().split(';')[0];assert job.isdigit(),result.stdout
  state.update(status='SUBMITTED',job_id=int(job),sbatch_stdout=result.stdout,sbatch_stderr=result.stderr)
  pending=record.with_suffix('.submitted.tmp')
  with pending.open('x') as f:json.dump(state,f);f.write('\n');f.flush();os.fsync(f.fileno())
  os.replace(pending,record);directory=os.open(str(record.parent),os.O_DIRECTORY);os.fsync(directory);os.close(directory)
 print('SUBMISSION_RECORD',json.dumps(state),flush=True)
INNER'''
result=subprocess.run(['python3','work/remote_topd.py'],input=remote,text=True,capture_output=True,timeout=180)
Path('outputs/topd32b-ipc-native-cpu-v1-deploy-submit.log').write_text(result.stdout+result.stderr)
print(result.stdout,end='');print(result.stderr,end='')
raise SystemExit(result.returncode)
