python3 - <<'PY'
import pathlib,hashlib,json,tarfile,fcntl,subprocess,os,time
r=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd');a=r/'tools/g1-stage-v7-native-stress-v1.tar.gz';target=r/'tools/g1-stage-v7-native-stress-v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(a)=='7cdb4751e51b34f714e5a87ef149d8b40c6cc8a7b310438f5855baf79c0029df'
with tarfile.open(a) as t:
 members=t.getmembers();assert len(members)==5
 for m in members:
  parts=pathlib.PurePosixPath(m.name).parts;assert m.isfile() and len(parts)==2 and parts[0]==target.name and parts[1] not in ('.','..')
 if not target.exists():t.extractall(r/'tools')
assert not target.is_symlink() and sha(target/'files.json')=='07843f4adee2a00b21048a00680cf4b42f748deb940050564c72bd3ce80eaf2b'
f=json.loads((target/'files.json').read_text())['files'];assert {p.name for p in target.iterdir()}==set(f)|{'files.json'}
for n,v in f.items():assert not (target/n).is_symlink() and (target/n).stat().st_size==v['size'] and sha(target/n)==v['sha256']
producer=r/'tools/g1-staged-calibration-producer-v7'
assert sha(producer/'stage_cleanup.py')=='cc55f32fbb7eb65f6d4f13ed5b9a58d54bbb73ee25ae4cd6918a0d68dc9b3a80'
assert sha(producer/'owner_support.py')=='d4de70880869abbec64f81ccd86f6f8dd8755b01d9db2cb93e2ec112d3031a09'
record=r/'logs/g1-stage-v7-native-stress-v1-submission.json'
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
  assert not any(line.split('|',1)[-1]=='g1-stage-v7-proc-stress' for line in queue.splitlines()),queue
  v={'status':'UNKNOWN_BEFORE_SUBMISSION','epoch':time.time(),'archive_sha256':sha(a),'manifest_sha256':sha(target/'files.json'),'entry_sha256':sha(target/'native.sbatch'),'scope':'ONE_CPU_NO_GRES_NO_MODEL_NO_CALIBRATION'};persist(v)
  q=subprocess.run(['sbatch','--parsable','--output',str(r/'logs/g1-stage-v7-native-stress-%j.log'),str(target/'native.sbatch')],capture_output=True,text=True);v.update(returncode=q.returncode,stdout=q.stdout,stderr=q.stderr);persist(v);assert q.returncode==0,v
  jid=q.stdout.strip().split(';')[0];assert jid.isdigit(),v;v.update(status='SUBMITTED',job_id=int(jid));persist(v)
print(json.dumps(v));print(subprocess.check_output(['scontrol','show','job',str(v['job_id']),'-o'],text=True))
PY
