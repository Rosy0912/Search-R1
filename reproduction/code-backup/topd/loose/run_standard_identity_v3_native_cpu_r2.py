"""Single audited CPU-only deployment/submission. UNKNOWN never resubmits."""
import pathlib,hashlib,json,tarfile,io,subprocess,sys,shlex,fcntl
ROOT='/dataset/shaoshitong/search-r1-rl/topd'
NAME='standard-search-r1-teacher-identity-v3-native-r2'
ARCHIVES={'standard-search-r1-teacher-identity-v3.tar.gz':'88dee56abf81571934a7cb4141528643f0fb468129f04b12ad46e9a1f3bc2b45','standard-identity-v3-native-cpu-r2.tar.gz':'b20b784fd9e29eb738487281d826406a1c61f911cea3be87a3ecdabfadbe2d3a'}
REMOTE=r'''
import os,time,socket
r=pathlib.Path(ROOT);assert socket.gethostname().split('.')[0] in ('h10011','h100-11')
logs=r/'logs';tools=r/'tools';record=logs/(NAME+'-native-cpu-submission.json')
def durable(path,data):
 temp=path.with_name(path.name+'.tmp');assert not path.is_symlink() and not temp.is_symlink()
 with temp.open('w') as f:json.dump(data,f,indent=2);f.flush();os.fsync(f.fileno())
 os.replace(temp,path);fd=os.open(str(path.parent),os.O_RDONLY);os.fsync(fd);os.close(fd)
def verify_payload():
 data={}
 for name,h in ARCHIVES.items():
  blob=(tools/name).read_bytes();assert hashlib.sha256(blob).hexdigest()==h
  with tarfile.open(fileobj=io.BytesIO(blob),mode='r:gz') as tf:
   for m in tf.getmembers():
    p=pathlib.PurePosixPath(m.name);assert m.isfile() and not p.is_absolute() and '..' not in p.parts and m.name not in data
    data[m.name]=tf.extractfile(m).read()
 for name,h in [('standard-search-r1-teacher-identity-v3-files.json','19e8d6b26d41480c7e53b8365cf41175d2b05d56e04bd2fb9e7098466d91ef2e'),('native-files.json','bcf05b68228ae7144fe2e4689e050b9d557a94d7bcf9fec90d74f7bd3808cc67')]:
  assert hashlib.sha256(data[name]).hexdigest()==h
  for n,v in json.loads(data[name])['files'].items():assert len(data[n])==v['size'] and hashlib.sha256(data[n]).hexdigest()==v['sha256']
 assert set(data)==set(json.loads(data['standard-search-r1-teacher-identity-v3-files.json'])['files'])|set(json.loads(data['native-files.json'])['files'])|{'standard-search-r1-teacher-identity-v3-files.json','native-files.json'}
 return data
with (logs/(NAME+'-native-cpu.lock')).open('a') as lock:
 fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
 data=verify_payload();target=tools/NAME;assert not target.is_symlink()
 if target.exists():
  actual={str(p.relative_to(target)):p for p in target.rglob('*') if p.is_file()};assert set(actual)==set(data)
  for n,b in data.items():assert not actual[n].is_symlink() and actual[n].read_bytes()==b
 else:
  staging=tools/(NAME+'.staging');assert not staging.exists();staging.mkdir(mode=0o700)
  for n,b in data.items():
   p=staging/n;p.parent.mkdir(parents=True,exist_ok=True)
   with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
  for directory in sorted([p for p in staging.rglob('*') if p.is_dir()]+[staging],key=lambda p:len(p.parts),reverse=True):
   fd=os.open(str(directory),os.O_RDONLY);os.fsync(fd);os.close(fd)
  os.rename(staging,target);fd=os.open(str(tools),os.O_RDONLY);os.fsync(fd);os.close(fd)
 repo=r/'repo-shared32b';assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()=='d23c56c78fdb09f40b7859e02b14eeddbe83a863'
 assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip()
 subprocess.run(['bash','-n',str(target/'native-cpu.sbatch')],check=True)
 if record.exists():
  d=json.loads(record.read_text());assert d['archives']==ARCHIVES and d['status']=='SUBMITTED','UNKNOWN or mismatched record: inspect actual Slurm; never repeat'
 else:
  q=subprocess.check_output(['squeue','-h','-o','%i|%j'],text=True);assert not any(x.split('|')[-1]=='standard-r1-teacher-identity-v3-r2-cpu' for x in q.splitlines())
  d={'status':'UNKNOWN_BEFORE_SUBMISSION','archives':ARCHIVES,'epoch':time.time(),'scope':'NATIVE_CPU189_ONLY_NO_GPU_NO_MODEL'};durable(record,d)
  p=subprocess.run(['sbatch','--parsable','--chdir='+str(target),str(target/'native-cpu.sbatch')],capture_output=True,text=True)
  d.update(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr);durable(record,d);assert p.returncode==0
  jid=p.stdout.strip().split(';')[0];assert jid.isdigit();d.update(status='SUBMITTED',job_id=int(jid));durable(record,d)
 jid=str(d['job_id']);watch=logs/(NAME+'-native-cpu-'+jid+'-control.jsonl');end=time.monotonic()+660
 while time.monotonic()<end:
  try:
   ctl=subprocess.run(['scontrol','show','job','-dd',jid],capture_output=True,text=True,timeout=min(5,max(end-time.monotonic(),.1)))
  except (subprocess.TimeoutExpired,OSError) as exc:
   snap={'epoch':time.time(),'job_id':int(jid),'observer_error':repr(exc),'scope':'OBSERVER_ERROR_NOT_TERMINAL_NOT_RESUBMISSION'}
   with watch.open('a') as f:f.write(json.dumps(snap)+'\n');f.flush();os.fsync(f.fileno())
   raise
  snap={'epoch':time.time(),'job_id':int(jid),'returncode':ctl.returncode,'stdout':ctl.stdout,'stderr':ctl.stderr}
  with watch.open('a') as f:f.write(json.dumps(snap)+'\n');f.flush();os.fsync(f.fileno())
  if ctl.returncode!=0:raise RuntimeError('Actual terminal unavailable; retained source/control, no resubmission')
  if any('JobState='+x in ctl.stdout for x in ('COMPLETED','FAILED','CANCELLED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED')):
   final=logs/(NAME+'-native-cpu-'+jid+'-actual-final.json');assert not final.exists();durable(final,snap);d.update(actual_final_path=str(final),actual_final_sha256=hashlib.sha256(final.read_bytes()).hexdigest());durable(record,d);print(json.dumps(d),flush=True);break
  time.sleep(2)
 else:raise RuntimeError('Bounded observer timeout; no resubmit, use recorded job')
 run=logs/('standard-r1-teacher-identity-v3-r2-cpu-'+jid)
 for p in [run/'result.json']+sorted(run.glob('*.log')):
  print('RAW',str(p),hashlib.sha256(p.read_bytes()).hexdigest(),flush=True)
 print((run/'result.json').read_text(),flush=True)
'''
def main():
 cwd=pathlib.Path(__file__).resolve().parent.parent
 if '--execute' not in sys.argv:
  for n,h in ARCHIVES.items():assert hashlib.sha256((cwd/'work'/n).read_bytes()).hexdigest()==h
  compile(REMOTE,'remote','exec');print('LOCAL_STATIC_VALIDATED_NOT_EXECUTED');return
 with (cwd/'outputs/standard-identity-v3-native-cpu-r2-local.lock').open('a') as lock:
  fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
  for n in ARCHIVES:subprocess.run([sys.executable,str(cwd/'work/transfer_file.py'),str(cwd/'work'/n),ROOT+'/tools/'+n],check=True,cwd=cwd)
  prefix=pathlib.Path(__file__).read_text().split('REMOTE=',1)[0]
  script="python3 - <<'IDENTITY_NATIVE_CPU_ONCE'\n"+prefix+REMOTE+'\nIDENTITY_NATIVE_CPU_ONCE\n'
  out=cwd/'outputs/standard-identity-v3-native-cpu-r2-invocation.log'
  with out.open('a') as f:
   p=subprocess.run([sys.executable,str(cwd/'work/remote_topd.py')],input=script,text=True,stdout=f,stderr=subprocess.STDOUT,cwd=cwd);f.flush();__import__('os').fsync(f.fileno())
  print(out.read_text());assert p.returncode==0,'No blind retry: inspect record/job before recovery'
if __name__=='__main__':main()
