"""Deploy exact three frozen initial-only packages; submit unique CPU assembly."""
import base64,hashlib,json,pathlib,subprocess,sys
freeze=json.loads(pathlib.Path(sys.argv[1]).read_text())
code=r'''
import base64,fcntl,hashlib,io,json,os,pathlib,socket,subprocess,tarfile,time
r=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd');assert socket.gethostname()=='h100-11'
f=json.loads(base64.b64decode('FREEZE'))
for name,rec in f['packages'].items():
 blob=(r/'tools'/(name+'-tools.tar.gz')).read_bytes();assert hashlib.sha256(blob).hexdigest()==rec['archive_sha256']
 with tarfile.open(fileobj=io.BytesIO(blob),mode='r:gz') as t:
  members=t.getmembers();assert len(members)==rec['members'] and all(m.isfile() for m in members)
  data={}
  for m in members:
   p=pathlib.PurePosixPath(m.name);assert p.parts[0]==name and '..' not in p.parts and not p.is_absolute()
   n=str(p.relative_to(name));assert n not in data;data[n]=t.extractfile(m).read()
 assert hashlib.sha256(data['files.json']).hexdigest()==rec['manifest_sha256']
 manifest=json.loads(data['files.json']);assert set(data)==set(manifest['files'])|{'files.json'}
 for n,x in manifest['files'].items():assert len(data[n])==x['size'] and hashlib.sha256(data[n]).hexdigest()==x['sha256']
 target=r/'tools'/name
 if target.exists():
  assert not target.is_symlink() and {str(p.relative_to(target)) for p in target.rglob('*') if p.is_file()}==set(data)
  for n,b in data.items():assert not (target/n).is_symlink() and (target/n).read_bytes()==b
 else:
  staging=r/'tools'/(name+'.deploy-staging');assert not staging.exists();staging.mkdir(mode=0o700)
  for n,b in data.items():
   p=staging/n;p.parent.mkdir(parents=True,exist_ok=True)
   with p.open('xb') as out:out.write(b);out.flush();os.fsync(out.fileno())
  os.rename(staging,target)
recipe_name=next(n for n in f['packages'] if n.startswith('g1-native-recipe-'))
record=r/'logs'/(recipe_name+'-cpu-submission.json')
def persist(d):
 temp=record.with_suffix('.tmp')
 with temp.open('w') as out:json.dump(d,out,indent=2);out.flush();os.fsync(out.fileno())
 os.replace(temp,record);fd=os.open(record.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
with record.with_suffix('.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 if record.exists():
  d=json.loads(record.read_text());assert d['status']=='SUBMITTED',d
 else:
  entry=r/'tools'/recipe_name/'native-assemble.sbatch'
  subprocess.run(['bash','-n',str(entry)],check=True)
  queue=subprocess.check_output(['squeue','-h','-w','h100-11','-o','%i|%j|%T'],text=True)
  assert not any(x.split('|')[1]=='g1-initial-static-cpu' for x in queue.splitlines())
  d=dict(status='UNKNOWN_BEFORE_SUBMISSION',epoch=time.time(),scope='CPU_ONLY_NATIVE_RECIPE_ASSEMBLY_NO_GPU',packages=f['packages'],entry_sha256=hashlib.sha256(entry.read_bytes()).hexdigest(),queue_before=queue);persist(d)
  ordered=[next(n for n in f['packages'] if n.startswith(prefix)) for prefix in ('g1-staged-calibration-producer-','g1-production-pipeline-','g1-native-recipe-')]
  args=[f['packages'][n]['manifest_sha256'] for n in ordered]
  p=subprocess.run(['sbatch','--parsable',str(entry)]+args,text=True,capture_output=True)
  d.update(sbatch_returncode=p.returncode,stdout=p.stdout,stderr=p.stderr);persist(d);assert p.returncode==0,d
  jid=p.stdout.strip().split(';')[0];assert jid.isdigit(),d
  d.update(status='SUBMITTED',job_id=int(jid));persist(d)
  ctl=subprocess.run(['scontrol','show','job','-dd',jid],text=True,capture_output=True)
  saved=r/'logs'/('g1-initial-static-cpu-'+jid+'-scontrol-initial.txt')
  with saved.open('x') as out:out.write(ctl.stdout+ctl.stderr);out.flush();os.fsync(out.fileno())
  d['initial_scontrol_sha256']=hashlib.sha256(saved.read_bytes()).hexdigest();persist(d)
print(json.dumps(d))
'''.replace('FREEZE',base64.b64encode(json.dumps(freeze).encode()).decode())
p=subprocess.run([sys.executable,'work/remote_topd.py'],input="python3 - <<'G1_CPU_SUBMIT'\n"+code+'\nG1_CPU_SUBMIT\n',text=True,capture_output=True)
pathlib.Path('outputs/g1-initial-cpu-new-recipe-submission-invocation.log').write_text(p.stdout+p.stderr)
print(p.stdout,end='');print(p.stderr,end='',file=sys.stderr)
raise SystemExit(p.returncode)
