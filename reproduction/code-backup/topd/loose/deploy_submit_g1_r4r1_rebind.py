"""One native CPU source-only rebind; preserves previous actual model/data evidence."""
import base64,hashlib,json,pathlib,subprocess,sys
f=json.loads(pathlib.Path('outputs/g1-initial-v3r4r1-freeze.json').read_text());records=[]
for source in f['packages']:
 rec=dict(source);rec['members']=rec.pop('regular_members');p=pathlib.Path(rec['archive']);b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==rec['archive_sha256'];records.append(dict(name=p.name[:-13],archive_name=p.name,data=base64.b64encode(b).decode(),**rec))
root='/dataset/shaoshitong/search-r1-rl/topd';args=[records[0]['manifest_sha256'],records[1]['manifest_sha256'],root+'/logs/g1-initial-static-cpu-3102/recipe/initial-only-static-config.json',root+'/logs/g1-initial-static-cpu-3102/recipe/assembly-receipt.json',root+'/logs/g1-initial3102-r4r1-static-config.json']
payload={'packages':records,'args':args}
code=r'''import base64,fcntl,hashlib,io,json,os,pathlib,socket,subprocess,tarfile,time
r=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd');assert socket.gethostname()=='h100-11';p=json.loads(base64.b64decode('PAYLOAD'))
for rec in p['packages']:
 b=base64.b64decode(rec['data']);assert hashlib.sha256(b).hexdigest()==rec['archive_sha256'];archive=r/'tools'/rec['archive_name']
 if archive.exists():assert archive.read_bytes()==b
 else:
  temp=archive.with_suffix('.install-private');assert not temp.exists()
  with temp.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
  os.link(temp,archive);temp.unlink()
 with tarfile.open(fileobj=io.BytesIO(b),mode='r:gz') as t:
  ms=t.getmembers();assert len(ms)==rec['members'] and all(m.isfile() for m in ms);data={}
  for m in ms:
   n=pathlib.PurePosixPath(m.name);assert n.parts[0]==rec['name'] and '..' not in n.parts and not n.is_absolute();rel=str(n.relative_to(rec['name']));assert rel not in data;data[rel]=t.extractfile(m).read()
 assert hashlib.sha256(data['files.json']).hexdigest()==rec['manifest_sha256'];manifest=json.loads(data['files.json']);assert set(data)==set(manifest['files'])|{'files.json'}
 for n,v in manifest['files'].items():assert len(data[n])==v['size'] and hashlib.sha256(data[n]).hexdigest()==v['sha256']
 target=r/'tools'/rec['name']
 if target.exists():
  assert not target.is_symlink() and {str(x.relative_to(target)) for x in target.rglob('*') if x.is_file()}==set(data)
  for n,b in data.items():assert not (target/n).is_symlink() and (target/n).read_bytes()==b
 else:
  tmp=target.with_name(target.name+'.deploy-private');assert not tmp.exists();tmp.mkdir(mode=0o700)
  for n,b in data.items():
   q=tmp/n;q.parent.mkdir(parents=True,exist_ok=True)
   with q.open('xb') as out:out.write(b);out.flush();os.fsync(out.fileno())
  os.rename(tmp,target)
record=r/'logs/g1-initial-r4r1-rebind-cpu-submission.json'
def persist(d):
 temp=record.with_suffix('.tmp')
 with temp.open('w') as out:json.dump(d,out,indent=2);out.flush();os.fsync(out.fileno())
 os.replace(temp,record);fd=os.open(record.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
with record.with_suffix('.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 if record.exists():
  d=json.loads(record.read_text());assert d['status']=='SUBMITTED' and d['native_args']==p['args'],d
 else:
  entry=r/'tools/g1-native-recipe-v4-r1/native-rebind.sbatch';subprocess.run(['bash','-n',str(entry)],check=True)
  q=subprocess.check_output(['squeue','-h','-w','h100-11','-o','%i|%j'],text=True);assert not any(x.split('|')[1]=='g1-initial-r4-rebind-cpu' for x in q.splitlines())
  d=dict(status='UNKNOWN_BEFORE_SUBMISSION',scope='NATIVE_CPU_INCREMENTAL29_SOURCE_ONLY_REBIND_NO_GPU_NO_REPEAT_MODELS_DATA',epoch=time.time(),entry_sha256=hashlib.sha256(entry.read_bytes()).hexdigest(),native_args=p['args']);persist(d)
  s=subprocess.run(['sbatch','--parsable','--chdir='+str(r),str(entry)]+p['args'],capture_output=True,text=True)
  d.update(returncode=s.returncode,stdout=s.stdout,stderr=s.stderr);persist(d);assert s.returncode==0,d
  jid=s.stdout.strip().split(';')[0];assert jid.isdigit(),d;d.update(status='SUBMITTED',job_id=int(jid));persist(d)
  ctl=subprocess.run(['scontrol','show','job','-dd',jid],capture_output=True,text=True);dest=r/'logs'/('g1-initial-r4r1-rebind-cpu-'+jid+'-scontrol-initial.txt')
  with dest.open('x') as out:out.write(ctl.stdout+ctl.stderr);out.flush();os.fsync(out.fileno())
  d['initial_scontrol_sha256']=hashlib.sha256(dest.read_bytes()).hexdigest();persist(d)
print(json.dumps(d))
'''.replace('PAYLOAD',base64.b64encode(json.dumps(payload).encode()).decode())
p=subprocess.run([sys.executable,'work/remote_topd.py'],input="python3 - <<'DEPLOY_G1_R4R1'\n"+code+'\nDEPLOY_G1_R4R1\n',text=True,capture_output=True)
pathlib.Path('outputs/g1-initial-r4r1-rebind-submission-invocation.log').write_text(p.stdout+p.stderr);print(p.stdout,end='');print(p.stderr,end='',file=sys.stderr);raise SystemExit(p.returncode)
