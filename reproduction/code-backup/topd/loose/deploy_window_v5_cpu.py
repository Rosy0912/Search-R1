"""WindowV5 flat manifest operation only; same once-only CPU submission observer."""
import hashlib,json,pathlib,re,sys
sys.path.insert(0,'work/g1-native-cpu-deploy-v6')
from stream_transport import send
freeze=pathlib.Path(sys.argv[1]);f=json.loads(freeze.read_text())
assert sys.argv[2]=='window-v5' and f['native_expected']==111 and set(f['packages'])=={'tools'}
assert f['packages']['tools']['archive_sha256']=='fe975a1b6598fd3d815cdc2530aec3b037939637c73a04f705f8dc44f8aff0b7'
assert f['packages']['tools']['manifest_sha256']=='c8506109114ac45d2f18ac7f4eaf7c1f4d411ce8c0fabfabb0e78d82f689dac0'
items=[]
for kind,r in f['packages'].items():
 p=pathlib.Path(r['archive']);assert hashlib.sha256(p.read_bytes()).hexdigest()==r['archive_sha256']
 items.append(dict(r,kind=kind,prefix=pathlib.Path(r['directory']).name,
   remote_archive='/dataset/shaoshitong/search-r1-rl/topd/tools/r1-standard-'+kind+'-'+r['archive_sha256'][:16]+'.tar.gz',
   manifest_name='r1-files.json' if kind=='source' else 'files.json'))
assert re.fullmatch(r'[a-z0-9][a-z0-9-]{0,63}',sys.argv[2])
payload=dict(items=items,native_expected=f['native_expected'],attempt=sys.argv[2])
source=r'''
import sys
sys.dont_write_bytecode=True
import fcntl,hashlib,json,os,pathlib,re,socket,subprocess,tarfile,time
P=PAYLOAD
ROOT=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd');assert socket.gethostname()=='h100-11'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def atomic(p,value):
 t=p.with_name(p.name+'.private');assert not t.exists()
 with t.open('x') as out:json.dump(value,out,indent=2);out.flush();os.fsync(out.fileno())
 os.replace(t,p);fd=os.open(p.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
for r in P['items']:
 archive=pathlib.Path(r['remote_archive']);assert not archive.is_symlink() and sha(archive)==r['archive_sha256']
 data={}
 with tarfile.open(archive) as tar:
  members=tar.getmembers();assert len(members)==r['archive_members']
  for m in members:
   q=pathlib.PurePosixPath(m.name);assert m.isfile() and not q.is_absolute() and q.parts[0]==r['prefix'] and '..' not in q.parts
   name=str(q.relative_to(r['prefix']));assert name not in data;data[name]=tar.extractfile(m).read()
 assert hashlib.sha256(data[r['manifest_name']]).hexdigest()==r['manifest_sha256']
 records=json.loads(data[r['manifest_name']]);assert type(records) is dict and len(records)==12 and set(data)==set(records)|{r['manifest_name']}
 for name,v in records.items():
  assert type(v) is str and re.fullmatch(r'[0-9a-f]{64}',v)
  assert hashlib.sha256(data[name]).hexdigest()==v
 target=pathlib.Path(r['deployment_directory']);assert ROOT in target.resolve().parents and not target.is_symlink()
 if not target.exists():
  temp=target.with_name(target.name+'.install-private');assert not temp.exists();temp.mkdir(mode=0o700)
  for name,b in data.items():
   q=temp/name;q.parent.mkdir(parents=True,exist_ok=True)
   with q.open('xb') as out:out.write(b);out.flush();os.fsync(out.fileno())
  os.rename(temp,target)
 assert {str(q.relative_to(target)) for q in target.rglob('*') if q.is_file()}==set(data)
 for q in target.rglob('*'):assert not q.is_symlink()
 for name,b in data.items():assert (target/name).read_bytes()==b
record=ROOT/'logs'/('r1-standard32b-sourcecpu-'+P['attempt']+'-submission.json')
with record.with_suffix('.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 if record.exists():
  d=json.loads(record.read_text());assert d['status']=='SUBMITTED' and d['binding']==P
 else:
  tools=next(pathlib.Path(r['deployment_directory']) for r in P['items'] if r['kind']=='tools');entry=tools/'native-cpu.sbatch'
  subprocess.run(['bash','-n',str(entry)],check=True)
  queue=subprocess.check_output(['squeue','-h','-w','h100-11','-o','%i|%j|%T'],text=True)
  assert not any(row.split('|')[1]=='r1-32b-source-cpu' for row in queue.splitlines()),'Inspect existing standard source CPU job before another attempt'
  d=dict(status='UNKNOWN_BEFORE_SUBMISSION',binding=P,epoch=time.time(),scope='SOURCE_CPU_ONLY_NO_GPU_MODELS_OR_TRAINING');atomic(record,d)
  env=dict(os.environ,R1_TOOLS=str(tools),PYTHONDONTWRITEBYTECODE='1')
  run=subprocess.run(['sbatch','--parsable','--chdir='+str(ROOT),str(entry)],env=env,text=True,capture_output=True)
  d.update(returncode=run.returncode,stdout=run.stdout,stderr=run.stderr);atomic(record,d);assert run.returncode==0,d
  job=run.stdout.strip().split(';')[0];assert job.isdigit();d.update(status='SUBMITTED',job_id=int(job));atomic(record,d)
 job=d['job_id'];deadline=time.monotonic()+60;terminal=None
 observations=ROOT/'logs'/('r1-standard32b-sourcecpu-'+str(job)+'-observations.jsonl')
 while time.monotonic()<deadline:
  remaining=deadline-time.monotonic()
  try:
   run=subprocess.run(['scontrol','show','job','-dd',str(job)],text=True,capture_output=True,timeout=min(5,remaining))
   raw=dict(job_id=job,epoch=time.time(),returncode=run.returncode,stdout=run.stdout,stderr=run.stderr,final_state_observed=False)
   fields=dict(re.findall(r'(\w+)=([^\s]+)',run.stdout))
   ended=run.returncode==0 and fields.get('JobId')==str(job) and fields.get('NodeList')=='h100-11' and fields.get('JobState') in ('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL')
   raw['final_state_observed']=ended
  except (subprocess.TimeoutExpired,OSError) as exc:
   raw=dict(job_id=job,epoch=time.time(),error_type=type(exc).__name__,error=str(exc),final_state_observed=False);ended=False
  with observations.open('a') as out:out.write(json.dumps(raw)+'\n');out.flush();os.fsync(out.fileno())
  if ended:
   terminal=dict(raw=raw,state=fields['JobState'],exit_code=fields.get('ExitCode'));break
  time.sleep(min(2,max(0,deadline-time.monotonic())))
 d['observation']=dict(terminal=terminal,path=str(observations),sha256=sha(observations));atomic(record,d)
 print(json.dumps(d))
'''.replace('PAYLOAD',repr(payload))
compile(source,'reviewed-standard-source-cpu-deploy','exec')
if '--render-only' in sys.argv:
 pathlib.Path('outputs/standard-r1-cpu-deploy-render.py').write_text(source);print('SOURCE_ONLY_RENDERED');sys.exit()
result=send(source,attempts=1,timeout=180)
pathlib.Path('outputs/standard-r1-cpu-deploy-'+payload['attempt']+'-invocation.log').write_bytes(result.stdout+result.stderr)
sys.stdout.buffer.write(result.stdout);sys.stderr.buffer.write(result.stderr);sys.exit(result.returncode)
