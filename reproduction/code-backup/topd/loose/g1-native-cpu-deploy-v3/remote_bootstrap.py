"""Injected source archives only; ROOT-only entry submits one noGRES CPU job."""
import base64,fcntl,hashlib,io,json,os,pathlib,socket,subprocess,tarfile,time
ROOT=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd')
PAYLOADS=[] # local approved deploy tool replaces this literal, never remote argv
ENTRY_MANIFEST_SHA256=None

def sha(path):return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
def atomic(path,data):
 p=pathlib.Path(path);tmp=p.with_name(p.name+'.private')
 with tmp.open('x') as f:json.dump(data,f,indent=2);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p);fd=os.open(p.parent,os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)
def install(item):
 name=item['name'];raw=base64.b64decode(item['base64'],validate=True)
 if hashlib.sha256(raw).hexdigest()!=item['archive_sha256']:raise ValueError('Source archive drift')
 if (ROOT/'tools').is_symlink():raise ValueError('Own tools alias')
 prefix=ROOT/'tools'/name
 archive=ROOT/'tools'/(name+'-tools.tar.gz')
 if archive.is_symlink():raise ValueError('Source archive alias')
 if archive.exists():
  if sha(archive)!=item['archive_sha256']:raise ValueError('Existing source archive differs')
 else:
  with archive.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 if prefix.is_symlink():raise ValueError('Source root alias')
 members={}
 with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as t:
  for member in t.getmembers():
   parts=pathlib.PurePosixPath(member.name).parts
   if not member.isfile() or len(parts)<2 or parts[0]!=name or any(p in ('.','..') for p in parts) or member.name.startswith('/'):raise ValueError('Regular own source members only')
   relative='/'.join(parts[1:])
   if relative in members:raise ValueError('Duplicate source member')
   members[relative]=t.extractfile(member).read()
 if hashlib.sha256(members['files.json']).hexdigest()!=item['manifest_sha256']:raise ValueError('Manifest drift')
 manifest=json.loads(members['files.json']);records=manifest['files']
 if set(members)!=set(records)|{'files.json'}:raise ValueError('Archive exact manifest set differs')
 for name,rec in records.items():
  if len(members[name])!=rec['size'] or hashlib.sha256(members[name]).hexdigest()!=rec['sha256']:raise ValueError('Archive payload bytes differ')
 prefix.mkdir(exist_ok=True)
 for relative,b in members.items():
  p=prefix/relative
  if not p.parent.resolve().is_relative_to(prefix.resolve()):raise ValueError('Source destination escaped')
  for parent in p.parents:
   if parent==prefix.parent:break
   if parent.is_symlink():raise ValueError('Source ancestor alias')
  p.parent.mkdir(parents=True,exist_ok=True)
  if p.is_symlink():raise ValueError('Source file alias')
  if p.exists():
   if p.read_bytes()!=b:raise ValueError('Existing frozen source differs, never overwrite')
  else:
   with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
 actual=set()
 for p in prefix.rglob('*'):
  if p.is_symlink():raise ValueError('Existing source alias')
  if p.is_file():actual.add(str(p.relative_to(prefix)))
 if actual!=set(members):raise ValueError('Existing source extra files')
 fd=os.open(prefix,os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)
 return {'directory':str(prefix),'manifest_sha256':item['manifest_sha256'],'archive_sha256':item['archive_sha256'],'files':len(members)}

def main():
 if socket.gethostname()!='h100-11':raise ValueError('Own h11 only')
 allowed={'g1-staged-calibration-producer-v5','g1-production-pipeline-v11','g1-known-run-tools-v2','g1-native-cpu-deploy-v3'}
 if {p['name'] for p in PAYLOADS}!=allowed or len(PAYLOADS)!=4:raise ValueError('Only pinned source tool packages, no model transport')
 record=ROOT/'logs/g1-production-v11-known-tools-native-cpu-submission.json'
 with record.with_suffix('.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);installed=[install(p) for p in PAYLOADS]
  binding={'source':installed,'entry_manifest_sha256':ENTRY_MANIFEST_SHA256}
  if record.exists():
   result=json.loads(record.read_text())
   if result.get('status')!='SUBMITTED' or result.get('binding')!=binding:raise ValueError('UNKNOWN/mismatched known record, reconcile Slurm; no retry')
  else:
   queue=subprocess.check_output(['squeue','-h','-o','%i|%j'],text=True)
   if any(line.split('|',1)[-1]=='g1-v11-known-cpu' for line in queue.splitlines()):raise ValueError('Actual same CPU job exists without record')
   entry=ROOT/'tools/g1-native-cpu-deploy-v3/native-combined.sbatch'
   if '--gres' in entry.read_text() or '--gpus' in entry.read_text():raise ValueError('CPU entry must not request GPU')
   subprocess.run(['bash','-n',str(entry)],check=True)
   result={'status':'UNKNOWN_BEFORE_SUBMISSION','scope':'CPU_SOURCE_ONLY_NO_GPU_CALIBRATION_OR_TRAINING','binding':binding,'epoch':time.time()};atomic(record,result)
   env=dict(os.environ,CUDA_VISIBLE_DEVICES='',DEPLOY_ENTRY_MANIFEST_SHA256=ENTRY_MANIFEST_SHA256)
   r=subprocess.run(['sbatch','--parsable','--chdir='+str(ROOT),str(entry)],env=env,text=True,capture_output=True)
   result.update(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr);atomic(record,result)
   number=r.stdout.strip().split(';')[0]
   if r.returncode or not number.isdigit():raise RuntimeError('CPU submission UNKNOWN, raw saved; no automatic new job')
   result.update(status='SUBMITTED',job_id=int(number));atomic(record,result)
 print(json.dumps(result))
if __name__=='__main__':main()
