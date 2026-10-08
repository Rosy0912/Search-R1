"""Once own Slurm8 tiny CUDA original-math proof; never training/capacity."""
import pathlib,json,hashlib,sys,subprocess
sys.path.insert(0,'work/g1-native-cpu-deploy-v6')
from stream_transport import send
F=json.loads(pathlib.Path('outputs/entropy-device-proof-v1-r1-final-freeze.json').read_text())
assert F['archive_sha256']=='5ba8ed7e5a8e2edc2c7e4940a944850c0f095a5605810c983512dc24b8e83c41'
assert hashlib.sha256(pathlib.Path(F['archive']).read_bytes()).hexdigest()==F['archive_sha256']
F['files']=json.loads((pathlib.Path(F['directory'])/'files.json').read_text())
assert hashlib.sha256((pathlib.Path(F['directory'])/'files.json').read_bytes()).hexdigest()==F['manifest_sha256']
REMOTE=r'''
import sys
sys.dont_write_bytecode=True
import pathlib,json,hashlib,tarfile,os,fcntl,socket,subprocess,time,re
F=FREEZE
R=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd');assert socket.gethostname()=='h100-11'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def save(p,d):
 t=p.with_name(p.name+'.private')
 with t.open('w') as f:json.dump(d,f,indent=2);f.flush();os.fsync(f.fileno())
 os.replace(t,p);fd=os.open(p.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
record=R/'logs/entropy-device-proof-v1-r1-final-submission.json'
with record.with_suffix('.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 if record.exists():
  d=json.loads(record.read_text());assert d['status']=='SUBMITTED' and d['archive_sha256']==F['archive_sha256'],'UNKNOWN or different record: inspect; never resubmit'
 else:
  archive=R/'tools/entropy-device-proof-v1-r1-final.tar.gz';assert sha(archive)==F['archive_sha256']
  data={}
  with tarfile.open(archive) as t:
   for m in t.getmembers():
    q=pathlib.PurePosixPath(m.name);assert m.isfile() and not q.is_absolute() and q.parts[0]=='entropy-device-proof-v1-r1-final' and '..' not in q.parts
    name=str(q.relative_to(q.parts[0]));assert name not in data;data[name]=t.extractfile(m).read()
  assert len(data)==7 and hashlib.sha256(data['files.json']).hexdigest()==F['manifest_sha256']
  assert json.loads(data['files.json'])==F['files'] and set(data)==set(F['files'])|{'files.json'}
  for n,h in F['files'].items():assert hashlib.sha256(data[n]).hexdigest()==h
  target=R/'tools/entropy-device-proof-v1-r1-final'
  if not target.exists():
   tmp=target.with_name(target.name+'.install-private');tmp.mkdir(mode=0o700)
   for n,b in data.items():
    with (tmp/n).open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
   os.rename(tmp,target)
  assert {p.name for p in target.iterdir()}==set(data)
  for n,b in data.items():assert not (target/n).is_symlink() and (target/n).read_bytes()==b
  pins={
   'repo-terminal-fixed413/search_r1/baselines/preflight.py':'cc9445265a3835e1623c0de9d95380717f0e44d4a303d45e3fe1f559431e1605',
   'repo-terminal-fixed413/search_r1/baselines/topd_local_gpu.py':'4861cb7cd20fe9277a89dca21eff77e7e6107d91940e0f6591b6b75d3e5f6a89',
   'tools/topd-short-run.py':'d4de70880869abbec64f81ccd86f6f8dd8755b01d9db2cb93e2ec112d3031a09',
   'tools/qwen25-32b-terminal-fix-refreeze/eval64.py':'24d144fdac16b0983a11e3bd99d111f44d0bc312dc98b74847f3d16524076bee',
   'tools/qwen25-32b-window-safety-candidate-v6/runtime_safety.py':'d9c495d8f2eb5bf9b1ff5680b6a8c0c2626d1e5375e92ae4971a78a9bf3dac70',
   'tools/search-r1-32b-grpo100-tools-v14/retrieval_startup.py':'1f401e123b97e240f1040beb9ee02fbcb525d86fa90b9494951f231f9003ecf4'}
  for n,h in pins.items():assert not (R/n).is_symlink() and sha(R/n)==h,n
  # Existing official static archives are already transferred; exact immutable install.
  for name,ah,mname,mh,count in (
   ('search-r1-32b-grpo100-source-v5','4b96c2836a151682a2380853033759f654f9c90708630eb798bed3671fa35004','r1-files.json','72c5fd163b9cbe6bfb502d098b7bc14e7dfa386e5c008ffd7a202eebd14b57b4',340),
   ('entropy-source-v5-contracts','f00490f47f854f5dc008eeae2fa1dabdbff4a4a8541326510e8d4f6caddd2595','files.json','4697569d9509aaa86cad24dc5316b8c6385d0af565ef11ed81c89ea6ef6d232e',6)):
   ap=R/'tools'/(name+'.tar.gz');assert sha(ap)==ah;payload={}
   with tarfile.open(ap) as tf:
    for member in tf.getmembers():
     rel=pathlib.PurePosixPath(member.name);assert member.isfile() and not rel.is_absolute() and rel.parts[0]==name and '..' not in rel.parts
     n=str(rel.relative_to(name));assert n not in payload;payload[n]=tf.extractfile(member).read()
   assert len(payload)==count and hashlib.sha256(payload[mname]).hexdigest()==mh
   listed=json.loads(payload[mname])['files'];assert set(payload)==set(listed)|{mname}
   for n,v in listed.items():assert len(payload[n])==v['size'] and hashlib.sha256(payload[n]).hexdigest()==v['sha256']
   dest=R/'tools'/name
   if not dest.exists():
    temp=dest.with_name(dest.name+'.entropy-install-private');temp.mkdir(mode=0o700)
    for n,b in payload.items():
     p=temp/n;p.parent.mkdir(parents=True,exist_ok=True)
     with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    for p in sorted((p for p in temp.rglob('*') if p.is_dir()),reverse=True):
     fd=os.open(p,os.O_RDONLY);os.fsync(fd);os.close(fd)
    fd=os.open(temp,os.O_RDONLY);os.fsync(fd);os.close(fd);os.rename(temp,dest);fd=os.open(dest.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
   assert {str(p.relative_to(dest)) for p in dest.rglob('*') if p.is_file()}==set(payload)
   assert not any(p.is_symlink() for p in dest.rglob('*'))
   for n,b in payload.items():assert (dest/n).read_bytes()==b
  assert sha(R/'env/lib/python3.10/site-packages/torch/_inductor/codegen/triton.py')=='2f0105a59f6e341c2e8a695e02e192e2603a6ef31e6e914c8b6dd33b9a46235a'
  source=R/'tools/search-r1-32b-grpo100-source-v4';manifest=source/'r1-files.json'
  assert sha(manifest)=='12da9b40250360f5a9448d546c93949186eed2e35bf80be9250a1b5e1db338be'
  members=json.loads(manifest.read_text())['files'];assert len(members)==339
  for n,v in members.items():assert not (source/n).is_symlink() and (source/n).stat().st_size==v['size'] and sha(source/n)==v['sha256'],n
  # Captured previous failure is history; admission requires its actual empty state.
  prior=R/'logs/r1-standard32b-profile2-v14/evidence/owned-cleanup.json'
  assert sha(prior)=='7fd2c26a343f5c85800fa47fa56f9e553ed3ed95c37cc03b96066f48c12fd9cc'
  c=json.loads(prior.read_text());assert c['cleanup']['actual_cgroup_empty'] and c['cleanup']['remaining']==[]
  assert c['own_gpu_records_remaining']==[] and c['own_processes_remaining']==[] and c['baseline_identity_missing']==[] and c['original_gpu_baseline_missing']==[]
  terminal=R/'logs/standard3209-captured-controller-terminal.json';assert sha(terminal)=='9bdb19180e5108dca6e5cad10ee29f1e2e411010466b32c729679fb5fbd18f17'
  tr=json.loads(terminal.read_text());assert tr['returncode']==0 and tr['job_id']==3209 and 'JobState=FAILED' in tr['stdout'] and 'ExitCode=1:0' in tr['stdout']
  result=R/'logs/r1-standard32b-profile2-v14/evidence/train-result.json';assert sha(result)=='624a9b7e8124ebe93100f2cd6eca6dbbdac6af4e42eddd5e435b60556cafaece' and json.loads(result.read_text())['status']=='FAILED' and json.loads(result.read_text())['cleanup_passed'] is False
  # Current proc/cgroup admission does not treat historic empty as current.
  for proc in pathlib.Path('/proc').iterdir():
   if not proc.name.isdecimal():continue
   try:cg=(proc/'cgroup').read_text()
   except FileNotFoundError:continue
   assert '/job_3209/' not in cg and not cg.rstrip().endswith('/job_3209'),'Old job live member '+str(proc)
  for ident in c['identities'].values():
   proc=pathlib.Path('/proc')/str(ident['pid'])
   try:stat=(proc/'stat').read_text()
   except FileNotFoundError:continue
   tail=stat.rsplit(')',1)[1].split();assert int(tail[19])!=int(ident['starttime_ticks']),'Old registered PID still exists '+str(proc)
  queue=subprocess.check_output(['squeue','-h','-w','h100-11','-o','%i|%j|%T'],text=True,timeout=10)
  assert not any('entropy-device-proof' in row or 'r1-32b-grpo' in row for row in queue.splitlines()),queue
  # Original entry fresh8 Slurm FD leases/double15 guard precedes any CUDA.
  subprocess.run(['bash','-n',str(target/'probe.sbatch')],check=True)
  d=dict(status='UNKNOWN_BEFORE_SUBMISSION',epoch=time.time(),archive_sha256=F['archive_sha256'],manifest_sha256=F['manifest_sha256'],dependency_pins=pins,scope='CUDA_ENTROPY_VALUE_GRADIENT_AND_NO_AUTOTUNE_BENCH_NOT_CAPACITY_NOT_TRAINING');save(record,d)
  r=subprocess.run(['sbatch','--parsable',str(target/'probe.sbatch')],text=True,capture_output=True,timeout=20)
  d.update(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr);save(record,d);assert r.returncode==0,d
  job=r.stdout.strip().split(';')[0];assert job.isdecimal();d.update(status='SUBMITTED',job_id=int(job));save(record,d)
 if d.get('terminal'):
  saved=pathlib.Path(d['terminal']['path']);assert sha(saved)==d['terminal']['sha256'];raw=json.loads(saved.read_text());assert raw['job_id']==d['job_id'] and raw['returncode']==0
  print(json.dumps(d));raise SystemExit
 job=d['job_id'];deadline=time.monotonic()+1350;observations=R/'logs'/('entropy-device-proof-v1-r1-final-'+str(job)+'-observations.jsonl')
 while time.monotonic()<deadline:
  try:
   r=subprocess.run(['scontrol','show','job','-dd',str(job)],text=True,capture_output=True,timeout=min(5,max(.1,deadline-time.monotonic())))
   row=dict(job_id=job,epoch=time.time(),returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
   fields=dict(re.findall(r'(\w+)=([^\s]+)',r.stdout));ended=r.returncode==0 and fields.get('JobId')==str(job) and fields.get('JobState') in ('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL')
  except (OSError,subprocess.TimeoutExpired) as e:row=dict(job_id=job,epoch=time.time(),transport_error=repr(e));ended=False
  with observations.open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
  if ended:
   terminal=R/'logs'/('entropy-device-proof-v1-r1-final-'+str(job)+'-actual-final.json');assert not terminal.exists(),'Do not replace original terminal';save(terminal,row);d['terminal']={'path':str(terminal),'sha256':sha(terminal)};save(record,d);break
  time.sleep(min(2,max(0,deadline-time.monotonic())))
 print(json.dumps(d))
'''.replace('FREEZE',repr(F))
compile(REMOTE,'device-operation','exec')
if '--render-only' in sys.argv:
 pathlib.Path('outputs/entropy-device-proof-v1-r1-final-operation-r2-render.py').write_text(REMOTE);print('RENDERED_SOURCE_ONLY');raise SystemExit
assert '--submit' in sys.argv,'Requires explicit submit flag after independent source review'
# ROOT already transferred all three pinned immutable archives; no duplicate transfer.
r=send(REMOTE,attempts=1,timeout=1450)
pathlib.Path('outputs/entropy-device-proof-v1-r1-final-operation-invocation.log').write_bytes(r.stdout+r.stderr)
sys.stdout.buffer.write(r.stdout);sys.stderr.buffer.write(r.stderr);raise SystemExit(r.returncode)
