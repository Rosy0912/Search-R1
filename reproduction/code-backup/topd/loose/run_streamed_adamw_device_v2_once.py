"""Once own Slurm8 tiny CUDA original-math proof; never training/capacity."""
import pathlib,json,hashlib,sys,subprocess
sys.path.insert(0,'work/g1-native-cpu-deploy-v6')
from stream_transport import send
F=json.loads(pathlib.Path('outputs/streamed-adamw-device-parent-v2-freeze.json').read_text())
assert F['archive_sha256']=='c094ec18f1151cd73c6e5f7d80107910a66e2d0660cb773f33fbe42e16b880e5'
assert hashlib.sha256(pathlib.Path(F['archive']).read_bytes()).hexdigest()==F['archive_sha256']
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
record=R/'logs/streamed-adamw-device-v2-submission.json'
with record.with_suffix('.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 if record.exists():
  d=json.loads(record.read_text());assert d['status']=='SUBMITTED' and d['archive_sha256']==F['archive_sha256'],'UNKNOWN or different record: inspect; never resubmit'
 else:
  archive=R/'tools/streamed-adamw-device-parent-v2.tar.gz';assert sha(archive)==F['archive_sha256']
  data={}
  with tarfile.open(archive) as t:
   for m in t.getmembers():
    q=pathlib.PurePosixPath(m.name);assert m.isfile() and not q.is_absolute() and q.parts[0]=='streamed-adamw-device-parent-v2' and '..' not in q.parts
    name=str(q.relative_to(q.parts[0]));assert name not in data;data[name]=t.extractfile(m).read()
  assert len(data)==7 and hashlib.sha256(data['files.json']).hexdigest()==F['manifest_sha256']
  assert json.loads(data['files.json'])==F['files'] and set(data)==set(F['files'])|{'files.json'}
  for n,h in F['files'].items():assert hashlib.sha256(data[n]).hexdigest()==h
  target=R/'tools/streamed-adamw-device-parent-v2'
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
   'tools/qwen25-32b-window-safety-candidate-v5/runtime_safety.py':'9e41464d49054a1765fed29f5839f9fd8c568f0040aedbf6cf36b3c096adb3c6',
   'tools/search-r1-32b-grpo100-tools-v9/retrieval_startup.py':'1f401e123b97e240f1040beb9ee02fbcb525d86fa90b9494951f231f9003ecf4'}
  for n,h in pins.items():assert not (R/n).is_symlink() and sha(R/n)==h,n
  source=R/'tools/search-r1-32b-grpo100-source-v4';manifest=source/'r1-files.json'
  assert sha(manifest)=='12da9b40250360f5a9448d546c93949186eed2e35bf80be9250a1b5e1db338be'
  members=json.loads(manifest.read_text())['files'];assert len(members)==339
  for n,v in members.items():assert not (source/n).is_symlink() and (source/n).stat().st_size==v['size'] and sha(source/n)==v['sha256'],n
  # Captured previous failure is history; admission requires its actual empty state.
  prior=R/'logs/r1-standard32b-profile2-v9/evidence/owned-cleanup.json'
  assert sha(prior)=='51deb7e3645db562cc3f76adacc3a0b1a22d099106f8d03bd8261653165c5a91'
  c=json.loads(prior.read_text());assert c['cleanup']['actual_cgroup_empty'] and c['cleanup']['remaining']==[]
  assert c['own_gpu_records_remaining']==[] and c['own_processes_remaining']==[] and c['baseline_identity_missing']==[] and c['original_gpu_baseline_missing']==[]
  terminal=R/'logs/standard3197-captured-controller-terminal.json';assert sha(terminal)=='fb8f493c69b1f68aa9b1d19437dac74831d7c99e277ff6fcf0a7affaee9fe9d3'
  tr=json.loads(terminal.read_text());assert tr['returncode']==0 and tr['job_id']==3197 and 'JobState=FAILED' in tr['stdout'] and 'ExitCode=1:0' in tr['stdout']
  result=R/'logs/r1-standard32b-profile2-v9/evidence/train-result.json';assert sha(result)=='e44536779e4133f3a8d426f949fb291a6c0cb805d67531d2114b98c5c319eee7' and json.loads(result.read_text())['status']=='FAILED'
  # Current proc/cgroup admission does not treat historic empty as current.
  for proc in pathlib.Path('/proc').iterdir():
   if not proc.name.isdecimal():continue
   try:cg=(proc/'cgroup').read_text()
   except FileNotFoundError:continue
   assert '/job_3197/' not in cg and not cg.rstrip().endswith('/job_3197'),'Old job live member '+str(proc)
  for ident in c['identities'].values():
   proc=pathlib.Path('/proc')/str(ident['pid'])
   try:stat=(proc/'stat').read_text()
   except FileNotFoundError:continue
   tail=stat.rsplit(')',1)[1].split();assert int(tail[19])!=int(ident['starttime_ticks']),'Old registered PID still exists '+str(proc)
  queue=subprocess.check_output(['squeue','-h','-w','h100-11','-o','%i|%j|%T'],text=True,timeout=10)
  assert not any('streamed-adamw-device-proof' in row or 'r1-32b-grpo' in row for row in queue.splitlines()),queue
  # Original entry fresh8 Slurm FD leases/double15 guard precedes any CUDA.
  subprocess.run(['bash','-n',str(target/'probe.sbatch')],check=True)
  d=dict(status='UNKNOWN_BEFORE_SUBMISSION',epoch=time.time(),archive_sha256=F['archive_sha256'],manifest_sha256=F['manifest_sha256'],dependency_pins=pins,scope='CUDA_DEVICE_MATH_ONLY_NOT_CAPACITY_NOT_TRAINING');save(record,d)
  r=subprocess.run(['sbatch','--parsable',str(target/'probe.sbatch')],text=True,capture_output=True,timeout=20)
  d.update(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr);save(record,d);assert r.returncode==0,d
  job=r.stdout.strip().split(';')[0];assert job.isdecimal();d.update(status='SUBMITTED',job_id=int(job));save(record,d)
 if d.get('terminal'):
  saved=pathlib.Path(d['terminal']['path']);assert sha(saved)==d['terminal']['sha256'];raw=json.loads(saved.read_text());assert raw['job_id']==d['job_id'] and raw['returncode']==0
  print(json.dumps(d));raise SystemExit
 job=d['job_id'];deadline=time.monotonic()+660;observations=R/'logs'/('streamed-adamw-device-v2-'+str(job)+'-observations.jsonl')
 while time.monotonic()<deadline:
  try:
   r=subprocess.run(['scontrol','show','job','-dd',str(job)],text=True,capture_output=True,timeout=min(5,max(.1,deadline-time.monotonic())))
   row=dict(job_id=job,epoch=time.time(),returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
   fields=dict(re.findall(r'(\w+)=([^\s]+)',r.stdout));ended=r.returncode==0 and fields.get('JobId')==str(job) and fields.get('JobState') in ('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL')
  except (OSError,subprocess.TimeoutExpired) as e:row=dict(job_id=job,epoch=time.time(),transport_error=repr(e));ended=False
  with observations.open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
  if ended:
   terminal=R/'logs'/('streamed-adamw-device-v2-'+str(job)+'-actual-final.json');assert not terminal.exists(),'Do not replace original terminal';save(terminal,row);d['terminal']={'path':str(terminal),'sha256':sha(terminal)};save(record,d);break
  time.sleep(min(2,max(0,deadline-time.monotonic())))
 print(json.dumps(d))
'''.replace('FREEZE',repr(F))
compile(REMOTE,'device-operation','exec')
if '--render-only' in sys.argv:
 pathlib.Path('outputs/streamed-adamw-device-v2-operation-render.py').write_text(REMOTE);print('RENDERED_SOURCE_ONLY');raise SystemExit
assert '--submit' in sys.argv,'Requires explicit submit flag after independent source review'
subprocess.run([sys.executable,'work/transfer_file.py',F['archive'],'/dataset/shaoshitong/search-r1-rl/topd/tools/streamed-adamw-device-parent-v2.tar.gz'],check=True)
r=send(REMOTE,attempts=1,timeout=750)
pathlib.Path('outputs/streamed-adamw-device-v2-operation-invocation.log').write_bytes(r.stdout+r.stderr)
sys.stdout.buffer.write(r.stdout);sys.stderr.buffer.write(r.stderr);raise SystemExit(r.returncode)
