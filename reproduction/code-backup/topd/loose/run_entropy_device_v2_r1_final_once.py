"""Once own Slurm8 tiny CUDA original-math proof; never training/capacity."""
import pathlib,json,hashlib,sys,subprocess
sys.path.insert(0,'work/g1-native-cpu-deploy-v6')
from stream_transport import send
F=json.loads(pathlib.Path('outputs/entropy-device-proof-v2-r1-final-freeze.json').read_text())
assert F['archive_sha256']=='7d9bfe6d6623a2ec1b11dbc253dfa55d03b35fe73cb24071b7f91f3839cbc436'
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
record=R/'logs/entropy-device-proof-v2-r1-final-submission.json'
with record.with_suffix('.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 if record.exists():
  d=json.loads(record.read_text());assert d['status']=='SUBMITTED' and d['archive_sha256']==F['archive_sha256'],'UNKNOWN or different record: inspect; never resubmit'
 else:
  archive=R/'tools/entropy-device-proof-v2-r1-final.tar.gz';assert sha(archive)==F['archive_sha256']
  data={}
  with tarfile.open(archive) as t:
   for m in t.getmembers():
    q=pathlib.PurePosixPath(m.name);assert m.isfile() and not q.is_absolute() and q.parts[0]=='entropy-device-proof-v2-r1-final' and '..' not in q.parts
    name=str(q.relative_to(q.parts[0]));assert name not in data;data[name]=t.extractfile(m).read()
  assert len(data)==7 and hashlib.sha256(data['files.json']).hexdigest()==F['manifest_sha256']
  assert json.loads(data['files.json'])==F['files'] and set(data)==set(F['files'])|{'files.json'}
  for n,h in F['files'].items():assert hashlib.sha256(data[n]).hexdigest()==h
  target=R/'tools/entropy-device-proof-v2-r1-final'
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
  prior=R/'logs/entropy-device-proof-3212/parent-result.json';assert sha(prior)=='5aea119335250c51606f6a5fd654876af67afb461da5eb384e566db90d397c94'
  c=json.loads(prior.read_text());assert c['status']=='FAIL_ENTROPY_PARENT_NOT_CAPACITY_NOT_TRAINING' and c['job']=='3212'
  assert c['cleanup']['actual_cgroup_empty'] and c['cleanup']['remaining']==[] and c['baseline_preserved'] is True and c['parent_cuda_initialized'] is False
  assert c['final']['identities']==c['baseline']['identities'] and c['final']['applications']==c['baseline']['applications']
  terminal=R/'logs/entropy-device-proof-v1-r1-final-3212-actual-final.json';assert sha(terminal)=='8d640a0b1d6696aa90e5ccace27c234d3c1c333c85ea044382ebd3dc64e2c659'
  tr=json.loads(terminal.read_text());assert tr['returncode']==0 and tr['job_id']==3212 and 'JobState=FAILED' in tr['stdout'] and 'ExitCode=1:0' in tr['stdout']
  result=R/'logs/entropy-device-proof-3212/worker-result.json';assert sha(result)=='61cbeb9dbf322c138974e8b9b7f134f0e26491f08f2b73307819dc1d1703dfd1';w=json.loads(result.read_text());assert w['status']=='FAIL_ENTROPY_DEVICE_CORRECTNESS_NOT_CAPACITY_NOT_TRAINING' and w['error']=='No actual candidate generated/precompile metadata'
  for proc in pathlib.Path('/proc').iterdir():
   if not proc.name.isdecimal():continue
   try:cg=(proc/'cgroup').read_text()
   except FileNotFoundError:continue
   assert '/job_3212/' not in cg and not cg.rstrip().endswith('/job_3212'),'Old job live member '+str(proc)
  for ident in [c['worker_identity']]+[row['identity'] for row in c['cleanup']['signals']]:
   proc=pathlib.Path('/proc')/str(ident['pid'])
   try:stat=(proc/'stat').read_text()
   except FileNotFoundError:continue
   tail=stat.rsplit(')',1)[1].split();assert int(tail[19])!=int(ident['starttime_ticks']),'Old owned PID still exists '+str(proc)
  queue=subprocess.check_output(['squeue','-h','-w','h100-11','-o','%i|%j|%T'],text=True,timeout=10)
  assert not any('entropy-device-proof' in row or 'r1-32b-grpo' in row for row in queue.splitlines()),queue
  # Original entry fresh8 Slurm FD leases/double15 guard precedes any CUDA.
  subprocess.run(['bash','-n',str(target/'probe.sbatch')],check=True)
  # Existing reviewed official static headers; no global install or download.
  hb=R/'assets/compiler-headers/python3.10-3.10.12-22.04.18-amd64';hi=hb/'inventory.json';hs=hb/'cpu-capi-smoke.json'
  assert not hb.is_symlink() and sha(hi)=='2fad02f3fdc7e74f7f734880022eb2e2b2edc94c24d597dd0b947d6a881a7201' and sha(hs)=='5286484a3b15a8a69344b7b1f9cc9a766af40ade951dc632a6ed0f1032af7871'
  inventory=json.loads(hi.read_text());receipt=json.loads(hs.read_text());overlay=hb/'overlay';actualfiles={};actuallinks={}
  assert inventory['actual_node']=='h100-11' and inventory['actual_python_deb_version']=='3.10.12-1~22.04.18' and inventory['overlay_directory']==str(overlay)
  for directory,dirs,names in os.walk(overlay,followlinks=False):
   for name in dirs+names:
    hp=pathlib.Path(directory)/name;rel=str(hp.relative_to(overlay))
    if hp.is_symlink():actuallinks[rel]=os.readlink(hp)
    elif hp.is_file():actualfiles[rel]={'size':hp.stat().st_size,'sha256':sha(hp)}
    else:assert hp.is_dir()
  assert actualfiles==inventory['files'] and actuallinks==inventory['symlinks']
  cpath=str(overlay/'usr/include/python3.10')+':'+str(overlay/'usr/include')
  assert receipt['status']=='CPU_CAPI_COMPILE_IMPORT_PASS_NO_GPU_PROOF' and receipt['inventory_sha256']=='2fad02f3fdc7e74f7f734880022eb2e2b2edc94c24d597dd0b947d6a881a7201' and receipt['CPATH']==cpath and receipt['actual_import']['actual_executable']==str(R/'env/bin/python')
  cache=R/'logs/entropy-device-proof-v2-r1-final-compile-cache';cache.mkdir(mode=0o700,exist_ok=False)
  for name in ('triton','extensions','tmp'):(cache/name).mkdir(mode=0o700)
  submitenv=dict(os.environ,CPATH=cpath,TRITON_CACHE_DIR=str(cache/'triton'),TORCH_EXTENSIONS_DIR=str(cache/'extensions'),TMPDIR=str(cache/'tmp'))
  d=dict(status='UNKNOWN_BEFORE_SUBMISSION',epoch=time.time(),archive_sha256=F['archive_sha256'],manifest_sha256=F['manifest_sha256'],dependency_pins=pins,compiler_headers_inventory_sha256=sha(hi),compiler_headers_smoke_sha256=sha(hs),compiler_cpath=cpath,private_compile_cache=str(cache),scope='CUDA_ENTROPY_VALUE_GRADIENT_AND_NO_AUTOTUNE_BENCH_NOT_CAPACITY_NOT_TRAINING');save(record,d)
  r=subprocess.run(['sbatch','--parsable',str(target/'probe.sbatch')],text=True,capture_output=True,timeout=20,env=submitenv)
  d.update(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr);save(record,d);assert r.returncode==0,d
  job=r.stdout.strip().split(';')[0];assert job.isdecimal();d.update(status='SUBMITTED',job_id=int(job));save(record,d)
 if d.get('terminal'):
  saved=pathlib.Path(d['terminal']['path']);assert sha(saved)==d['terminal']['sha256'];raw=json.loads(saved.read_text());assert raw['job_id']==d['job_id'] and raw['returncode']==0
  print(json.dumps(d));raise SystemExit
 job=d['job_id'];deadline=time.monotonic()+1350;observations=R/'logs'/('entropy-device-proof-v2-r1-final-'+str(job)+'-observations.jsonl')
 while time.monotonic()<deadline:
  try:
   r=subprocess.run(['scontrol','show','job','-dd',str(job)],text=True,capture_output=True,timeout=min(5,max(.1,deadline-time.monotonic())))
   row=dict(job_id=job,epoch=time.time(),returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
   fields=dict(re.findall(r'(\w+)=([^\s]+)',r.stdout));ended=r.returncode==0 and fields.get('JobId')==str(job) and fields.get('JobState') in ('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL')
  except (OSError,subprocess.TimeoutExpired) as e:row=dict(job_id=job,epoch=time.time(),transport_error=repr(e));ended=False
  with observations.open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
  if ended:
   terminal=R/'logs'/('entropy-device-proof-v2-r1-final-'+str(job)+'-actual-final.json');assert not terminal.exists(),'Do not replace original terminal';save(terminal,row);d['terminal']={'path':str(terminal),'sha256':sha(terminal)};save(record,d);break
  time.sleep(min(2,max(0,deadline-time.monotonic())))
 print(json.dumps(d))
'''.replace('FREEZE',repr(F))
compile(REMOTE,'device-operation','exec')
if '--render-only' in sys.argv:
 pathlib.Path('outputs/entropy-device-proof-v2-r1-final-operation-render.py').write_text(REMOTE);print('RENDERED_SOURCE_ONLY');raise SystemExit
assert '--submit' in sys.argv,'Requires explicit submit flag after independent source review'
# ROOT already transferred all three pinned immutable archives; no duplicate transfer.
r=send(REMOTE,attempts=1,timeout=1450)
pathlib.Path('outputs/entropy-device-proof-v2-r1-final-operation-invocation.log').write_bytes(r.stdout+r.stderr)
sys.stdout.buffer.write(r.stdout);sys.stderr.buffer.write(r.stderr);raise SystemExit(r.returncode)
