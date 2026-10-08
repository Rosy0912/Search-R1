
import os,socket,stat,pathlib,hashlib,base64,json,subprocess,datetime
from pathlib import Path
assert os.getuid()==0 and socket.gethostname().split('.')[0]=='h100-10'
root=Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-unclipped-multitraj200-20261006');base=root/'runs/formal-3304/evidence'
dirs=[base/'c5bf-native-cpu-launch-once-v1',base/'c5bf-native-cpu-preparation-once-v1'];result={};total=0;inodes=set()
for directory in dirs:
 assert directory.resolve(strict=True)==directory and directory.stat().st_uid==0
 for p in sorted(directory.iterdir()):
  a=p.lstat(); assert p.resolve(strict=True)==p and stat.S_ISREG(a.st_mode) and a.st_uid==0 and a.st_nlink==1 and a.st_size<8*1024**2
  fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
  try:
   b=os.fstat(fd);chunks=[]
   while True:
    chunk=os.read(fd,1024**2)
    if not chunk:break
    chunks.append(chunk)
   c=os.fstat(fd)
  finally:os.close(fd)
  d=p.lstat();keys=('st_dev','st_ino','st_size','st_uid','st_mode','st_nlink','st_mtime_ns','st_ctime_ns')
  assert all(getattr(a,k)==getattr(s,k) for s in (b,c,d) for k in keys)
  raw=b''.join(chunks);total+=len(raw);assert total<12*1024**2 and len(raw)==a.st_size and (a.st_dev,a.st_ino) not in inodes
  inodes.add((a.st_dev,a.st_ino));name=directory.name+'/'+p.name
  result[name]={'physical_path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'device':a.st_dev,'inode':a.st_ino,'uid':a.st_uid,'mode':stat.S_IMODE(a.st_mode),'nlink':a.st_nlink,'mtime_ns':a.st_mtime_ns,'ctime_ns':a.st_ctime_ns,'base64':base64.b64encode(raw).decode()}
controller=json.loads(base64.b64decode(result['c5bf-native-cpu-preparation-once-v1/intent.original.json']['base64']))['execution']
child=json.loads(base64.b64decode(result['c5bf-native-cpu-preparation-once-v1/native-cpu-preparation.json']['base64']))['execution']
former=[]
for identity in (controller,child):
 pid=identity['pid'];p=Path('/proc')/str(pid);present=p.exists();same=False;state=None;start=None
 if present:
  try:
   fields=(p/'stat').read_text().rsplit(')',1)[1].split();start=int(fields[19]);state=fields[0];same=start==int(identity['start_ticks'])
  except FileNotFoundError:present=False
 assert not(same and state not in ('Z','X'))
 former.append({'pid':pid,'original_start_ticks':identity['start_ticks'],'current_present':present,'current_start_ticks':start,'same_process_identity':same,'current_state':state,'scope':'CPUchecker/controller only, not training resource release'})
gits=[]
for repo,wanted in [(root/'repo-fcecbea5d939','fcecbea5d939e41dcfb3b118a26ef049a9bda3d6'),(root/'repo-final200-eval-c5bf4c8db335','c5bf4c8db33562a259fa227a940dc69543901b98')]:
 for args in [('rev-parse','HEAD'),('status','--porcelain','--untracked-files=all')]:
  argv=['git','--no-optional-locks','-C',str(repo)]+list(args);r=subprocess.run(argv,capture_output=True,text=True);assert r.returncode==0
  assert r.stdout.strip()==(wanted if args[0]=='rev-parse' else '')
  gits.append({'argv':argv,'stdout':r.stdout,'stderr':r.stderr,'returncode':r.returncode,'waited':True})
print(json.dumps({'version':1,'node':socket.gethostname(),'uid':os.getuid(),'collector_pid':os.getpid(),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':result,'former_CPU_processes':former,'source_git_originals':gits,'total_member_bytes':total,'scope':'Closed CPU preparation originals only, noGPU/model/evaluation/200/terminal training claim'},sort_keys=True))
