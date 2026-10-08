
import pathlib,os,socket,stat,json,hashlib,datetime,subprocess
from pathlib import Path
assert os.getuid()==0 and socket.gethostname().split('.')[0]=='h100-10'
root=Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-unclipped-multitraj200-20261006');target=root/'tools/final200-prerequisites-85f014e167c8'
result={'target':str(target),'exists':os.path.lexists(target),'node':socket.gethostname(),'uid':os.getuid(),'reader_pid':os.getpid(),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'publisher_true_wait':None,'scope':'Read-only recovery of unknown publisher outcome, never inferred wait0'}
if result['exists']:
 s=target.lstat();assert target.resolve(strict=True)==target and stat.S_ISDIR(s.st_mode) and s.st_uid==0
 result['directory_stat']={'device':s.st_dev,'inode':s.st_ino,'mode':stat.S_IMODE(s.st_mode),'mtime_ns':s.st_mtime_ns,'ctime_ns':s.st_ctime_ns}
 members={}
 for p in sorted(target.iterdir()):
  a=p.lstat();assert p.resolve(strict=True)==p and stat.S_ISREG(a.st_mode) and a.st_uid==0 and a.st_nlink==1
  fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
  try:
   b=os.fstat(fd);chunks=[]
   while True:
    v=os.read(fd,1024**2)
    if not v:break
    chunks.append(v)
   c=os.fstat(fd)
  finally:os.close(fd)
  d=p.lstat();keys=('st_dev','st_ino','st_size','st_uid','st_mode','st_nlink','st_mtime_ns','st_ctime_ns');stable=all(getattr(a,k)==getattr(s,k) for s in (b,c,d) for k in keys)
  raw=b''.join(chunks);members[p.name]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'mode':stat.S_IMODE(a.st_mode),'uid':a.st_uid,'nlink':a.st_nlink,'inode':a.st_ino,'device':a.st_dev,'stable_read':stable}
 result['members']=members
processes=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  if p.stat().st_uid!=0:continue
  raw=(p/'cmdline').read_bytes()
  if b'exclusiveStrict200ToolsPublisher' not in raw:continue
  # Exclude this read-only script itself, whose source contains that literal.
  argv=raw.rstrip(b'\0').split(b'\0')
  if p.name==str(os.getpid()):continue
  if not any(x in (b'python3',b'/usr/bin/python3') for x in argv[:1]):continue
  st=(p/'stat').read_text();status=(p/'status').read_text()
  processes.append({'pid':int(p.name),'cmdline_sha256':hashlib.sha256(raw).hexdigest(),'stat_original':st,'status_original':status,'cgroup_original':(p/'cgroup').read_text(),'fd_paths':{n.name:os.readlink(n) for n in (p/'fd').iterdir() if n.name in ('0','1','2')}})
 except (FileNotFoundError,ProcessLookupError):pass
result['matching_live_publisher_processes']=processes
repo=root/'repo-fcecbea5d939';commands=[]
for args in [('rev-parse','HEAD'),('status','--porcelain','--untracked-files=all')]:
 argv=['git','--no-optional-locks','-C',str(repo)]+list(args);p=subprocess.run(argv,capture_output=True,text=True);commands.append({'argv':argv,'stdout':p.stdout,'stderr':p.stderr,'returncode':p.returncode,'waited':True})
result['training_source_originals']=commands
print(json.dumps(result,sort_keys=True))
