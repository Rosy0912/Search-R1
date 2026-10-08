import pathlib,os,stat,json,base64,hashlib,socket,subprocess,sys,datetime
assert socket.gethostname().split('.')[0]=='h100-10' and os.getuid()==0
root=pathlib.Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-unclipped-multitraj200-20261006');repo=root/'repo-fcecbea5d939';parent=root/'tools'
assert parent.resolve(strict=True)==parent and parent.is_dir() and parent.stat().st_uid==0
assert subprocess.check_output(['git','--no-optional-locks','-C',str(repo),'rev-parse','HEAD'],text=True).strip()=='fcecbea5d939e41dcfb3b118a26ef049a9bda3d6'
assert not subprocess.check_output(['git','--no-optional-locks','-C',str(repo),'status','--porcelain'],text=True).strip()
payload=json.loads(sys.stdin.buffer.read());raw=base64.b64decode(payload['manifest_base64'],validate=True);sha=hashlib.sha256(raw).hexdigest();assert sha==payload['manifest_sha256']=='ee31da65570318227ad6b037187cffa14e4dc489bac2a504d2bc54010c95da05'
m=json.loads(raw);assert set(m)=={'version','kind','source_commit','tools'} and m['version']==1 and m['kind']=='IMMUTABLE_C5BF_NATIVE_CPU_PREPARATION_SOFTWARE_BUNDLE' and m['source_commit']=='c5bf4c8db33562a259fa227a940dc69543901b98'
assert set(payload['files'])==set(m['tools']) and len(m['tools'])==22
files={n:base64.b64decode(payload['files'][n],validate=True) for n in m['tools']}
assert all(pathlib.PurePosixPath(n).name==n and n!='manifest.json' and hashlib.sha256(v).hexdigest()==m['tools'][n] for n,v in files.items())
def syncdir(p):
 fd=os.open(p,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)
out=parent/('final200-eval-native-cpu-'+sha[:12]);out.mkdir(mode=0o700);syncdir(parent);syncdir(out);order=[]
for name,value in sorted(files.items()):
 fd=os.open(out/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,'wb') as f:f.write(value);f.flush();os.fsync(f.fileno());os.fchmod(f.fileno(),0o444);os.fsync(f.fileno())
 syncdir(out);order.append(name)
fd=os.open(out/'manifest.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno());os.fchmod(f.fileno(),0o444);os.fsync(f.fileno())
order.append('manifest.json');syncdir(out);os.chmod(out,0o555);syncdir(out);syncdir(parent)
records={}
for name,want in dict(m['tools'],**{'manifest.json':sha}).items():
 p=out/name;s=p.stat();assert p.resolve(strict=True)==p and stat.S_ISREG(s.st_mode) and s.st_uid==0 and s.st_nlink==1 and stat.S_IMODE(s.st_mode)==0o444
 value=p.read_bytes();assert hashlib.sha256(value).hexdigest()==want
 records[name]={'sha256':want,'bytes':len(value),'inode':s.st_ino,'device':s.st_dev,'uid':s.st_uid,'mode':stat.S_IMODE(s.st_mode),'nlink':s.st_nlink}
assert set(x.name for x in out.iterdir())==set(records)
assert not subprocess.check_output(['git','--no-optional-locks','-C',str(repo),'status','--porcelain'],text=True).strip()
print(json.dumps({'version':1,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'node':socket.gethostname(),'pid':os.getpid(),'uid':os.getuid(),'directory':str(out),'manifest_sha256':sha,'member_files':records,'actual_exclusive_write_order':order,'manifest_written_last':order[-1]=='manifest.json','all_files_and_directories_fsynced':True,'directory_mode':stat.S_IMODE(out.stat().st_mode),'source_clean_after':True,'scope':'Actual isolated c5bf CPU-preparation tool publication only; no native tool execution, actual200/model/GPU/eval/training/receipt/newallocation'},sort_keys=True))
