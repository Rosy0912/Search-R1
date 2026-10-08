import pathlib,hashlib,base64,json,os,stat,datetime,subprocess,sys
ROOT=pathlib.Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-unclipped-multitraj200-20261006');REPO=ROOT/'repo-fcecbea5d939';EXPECTED='fcecbea5d939e41dcfb3b118a26ef049a9bda3d6'
assert os.getuid()==0 and ROOT.resolve(strict=True)==ROOT
q=json.load(sys.stdin);raws={n:base64.b64decode(x,validate=True) for n,x in q['files'].items()};assert hashlib.sha256(raws['manifest.json']).hexdigest()==q['manifest_sha256']=='59af58ff4e80a224e9d55e68add3524217c32c22a084c45188dc6dbe631a5da1';m=json.loads(raws['manifest.json']);assert set(raws)==set(m['files'])|{'manifest.json'}
def git():
 h=subprocess.run(['git','--no-optional-locks','-C',str(REPO),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip();s=subprocess.run(['git','--no-optional-locks','-C',str(REPO),'status','--porcelain'],capture_output=True,text=True,check=True).stdout;assert h==EXPECTED and not s;return {'head':h,'status':s}
def sync(p):
 fd=os.open(p,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:os.fsync(fd)
 finally:os.close(fd)
before=git();D=ROOT/'tools/terminal3304-originals-59af58ff4e80';assert D.parent.resolve(strict=True)==D.parent and D.parent.stat().st_uid==0;D.mkdir(mode=0o700);sync(D.parent);written=[]
for n in list(m['files'])+['manifest.json']:
 assert pathlib.Path(n).name==n and n not in ('.','..');b=raws[n]
 if n!='manifest.json':assert len(b)==m['files'][n]['bytes'] and hashlib.sha256(b).hexdigest()==m['files'][n]['sha256']
 fd=os.open(D/n,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o444)
 with os.fdopen(fd,'wb') as f:f.write(b);f.flush();os.fsync(f.fileno())
 os.chmod(D/n,0o444);written.append(n)
sync(D);os.chmod(D,0o555);sync(D);sync(D.parent);records={}
for n in written:
 p=D/n;a=p.lstat();assert stat.S_ISREG(a.st_mode) and a.st_uid==0 and a.st_nlink==1 and stat.S_IMODE(a.st_mode)==0o444 and p.resolve(strict=True)==p
 with p.open('rb') as f:b=f.read();z=os.fstat(f.fileno())
 e=p.lstat();assert (a.st_ino,a.st_size,a.st_mtime_ns,a.st_ctime_ns)==(z.st_ino,z.st_size,z.st_mtime_ns,z.st_ctime_ns)==(e.st_ino,e.st_size,e.st_mtime_ns,e.st_ctime_ns) and b==raws[n];records[n]={'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'uid':a.st_uid,'mode':stat.S_IMODE(a.st_mode),'inode':a.st_ino,'nlink':a.st_nlink}
assert set(p.name for p in D.iterdir())==set(written);after=git();print(json.dumps({'kind':'ACTUAL_EXCLUSIVE_TERMINAL3304_COLLECTOR_SOFTWARE_PACKAGE_PUBLICATION_NOT_TERMINAL_OR_200','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'directory':str(D),'manifest_sha256':q['manifest_sha256'],'members':records,'directory_mode':stat.S_IMODE(D.stat().st_mode),'manifest_written_last':written[-1]=='manifest.json','write_order':written,'source_before':before,'source_after':after,'actual_main':None,'actual_terminal':None},sort_keys=True))
