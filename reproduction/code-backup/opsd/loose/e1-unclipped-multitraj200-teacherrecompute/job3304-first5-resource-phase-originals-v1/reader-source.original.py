import pathlib,hashlib,os,stat,json,base64,socket
root=pathlib.Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-unclipped-multitraj200-20261006/runs/formal-3304/first5-resource-training-phase-v1');assert os.getuid()==0 and socket.gethostname().split('.')[0]=='h100-10' and root.resolve(strict=True)==root and root.stat().st_uid==0
assert stat.S_IMODE(root.stat().st_mode)==0o555
def read(p,expected):
 a=p.lstat();assert p.resolve(strict=True)==p and stat.S_ISREG(a.st_mode) and a.st_uid==0 and a.st_nlink==1 and a.st_size<=32*1024**2
 fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
 try:
  z=os.fstat(fd);assert (a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns,a.st_ctime_ns)==(z.st_dev,z.st_ino,z.st_size,z.st_mtime_ns,z.st_ctime_ns)
  b=b''
  while True:
   x=os.read(fd,1024**2)
   if not x:break
   b+=x
  f=os.fstat(fd);e=p.lstat();assert z==f==e and hashlib.sha256(b).hexdigest()==expected
  return {'path':str(p),'sha256':expected,'bytes':len(b),'device':z.st_dev,'inode':z.st_ino,'base64':base64.b64encode(b).decode()}
 finally:os.close(fd)
r=read(root/'manifest.json','b2300073d9920e3d26a02edea8905c9fbfe374334e60c8c6b6a9c8c2a9f9d9ac');m=json.loads(base64.b64decode(r['base64']));rows={'manifest.json':r}
for n in ['all-resources.ndjson.gz','training-resources.ndjson.gz']:
 rec=m['files'][n];assert rec['path']==str(root/n);rows[n]=read(root/n,rec['sha256'])
print(json.dumps({'files':rows,'scope':'Exact immutable ALL and complete contiguous training-range originals, not acceptance'}))
