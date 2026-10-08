
import base64,datetime,gzip,hashlib,json,os,socket,stat
from pathlib import Path
root=Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct200-author-20261006/runs/formal-3261')
paths={
 'driver.out':Path('/tmp/o4e-3261/ray/session_2026-10-06_03-56-08_188781_1526172/logs/worker-8350026ae4bc8387c3d3b6880792734dd21e46ad5a9c09e08669a24e-02000000-1533312.out'),
 'train.log':root/'train.log',
 'resolved-config.json':root/'full_state/global_step_5/resolved-config.json',
 'driver200.json':root/'full_state/global_step_200/driver.json'
}
result={'scope':'BOUND_SMALL_ORIGINALS_FOR_CURVE_NOT_FINAL_ACCEPTANCE','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'node':socket.gethostname(),'files':{}}
assert result['node']=='h100-10' and os.getuid()==0
def ident(s):return [s.st_dev,s.st_ino,s.st_uid,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
for name,path in paths.items():
 assert path.resolve(strict=True)==path
 fd=os.open(str(path),os.O_RDONLY|os.O_NOFOLLOW)
 with os.fdopen(fd,'rb') as f:
  before=os.fstat(f.fileno());assert stat.S_ISREG(before.st_mode) and before.st_uid==0 and before.st_size<32*1024*1024
  raw=f.read();after=os.fstat(f.fileno())
 assert len(raw)==before.st_size and ident(before)==ident(after)==ident(path.stat())
 packed=gzip.compress(raw,compresslevel=6,mtime=0)
 result['files'][name]={'path':str(path),'identity':ident(before),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'gzip_sha256':hashlib.sha256(packed).hexdigest(),'gzip_base64':base64.b64encode(packed).decode()}
print(json.dumps(result,sort_keys=True))
