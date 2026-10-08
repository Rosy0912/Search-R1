
import datetime,hashlib,json,os,sys
from pathlib import Path
sys.path.insert(0,'/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct200-author-20261006/tools/final-acceptance-f5c72c942641')
import checkpoint_overlap as c
import base64
packet=json.loads(sys.stdin.buffer.read());mraw=base64.b64decode(packet['manifest_b64'],validate=True)
assert hashlib.sha256(mraw).hexdigest()==packet['manifest_sha256']=='90c2695a7f46d6cabed19661fae77b5b9734b008e3093b622555807c63dfbfc4'
m=c.unique(mraw);assert m['source_commit']==c.SOURCE
pins={**m['tools'],**m['review_records']};assert set(packet['members'])==set(pins) and len(pins)==26
members={n:base64.b64decode(b,validate=True) for n,b in packet['members'].items()}
assert all(Path(n).name==n and hashlib.sha256(raw).hexdigest()==pins[n] for n,raw in members.items())
before=c.allocation();verified={}
for p,h in m['dependencies'].items():
 raw=c.pinned_bytes(p,h);verified[p]={'physical_path':p,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
assert len(verified)==278
d=c.ROOT/'tools/final-acceptance-90c2695a7f46';c.physical(d.parent,True)
assert not d.exists() and not d.is_symlink()
assert not (c.RUN/'evidence/checkpoint-cpu-75-launch.json').exists()
d.mkdir(mode=0o700);published=[]
for name,raw in list(members.items())+[('manifest.json',mraw)]:
 if name=='manifest.json':
  members_directory=os.open(d,os.O_RDONLY|os.O_DIRECTORY)
  try:os.fsync(members_directory)
  finally:os.close(members_directory)
 p=d/name;fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o444)
 with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 p.chmod(0o444);pin=hashlib.sha256(raw).hexdigest();assert c.pinned_bytes(p,pin)==raw
 published.append({'name':name,'bytes':len(raw),'sha256':pin,'mode':oct(p.stat().st_mode & 0o777)})
assert published[-1]['name']=='manifest.json'
df=os.open(d,os.O_RDONLY|os.O_DIRECTORY)
try:os.fsync(df)
finally:os.close(df)
d.chmod(0o555)
parent=os.open(d.parent,os.O_RDONLY|os.O_DIRECTORY)
try:os.fsync(parent)
finally:os.close(parent)
after=c.allocation()
assert set(p.name for p in d.iterdir())==set(pins)|{'manifest.json'}
print(json.dumps({'scope':'ACTUAL_NEW_TOOLS_DEPLOYMENT_NOT75_CONTENT_OR_TRAINING_CHANGE','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'directory':str(d),'manifest_sha256':packet['manifest_sha256'],'manifest_last':True,'published':published,
 'verified_dependencies':verified,'source_commit':c.SOURCE,'allocation_before':before,'allocation_after':after,
 'directory_mode':oct(d.stat().st_mode & 0o777)},sort_keys=True))
