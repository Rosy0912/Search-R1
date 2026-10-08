import base64,datetime,hashlib,json,os,sys
from pathlib import Path
sys.path.insert(0,'/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct200-author-20261006/tools/final-acceptance-90c2695a7f46');import checkpoint_overlap as c
c.tools('90c2695a7f46d6cabed19661fae77b5b9734b008e3093b622555807c63dfbfc4');before=c.allocation();d=c.RUN/'evidence/checkpoint-175-cpu-inspection-once-v1';c.physical(d,True);files={}
paths=[c.RUN/'evidence/checkpoint-cpu-175-launch.json']+sorted(p for p in d.rglob('*') if p.is_file() and p.suffix in ('.json','.txt'))
for p in paths:
 c.physical(p);b=p.stat();fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
 with os.fdopen(fd,'rb') as f:
  op=os.fstat(f.fileno());raw=f.read(b.st_size);ap=os.fstat(f.fileno())
 a=p.stat();fields=('st_dev','st_ino','st_uid');assert tuple(getattr(b,k) for k in fields)==tuple(getattr(op,k) for k in fields)==tuple(getattr(ap,k) for k in fields)==tuple(getattr(a,k) for k in fields)
 assert len(raw)==b.st_size
 fields2=fields+('st_size','st_mtime_ns','st_ctime_ns');stable=all(tuple(getattr(v,k) for k in fields2)==tuple(getattr(b,k) for k in fields2) for v in [op,ap,a])
 n=str(p.relative_to(c.RUN));files[n]={'physical_path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'base64':base64.b64encode(raw).decode(),'unchanged_during_read':stable,'uid':a.st_uid,'device':a.st_dev,'inode':a.st_ino}
print(json.dumps({'scope':'ACTUAL175_CPU_OUTPUT_ORIGINALS_NOT_CONTENT_ACCEPTANCE','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'allocation_before':before,'allocation_after':c.allocation(),'files':files},sort_keys=True))
