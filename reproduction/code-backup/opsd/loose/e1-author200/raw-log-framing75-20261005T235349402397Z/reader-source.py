
import base64,datetime,hashlib,json,os,sys
from pathlib import Path
T=Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct200-author-20261006/tools/final-acceptance-f5c72c942641');sys.path.insert(0,str(T));import checkpoint_overlap as c
c.tools('f5c72c9426413a18e465030efb714c9015fa84274baa3fc4e401383adc96b3fd');allocation=c.allocation();files={}
for name,p in [('driver.out',Path(os.readlink('/proc/1533312/fd/1'))),('train.log',c.RUN/'train.log')]:
 c.physical(p);before=p.stat();fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
 with os.fdopen(fd,'rb') as f:
  opened=os.fstat(f.fileno());assert (opened.st_dev,opened.st_ino,opened.st_uid)==(before.st_dev,before.st_ino,before.st_uid);raw=f.read(before.st_size);opened_after=os.fstat(f.fileno())
 after=p.stat();assert (after.st_dev,after.st_ino,after.st_uid)==(before.st_dev,before.st_ino,before.st_uid)==(opened_after.st_dev,opened_after.st_ino,opened_after.st_uid)
 files[name]={'physical_path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'base64':base64.b64encode(raw).decode(),'source_offset':0,'same_opened_inode':True,'complete_prefix_at_start':len(raw)==before.st_size}
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'READ_ONLY_RAW_ORIGINALS_FOR_FRAMING_AUDIT_NOT_NEW_PROGRESS_ACCEPTANCE','allocation':allocation,'files':files},sort_keys=True))
