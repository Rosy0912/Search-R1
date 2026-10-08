
import base64,datetime,hashlib,json,os,re,sys
from pathlib import Path
T=Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-instruct200-author-20261006/tools/final-acceptance-f5c72c942641');sys.path.insert(0,str(T));import checkpoint_overlap as c
c.tools('f5c72c9426413a18e465030efb714c9015fa84274baa3fc4e401383adc96b3fd');allocation=c.allocation()
closure=c.RUN/'bindings/import_closure.json';closed=c.unique(c.pinned_bytes(closure,'3842c259b30c524e5ba5e067eb5b37f6dbac1042e162a5d93dd23511c13b5c9d'))
rec=closed['modules']['torch.utils._contextlib'];helper=Path(rec['path']);c.pinned_bytes(helper,rec['sha256']);torch_root=helper.parent.parent
source=torch_root/'distributed/fsdp/fully_sharded_data_parallel.py';stderr=Path(os.readlink('/proc/1533910/fd/2'));process=c.process(1533910,training_job=True)
assert stderr.parent==Path('/tmp/o4e-3261/ray/session_2026-10-06_03-56-08_188781_1526172/logs')
files={}
for name,p in [('fully_sharded_data_parallel.py',source),('rank0-original.err',stderr)]:
 c.physical(p);before=p.stat();fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
 with os.fdopen(fd,'rb') as f:
  op=os.fstat(f.fileno());assert (op.st_dev,op.st_ino,op.st_uid)==(before.st_dev,before.st_ino,before.st_uid);raw=f.read(before.st_size);end=os.fstat(f.fileno())
 after=p.stat();assert (end.st_dev,end.st_ino,end.st_uid)==(after.st_dev,after.st_ino,after.st_uid)==(before.st_dev,before.st_ino,before.st_uid)
 files[name]={'physical_path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'base64':base64.b64encode(raw).decode(),'before':{'device':before.st_dev,'inode':before.st_ino,'uid':before.st_uid,'bytes':before.st_size,'mtime_ns':before.st_mtime_ns,'ctime_ns':before.st_ctime_ns},'after':{'device':after.st_dev,'inode':after.st_ino,'uid':after.st_uid,'bytes':after.st_size,'mtime_ns':after.st_mtime_ns,'ctime_ns':after.st_ctime_ns},'same_opened_inode':True,'complete_prefix_at_start':len(raw)==before.st_size}
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'READ_ONLY_ADDITIONAL_ACTUAL_INSTALLED_TORCH_WARNING_SOURCE_NOT_NATIVE210_CHANGE','allocation':allocation,'native210_sha256':'3842c259b30c524e5ba5e067eb5b37f6dbac1042e162a5d93dd23511c13b5c9d','pinned_native210_torch_helper':rec,'derived_installed_torch_root':str(torch_root),'rank0_process':process,'rank0_stderr_actual_fd2':str(stderr),'files':files},sort_keys=True))
