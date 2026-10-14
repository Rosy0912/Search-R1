import os,json,time,stat
from pathlib import Path
assert os.getuid()==21930
root=Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-20261014/preparation')
receipt=root/'shared-checkpoint200-group-access-20261014.json'
assert not receipt.exists(),'Inspect previous permission audit instead of repeating'
bases=[Path('/data/user/jxu957/experiments/topd-20261008/runtime/logs/topd-hpc3-713717/evidence/checkpoints')/x/'global_step_200' for x in ['algorithm','actor']]
objects=[]
for base in bases:
 assert base.is_dir() and not base.is_symlink()
 for top,dirs,files in os.walk(str(base),followlinks=False):
  for p in [Path(top)]+[Path(top)/n for n in files]:
   st=p.lstat();assert st.st_uid==21930 and st.st_gid==21919 and not stat.S_ISLNK(st.st_mode),str(p)
   assert stat.S_ISDIR(st.st_mode) or stat.S_ISREG(st.st_mode)
   objects.append((p,st))
 for p in base.rglob('*'):
  assert not p.is_symlink(),'Checkpoint aliases require separate inspection'
before=[dict(path=str(p),uid=st.st_uid,gid=st.st_gid,mode=oct(stat.S_IMODE(st.st_mode)),bytes=st.st_size,device=st.st_dev,inode=st.st_ino,mtime_ns=st.st_mtime_ns) for p,st in objects]
v=dict(epoch=time.time(),actual_owner_uid=os.getuid(),shared_group=21919,authorized_by='latest human explicitly permits mou176 files shared to jxu957; prior experiments770 instruction',scope=[str(x) for x in bases],before=before,after=[],content_mtime_size_inode_unchanged=True)
with receipt.open('x') as f:json.dump(dict(v,state='PERMISSION_CHANGE_INTENT'),f,indent=2);f.flush();os.fsync(f.fileno())
for p,prior in objects:
 desired=0o770 if stat.S_ISDIR(prior.st_mode) else 0o660
 os.chmod(str(p),desired)
 current=p.lstat()
 assert current.st_uid==prior.st_uid and current.st_gid==prior.st_gid and (current.st_dev,current.st_ino,current.st_size,current.st_mtime_ns)==(prior.st_dev,prior.st_ino,prior.st_size,prior.st_mtime_ns)
 v['after'].append(dict(path=str(p),mode=oct(stat.S_IMODE(current.st_mode))))
v.update(state='SHARED_GROUP_ACCESS_APPLIED',completed_epoch=time.time())
completion=receipt.with_name(receipt.stem+'-complete.json')
with completion.open('x') as f:json.dump(v,f,indent=2);f.flush();os.fsync(f.fileno())
os.chmod(str(receipt),0o660);os.chmod(str(completion),0o660)
print(json.dumps(dict(actual_uid=os.getuid(),objects=len(objects),directories=sum(stat.S_ISDIR(x.st_mode) for _,x in objects),checkpoint_bytes=sum(x.st_size for _,x in objects if stat.S_ISREG(x.st_mode)),audit=str(completion),content_unchanged=True,shared_group=21919)))