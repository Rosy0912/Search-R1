import pathlib,os,json,time
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v11-20261009')
paths=[root/x for x in ['cleanup-lfs80m-diagnostic.bin','cleanup-lfs256m-diagnostic.bin','cleanup-lfs256k256m-diagnostic.bin']]
refs=[]
for p in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  for f in (p/'fd').glob('*'):
   try:dest=pathlib.Path(os.readlink(f))
   except OSError:continue
   if dest in paths:refs.append(dict(pid=p.name,path=str(dest)))
 except (PermissionError,FileNotFoundError):pass
assert not refs,json.dumps(refs)
deleted=[]
for p in paths:
 if p.exists():deleted.append(dict(path=str(p),bytes=p.stat().st_size));p.unlink()
print(json.dumps(dict(epoch=time.time(),deleted=deleted)))
