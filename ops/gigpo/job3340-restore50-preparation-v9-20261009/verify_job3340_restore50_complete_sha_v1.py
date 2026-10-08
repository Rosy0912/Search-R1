import concurrent.futures,hashlib,json,os,pathlib,sys,time
manifest=pathlib.Path(sys.argv[1]);destination=pathlib.Path(sys.argv[2])
expected="09b682b1661172760ca5d042372d0cca7aadb929e25214efeb2e484948ec8bc0"
raw=manifest.read_bytes();assert hashlib.sha256(raw).hexdigest()==expected
m=json.loads(raw)
def check(r):
 p=pathlib.Path(r["path"]);before=p.stat();assert not p.is_symlink() and before.st_size==r["bytes"]
 h=hashlib.sha256()
 with p.open("rb") as f:
  while True:
   b=f.read(8*1024*1024)
   if not b:break
   h.update(b)
 after=p.stat();assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
 return dict(path=str(p),bytes=after.st_size,sha256=h.hexdigest(),expected_sha256=r["sha256"],matched=h.hexdigest()==r["sha256"])
try:
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool: rows=list(pool.map(check,m["files"]))
 result=dict(status="PASS" if all(r["matched"] for r in rows) else "FAILED",manifest_sha256=expected,logical_step=m["logical_step"],actual_updates=m["actual_new_updates"],files=rows,all_complete_files_sha_verified=all(r["matched"] for r in rows),gpu_execution=False,semantic_reload_verified=False)
except BaseException as e:result=dict(status="FAILED",error=repr(e),semantic_reload_verified=False)
result.update(utc_epoch=time.time(),pid=os.getpid())
with destination.open("x") as f:json.dump(result,f,indent=2);f.flush();os.fsync(f.fileno())
