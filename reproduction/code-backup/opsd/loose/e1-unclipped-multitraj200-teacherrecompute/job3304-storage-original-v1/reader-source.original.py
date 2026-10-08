import pathlib,os,json,datetime,subprocess
r=pathlib.Path('/mnt/data/shaoshitong/search-r1-rl/opsd-e1-unclipped-multitraj200-20261006/runs/formal-3304');assert r.resolve(strict=True)==r and r.stat().st_uid==0
v=os.statvfs(r);q=subprocess.run(['df','-PT',str(r)],capture_output=True,text=True,timeout=10)
print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),run_root=str(r),filesystem_available_bytes=v.f_bavail*v.f_frsize,filesystem_total_bytes=v.f_blocks*v.f_frsize,df_returncode=q.returncode,df_stdout=q.stdout,df_stderr=q.stderr,meminfo=pathlib.Path('/proc/meminfo').read_text(),scope='Read-only filesystem and host memory snapshot; no future capacity guarantee'),sort_keys=True))
