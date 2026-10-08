from pathlib import Path
import os,json,time
root=Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-countfix-20261004');logs=root/'logs';rows=[]
for p in sorted(logs.glob('hardware-3156-raw-capture-*')):
 s=p.stat();rows.append(dict(path=str(p),bytes=s.st_size,mtime=s.st_mtime))
processes=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  opened=[]
  for fd in (p/'fd').iterdir():
   try:
    target=os.readlink(fd)
    if target.startswith(str(logs)+'/hardware-3156-raw-capture-'):opened.append(dict(fd=fd.name,target=target))
   except OSError:pass
  if opened:
   stat=(p/'stat').read_text().rsplit(')',1)[1].split();processes.append(dict(pid=int(p.name),state=stat[0],start_ticks=int(stat[19]),owned_archive_fds=opened))
 except (OSError,ValueError):pass
print(json.dumps(dict(scope='READ_ONLY_OWNED_ARCHIVE_FILES_AND_OPEN_FDS_NO_SIGNALS',time=time.time(),files=rows,processes=processes),indent=2))
