import os,signal,pathlib,json,tarfile,hashlib,time
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-rewardfix-b097');run=root/'runs/bapo-hardware-3064'
partial=root/'logs/bapo-rewardfix-hardware3064-raw.tar.gz'
handled=[]
for proc in pathlib.Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  raw=(proc/'cmdline').read_bytes().split(b'\0')
  if raw[:2]!=[str(root/'env/bin/python').encode(),b'-']:continue
  ticks=(proc/'stat').read_text().split(') ',1)[1].split()[19]
  matched=[]
  for fd in (proc/'fd').iterdir():
   if pathlib.Path(os.readlink(fd))!=partial:continue
   flags=[x for x in (proc/'fdinfo'/fd.name).read_text().splitlines() if x.startswith('flags:')][0].split()[1]
   if int(flags,8)&os.O_ACCMODE in [os.O_WRONLY,os.O_RDWR]:matched.append(fd.name)
  if not matched:continue
  pid=int(proc.name);handle=os.pidfd_open(pid)
  assert (proc/'stat').read_text().split(') ',1)[1].split()[19]==ticks
  signal.pidfd_send_signal(handle,signal.SIGTERM)
  os.close(handle);handled.append(dict(pid=pid,start_ticks=ticks,exact_open_output=str(partial),fd=matched))
 except (FileNotFoundError,ProcessLookupError,PermissionError):continue
assert len(handled)<=1
(root/'logs/archive-partial-own-process-handling.json').write_text(json.dumps(dict(handled=handled,partial_preserved=str(partial),reason='exclude regenerable corpus Arrow cache'),indent=2)+'\n')
archive=root/'logs/bapo-rewardfix-hardware3064-raw-measurements.tar.gz'
assert not archive.exists()
with tarfile.open(archive,'w:gz') as t:
 for path in sorted(run.iterdir()):
  if path.is_file():
   assert not path.is_symlink()
   t.add(path,arcname='bapo-hardware-3064/'+path.name,recursive=False)
record=dict(path=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),scope='NEW_B097_HARDWARE_RAW_MEASUREMENT_RESOURCE_IDENTITY_FILES',excluded_regenerable_directories=['cache','runtime','tmp'],own_archive_handling=handled)
(root/'logs/hardware3064-archive-manifest.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
