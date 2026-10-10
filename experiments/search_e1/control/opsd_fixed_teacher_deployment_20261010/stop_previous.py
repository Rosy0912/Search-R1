import pathlib,json,os,signal,time
b=pathlib.Path('/data/shaoshitong/search-e1-opsd-j-minus-p-bf16-continuation200-20261010');run='d040ebac9fa24ec28d43976299fd4f71';owner=3661504
s=json.loads((b/'runs/status.json').read_text());assert s['owner']['pid']==owner and s['owner']['starttick']==940523127
selected=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  status=(p/'status').read_text();uid=int(next(x.split()[1] for x in status.splitlines() if x.startswith('Uid:')))
  env=(p/'environ').read_bytes().split(b'\0');st=(p/'stat').read_text().split(') ',1)[1].split()
  if uid==1005 and ('E1_RUN_ID='+run).encode() in env:selected.append(dict(pid=int(p.name),tick=int(st[19]),uid=uid,argv=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()))
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
assert any(x['pid']==owner for x in selected) and all(x['pid']!=2239529 for x in selected)
dest=b/'preserved'/('replaced-by-fixed-teacher-'+str(int(time.time())));dest.mkdir();(dest/'before.json').write_text(json.dumps({'status':s,'owned':selected},indent=2))
for x in sorted(selected,key=lambda x:x['pid']!=owner):
 try:
  fd=os.pidfd_open(x['pid']);st=pathlib.Path('/proc',str(x['pid']),'stat').read_text().split(') ',1)[1].split();assert int(st[19])==x['tick'];signal.pidfd_send_signal(fd,signal.SIGTERM);os.close(fd)
 except ProcessLookupError:pass
print(json.dumps({'preserved':str(dest),'signaled':selected}))
