"""Read-only process/FD proof for this exact destination; can enforce a writer-free handoff."""
import json,os,time,subprocess
from pathlib import Path
BASE=Path('/home/shaoshitong/search-e1-archive100-eval-20261008')
prefix=str(BASE/'model.partial')+'/'
d={'checked_epoch':time.time(),'scope':str(BASE/'model.partial'),'tar_processes':[],'write_fds':[],'unreadable_owned_processes':[]}
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  if p.stat().st_uid!=os.getuid():continue
  args=(p/'cmdline').read_bytes().split(b'\0');cmd=b' '.join(args).decode(errors='replace').strip()
  if args and os.path.basename(args[0].decode(errors='replace'))=='tar' and str(BASE/'model.partial') in cmd:d['tar_processes'].append({'pid':int(p.name),'command':cmd})
  for fd in (p/'fd').iterdir():
   try:
    target=os.readlink(fd)
    if not target.startswith(prefix):continue
    info=(p/'fdinfo'/fd.name).read_text();flags=int(next(x.split()[1] for x in info.splitlines() if x.startswith('flags:')),8)
    if flags&3 in [1,2]:d['write_fds'].append({'pid':int(p.name),'fd':fd.name,'target':target,'flags_octal':oct(flags),'command':cmd})
   except (FileNotFoundError,ProcessLookupError):pass
 except (FileNotFoundError,ProcessLookupError):pass
 except PermissionError:
  z=subprocess.run(['ps','-p',p.name,'-o','comm=,args='],capture_output=True,text=True).stdout.strip()
  if z:d['unreadable_owned_processes'].append({'pid':int(p.name),'process':z,'task_related':str(BASE) in z or z.split()[0]=='tar'})
print(json.dumps(d))
