import subprocess,sys,shlex,hashlib,time
from pathlib import Path
p=Path(sys.argv[1]);target=sys.argv[2];partial=target+'.transfer'
h=hashlib.sha256()
with p.open('rb') as f:
 for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
sha=h.hexdigest()
def route(c):
 worker='env -u LD_LIBRARY_PATH ssh -T -S /dev/shm/topd-worker-connect-v2.sock -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=10 -o ServerAliveCountMax=3 h100-11 '+shlex.quote(c)
 head='env -u LD_LIBRARY_PATH ssh -T -S none -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=10 -o ServerAliveCountMax=3 h100-16 '+shlex.quote(worker)
 return ['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=3','-S','none','-o','ProxyCommand=nc -X 5 -x 127.0.0.1:7892 %h %p','mgmt-host',head]
for attempt in range(4):
 probe='for f in '+shlex.quote(target)+' '+shlex.quote(partial)+'; do test ! -f "$f" || sha256sum "$f"; done'
 r=subprocess.run(route(probe),capture_output=True,text=True,timeout=180)
 if r.returncode:print('TRANSFER_ROUTE_RETRY',attempt,flush=True);time.sleep(3);continue
 matches=[line.split(maxsplit=1)[1] for line in r.stdout.splitlines() if line.startswith(sha+' ')]
 if target in matches:print('TRANSFER_VERIFIED_EXISTING',p.name,sha,flush=True);break
 if partial not in matches:
  cmd='mkdir -p '+shlex.quote(str(Path(target).parent))+' && cat > '+shlex.quote(partial)+' && sha256sum '+shlex.quote(partial)
  with p.open('rb') as f:r=subprocess.run(route(cmd),stdin=f,capture_output=True,text=True)
  if r.returncode or not r.stdout.startswith(sha+' '):print('TRANSFER_RETRY_PARTIAL_RETAINED',attempt,flush=True);time.sleep(3);continue
 r=subprocess.run(route('mv '+shlex.quote(partial)+' '+shlex.quote(target)+' && sha256sum '+shlex.quote(target)),capture_output=True,text=True,timeout=180)
 if r.returncode==0 and r.stdout.startswith(sha+' '):print('TRANSFER_VERIFIED',p.name,sha,flush=True);break
 print('TRANSFER_RETRY_VERIFIED_PARTIAL_RETAINED',attempt,flush=True);time.sleep(3)
else:raise RuntimeError('Transfer failed; local and remote partial retained')
