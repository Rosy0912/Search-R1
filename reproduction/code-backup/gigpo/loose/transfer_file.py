import subprocess,sys,shlex,hashlib,time
from pathlib import Path
sys.path.insert(0,"/Users/shaoshitong/Documents/remote_codex_control")
import control
p=Path(sys.argv[1]);target=sys.argv[2]
h=hashlib.sha256()
with p.open('rb') as f:
 for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
sha=h.hexdigest()
partial=target+'.transfer.'+str(time.time_ns())
check=shlex.quote(sha+'  '+partial)
fmt=shlex.quote('%s'+chr(10))
cmd='mkdir -p '+shlex.quote(str(Path(target).parent))+' && cat > '+shlex.quote(partial)+' && printf '+fmt+' '+check+' | sha256sum -c - >/dev/null && sha256sum '+shlex.quote(partial)+' && mv '+shlex.quote(partial)+' '+shlex.quote(target)

def route(c):return control.ssh('mgmt-host')+['env -u LD_LIBRARY_PATH ssh -T -o ConnectTimeout=15 h100-16 '+shlex.quote('env -u LD_LIBRARY_PATH ssh -T -o ConnectTimeout=15 h100-8 '+shlex.quote(c))]
with p.open('rb') as f:r=subprocess.run(route(cmd),stdin=f,capture_output=True,text=True,timeout=120)
assert r.returncode==0 and r.stdout.split()[0]==sha,(r.returncode,r.stderr)

print('TRANSFER_VERIFIED',p.name,sha,flush=True)
