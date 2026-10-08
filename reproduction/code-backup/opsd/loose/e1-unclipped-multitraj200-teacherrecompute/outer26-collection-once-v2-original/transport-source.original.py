"""Own H100-node SSH transport. No new credentials or other-cluster routes."""
import subprocess,shlex,pathlib,json,hashlib
opts=['-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=3','-o','ControlMaster=no','-o','ControlPath=none']
def route(cmd):
 worker='env -u LD_LIBRARY_PATH ssh '+shlex.join(opts+['h100-10',cmd])
 head='env -u LD_LIBRARY_PATH ssh '+shlex.join(opts+['-o','HostName=10.33.0.16','-o','HostKeyAlias=100.108.173.4','-o','StrictHostKeyChecking=yes','h100-16',worker])
 return ['ssh']+opts+['-o','ProxyCommand=/usr/bin/nc -X 5 -x 127.0.0.1:7892 %h %p','mgmt-host',head]
def remote(cmd,data=None,timeout=60):return subprocess.run(route(cmd),input=data,capture_output=True,check=True,timeout=timeout)
def canonical(obj):return json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def publish(path,raw):
 p=pathlib.PurePosixPath(path); assert p.is_absolute()
 code=f'''import pathlib,os,secrets,sys,hashlib
p=pathlib.Path({str(p)!r});assert p.parent.resolve(strict=True)==p.parent and p.parent.stat().st_uid==os.getuid()==0
b=sys.stdin.buffer.read();assert hashlib.sha256(b).hexdigest()=={hashlib.sha256(raw).hexdigest()!r}
t=p.parent/('.'+p.name+'.'+secrets.token_hex(12));fd=os.open(str(t),os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
with os.fdopen(fd,'wb') as f:f.write(b);f.flush();os.fsync(f.fileno())
os.link(t,p,follow_symlinks=False);t.unlink();fd=os.open(str(p.parent),os.O_DIRECTORY|os.O_RDONLY)
try:os.fsync(fd)
finally:os.close(fd)
print(hashlib.sha256(p.read_bytes()).hexdigest())
'''
 return remote('python3 -c '+shlex.quote(code),raw)
