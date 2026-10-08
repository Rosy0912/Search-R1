"""Transfer one task-owned code/proof file to an absent h100-8 destination."""
import hashlib,json,shlex,subprocess,sys
from pathlib import Path
sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control')
import control
local=Path(sys.argv[1]);target=sys.argv[2]
raw=local.read_bytes()
if len(raw)>16*1024*1024: raise ValueError('metadata/code transfer exceeds limit')
if not target.startswith('/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004/'): raise ValueError('wrong task root')
expected=hashlib.sha256(raw).hexdigest()
code='''import hashlib,json,os,sys,time
from pathlib import Path
p=Path(sys.argv[1]); expected=sys.argv[2];data=sys.stdin.buffer.read()
assert hashlib.sha256(data).hexdigest()==expected,'input SHA differs'
assert not os.path.lexists(p),'destination exists: preserve/inspect'
assert p.parent.is_dir() and p.parent.resolve()==p.parent,'parent must exist without symlink'
part=Path(str(p)+'.transfer-'+str(time.time_ns()))
with part.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
assert hashlib.sha256(part.read_bytes()).hexdigest()==expected,'written SHA differs'
os.link(part,p);part.unlink()
fd=os.open(p.parent,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
print(json.dumps(dict(path=str(p),bytes=len(data),sha256=expected,exclusive=True)))
'''
compile(code,'remote-transfer','exec')
inner='/usr/bin/python3 -c '+shlex.quote(code)+' '+shlex.quote(target)+' '+shlex.quote(expected)
mid='env -u LD_LIBRARY_PATH ssh -T -o ConnectTimeout=15 h100-8 '+shlex.quote(inner)
route=control.ssh('mgmt-host')+['env -u LD_LIBRARY_PATH ssh -T -o ConnectTimeout=15 h100-16 '+shlex.quote(mid)]
r=subprocess.run(route,input=raw,capture_output=True,timeout=120)
if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace'))
receipt=json.loads(r.stdout)
assert receipt['sha256']==expected and receipt['bytes']==len(raw)
print(json.dumps(receipt))
