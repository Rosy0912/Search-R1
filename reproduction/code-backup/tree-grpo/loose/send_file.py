import hashlib, pathlib, shlex, subprocess, sys, time
from remote import route
p=pathlib.Path(sys.argv[1]); dest=sys.argv[2]
h=hashlib.sha256()
with p.open('rb') as stream:
 for b in iter(lambda:stream.read(8*1024**2),b''):h.update(b)
sha=h.hexdigest(); temp=dest+'.incoming.'+str(time.time_ns())
verify='import pathlib,hashlib; p=pathlib.Path('+repr(temp)+'); h=hashlib.sha256(); f=p.open("rb"); [h.update(b) for b in iter(lambda:f.read(8*1024**2),b"")]; assert h.hexdigest()=='+repr(sha)+'; p.replace('+repr(dest)+'); print("TRANSFER_SHA256_OK",h.hexdigest())'
cmd='set -e; mkdir -p '+shlex.quote(str(pathlib.PurePosixPath(dest).parent))+'; cat > '+shlex.quote(temp)+'; /usr/bin/python3.10 -c '+shlex.quote(verify)
with p.open('rb') as stream:r=subprocess.run(route(cmd),stdin=stream)
sys.exit(r.returncode)
