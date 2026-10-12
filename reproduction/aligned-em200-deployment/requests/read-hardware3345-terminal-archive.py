"""Transfer bounded own completed raw evidence; no asset files or remote mutation."""
import base64,hashlib,io,json,os,stat,tarfile
from pathlib import Path
ROOT=Path('/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-baseline-aligned-em200-20261009')
P=ROOT/'evidence/hardware3345-terminal-raw-20261009-observation1'
if os.uname().nodename!='h100-5' or P.resolve()!=P or not P.is_dir():raise RuntimeError('Canonical actual terminal archive required')
raw=(P/'file-manifest.json').read_bytes();m=json.loads(raw)
files=[];total=0
for p in sorted(P.rglob('*')):
 if p.is_symlink():raise RuntimeError('No symlink evidence')
 if p.is_dir():continue
 s=p.stat()
 if not stat.S_ISREG(s.st_mode) or s.st_nlink!=1 or s.st_size>32*1024*1024:raise RuntimeError('Bounded ordinary evidence only')
 total+=s.st_size
 if total>64*1024*1024:raise RuntimeError('Evidence exceeds bounded transfer')
 files.append((p,s))
buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w:gz') as t:
 for p,before in files:
  data=p.read_bytes();after=p.stat()
  if (before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):raise RuntimeError('Archive changed during transfer')
  i=tarfile.TarInfo(str(p.relative_to(P)));i.size=len(data);i.mode=0o600;t.addfile(i,io.BytesIO(data))
packed=buf.getvalue()
print(json.dumps({'scope':'RAW_ARCHIVE_TRANSFER_NOT_ACCEPTANCE','path':str(P),'manifest_sha256':hashlib.sha256(raw).hexdigest(),'tar_sha256':hashlib.sha256(packed).hexdigest(),'files':len(files),'bytes_uncompressed':total,'tar_gzip_base64':base64.b64encode(packed).decode()}))
