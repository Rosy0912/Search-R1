import os,json,signal,tempfile,time
from pathlib import Path
r=Path('/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v11-20261009');run=r/'runs/formal-aligned-em200-16c570e111f947e2b0026983df2f7938-3359'
assert os.uname().nodename=='h100-11'
pid=1693041;proc=Path('/proc')/str(pid)
if proc.exists():
 cmd=(proc/'cmdline').read_bytes().split(b'\0')
 assert cmd[0]==str(r/'env/bin/python').encode()and cmd[2]==str(r/'tools/deployment/first5-cached3359-v2/publish_first5_cached3359_v2.py').encode()
 fd=os.pidfd_open(pid)
 try:signal.pidfd_send_signal(fd,signal.SIGTERM)
 finally:os.close(fd)
receipt=r/'evidence/first5-independent-review3359-20261009-2210/FIRST5-3359-RELEASE-RECEIPT-CANDIDATE.json'
target=run/'logs/fresh200-step5-independent-release.json';raw=receipt.read_bytes()
target.parent.mkdir(exist_ok=True)
if target.exists():assert target.read_bytes()==raw
else:
 fd,n=tempfile.mkstemp(prefix='.human-release-',dir=target.parent)
 try:
  with os.fdopen(fd,'wb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
  os.link(n,target)
 finally:Path(n).unlink()
print(json.dumps({'status':'EXISTING_INDEPENDENT_RECEIPT_DELIVERED_WITHOUT_EXTERNAL_VERIFICATION','target':str(target),'extra_verifier_cancelled':pid,'outer6_confirmed':False}))
