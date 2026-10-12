import subprocess,pathlib,shlex
code='''import sys,json,hashlib,time,subprocess,shlex,pathlib
sys.path.insert(0,'/home/lichen/h100-cleanup-20261012-bapo')
import uploader_parallel4x4_gated as mod
from huggingface_hub.lfs import SliceFileObj
mod.ROOT=pathlib.Path('/home/lichen/h100-cleanup-20261012-bapo/gated-reader-test');mod.ROOT.mkdir(exist_ok=True)
j=json.load(open('/home/lichen/h100-cleanup-20261012-bapo/plan.json'))
rec=j['records'][4];rec=dict(rec,archive=rec['path'],archive_bytes=rec['size']);r=mod.ParallelRemoteFile(rec);t=time.time();h=hashlib.sha256()
for n in range(3):
 with SliceFileObj(r,seek_from=n*8000000,read_limit=8000000) as sliced:
  while True:
   b=sliced.read(65536)
   if not b:break
   h.update(b)
opens=r.source_opens;assert opens==4,opens
first=h.hexdigest();elapsed=time.time()-t;r.close()
src='import hashlib;f=open('+repr(rec['path'])+',"rb");print(hashlib.sha256(f.read(24000000)).hexdigest())'
p=subprocess.run(mod.source_args('python3 -c '+shlex.quote(src)),capture_output=True,text=True,timeout=50);assert p.returncode==0 and p.stdout.strip()==first
small=j['records'][12];small=dict(small,archive=small['path'],archive_bytes=small['size']);r=mod.ParallelRemoteFile(small)
a=r.read(512);r.seek(0);b=r.read(512);r.close();assert a==b and hashlib.sha256(b).hexdigest()==small['sha256']
print(json.dumps(dict(status='HF_MULTIPART_SEEK_REUSE_AND_SMALL_REWIND_SHA_PASS',bytes=24000000,elapsed=elapsed,MBps=24000000/elapsed/1e6,source_opens=opens,small_bytes=len(b))))
'''
p=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ProxyCommand=none','-o','ProxyJump=none','mgmt-host','python3 -c '+shlex.quote(code)],capture_output=True,text=True,timeout=150)
pathlib.Path('outputs/cleanup-h10011/gated-reader-test.json').write_text(p.stdout)
print(p.returncode,p.stdout,p.stderr[-1000:])
