import subprocess,pathlib,shlex
code='''import sys,json,hashlib,time,subprocess,shlex
sys.path.insert(0,'/home/lichen/h100-cleanup-20261012-bapo')
from uploader_parallel4x4 import ParallelRemoteFile,source_args
j=json.load(open('/home/lichen/h100-cleanup-20261012-bapo/plan.json'));rec=j['records'][4];rec=dict(rec,archive=rec['path'],archive_bytes=rec['size']);r=ParallelRemoteFile(rec);t=time.time();h=hashlib.sha256()
for n in range(2):h.update(r.read(8*1048576))
first=h.hexdigest();elapsed=time.time()-t;r.seek(3*1048576);cross=hashlib.sha256(r.read(2*1048576)).hexdigest();r.close()
src='import hashlib,json;f=open('+repr(rec['path'])+',"rb");a=hashlib.sha256(f.read(16*1048576)).hexdigest();f.seek(3*1048576);b=hashlib.sha256(f.read(2*1048576)).hexdigest();print(json.dumps([a,b]))'
p=subprocess.run(source_args('python3 -c '+shlex.quote(src)),capture_output=True,text=True,timeout=50);assert p.returncode==0
expected=json.loads(p.stdout);assert [first,cross]==expected
print(json.dumps(dict(status='NODE11_FOUR_LANE_SHA_AND_SEEK_PASS',bytes=16*1048576,elapsed=elapsed,MBps=16*1048576/elapsed/1e6,ram_cache=32*1048576)))
'''
p=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ProxyCommand=none','-o','ProxyJump=none','mgmt-host','python3 -c '+shlex.quote(code)],capture_output=True,text=True,timeout=120)
pathlib.Path('outputs/cleanup-h10011/reader-test.json').write_text(p.stdout)
print(p.returncode,p.stdout,p.stderr[-1000:])
