import sys
sys.dont_write_bytecode=True
import pathlib,json
sys.path.insert(0,'work/g1-native-cpu-deploy-v6')
from stream_transport import send
source=r'''
import sys
sys.dont_write_bytecode=True
import os,pathlib,json,time
P=pathlib.Path('/proc')
def read(p):
 try:return p.read_text()
 except OSError as e:return {'error':repr(e)}
def snap():
 rows=[]
 for d in P.iterdir():
  if not d.name.isdigit():continue
  cg=read(d/'cgroup')
  if not isinstance(cg,str) or 'job_3159' not in cg:continue
  cmd=read(d/'cmdline')
  if isinstance(cmd,str):cmd=cmd.replace('\x00',' ')
  fds=[]
  try:
   for f in (d/'fd').iterdir():
    try:
     target=os.readlink(f)
     if target.startswith('/dataset/shaoshitong/search-r1-rl/topd') or 'index' in target or 'wiki' in target:
      fds.append({'fd':f.name,'target':target,'fdinfo':read(d/'fdinfo'/f.name)})
    except OSError:pass
  except OSError:pass
  rows.append({'pid':int(d.name),'command':cmd,'cgroup':cg,'stat':read(d/'stat'),'status':read(d/'status'),'wchan':read(d/'wchan'),'io':read(d/'io'),'disk_fds':fds})
 return {'epoch':time.time(),'processes':rows,'diskstats':read(P/'diskstats'),'pressure_io':read(P/'pressure/io')}
a=snap();time.sleep(30);b=snap()
print(json.dumps({'scope':'READ_ONLY_JOB3159_ASSET_VERIFICATION_PROGRESS_NO_ATTACH_NO_SIGNAL','first':a,'second':b},sort_keys=True))
'''
r=send(source,timeout=120)
pathlib.Path('outputs/standard3159-asset-progress-transport.log').write_bytes(r.stderr)
assert r.returncode==0,r.stderr.decode()
p=pathlib.Path('outputs/remote-evidence/standard3159-asset-progress.json');p.write_bytes(r.stdout)
d=json.loads(r.stdout)
for label in ['first','second']:
 print(label,d[label]['epoch'])
 for r in d[label]['processes']:
  print(json.dumps({k:r[k] for k in ('pid','command','wchan','io','disk_fds')},ensure_ascii=False))
print('saved',p)
