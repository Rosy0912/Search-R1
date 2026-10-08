import sys
sys.dont_write_bytecode=True
import pathlib,json
sys.path.insert(0,'work/g1-native-cpu-deploy-v6')
from stream_transport import send
source=r'''
import sys
sys.dont_write_bytecode=True
import pathlib,os,time,json,hashlib
R=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd/logs/r1-standard32b-profile2-v4/evidence')
def read(p):
 try:return p.read_text()
 except OSError as e:return {'error':repr(e)}
def tail(p,n):
 try:
  st=p.stat()
  with p.open('rb') as f:f.seek(max(0,st.st_size-n));b=f.read(n)
  return {'path':str(p),'size':st.st_size,'mtime_ns':st.st_mtime_ns,'tail_bytes':len(b),'tail_sha256':hashlib.sha256(b).hexdigest(),'tail':b.decode('utf-8','replace')}
 except OSError as e:return {'error':repr(e),'path':str(p)}
def snap():
 rows=[]
 for p in pathlib.Path('/proc').iterdir():
  if not p.name.isdigit():continue
  cg=read(p/'cgroup')
  if not isinstance(cg,str) or 'job_3163' not in cg:continue
  cmd=read(p/'cmdline')
  if not isinstance(cmd,str):continue
  cmd=cmd.replace('\x00',' ').strip()
  if not ('ray::WorkerDict' in cmd or 'supervise.py' in cmd or 'train_entry.py' in cmd):continue
  status=read(p/'status');fds=[]
  try:
   for f in (p/'fd').iterdir():
    try:
     t=os.readlink(f)
     if t.startswith('/dataset/shaoshitong/search-r1-rl/topd/assets-root'):
      fds.append({'fd':f.name,'target':t,'fdinfo':read(p/'fdinfo'/f.name)})
    except OSError:pass
  except OSError:pass
  rows.append({'pid':int(p.name),'command':cmd,'stat':read(p/'stat'),'wchan':read(p/'wchan'),'status':status,'io':read(p/'io'),'asset_fds':fds})
 sample=tail(R/'own-resource-samples.jsonl',512*1024)
 if 'tail' in sample:
  lines=sample.pop('tail').splitlines();last=None
  for line in reversed(lines):
   try:last=json.loads(line);break
   except json.JSONDecodeError:continue
  sample['last_complete_record']=last
 return {'epoch':time.time(),'processes':rows,'trainer':tail(R/'trainer.log',9000),'retriever':tail(R/'retriever.log',2500),'last_resource_sample':sample}
a=snap();time.sleep(30);b=snap();print(json.dumps({'scope':'READ_ONLY_3163_CURRENT_STAGE_NO_ATTACH_NO_SIGNAL_NO_GPU_QUERY','first':a,'second':b},sort_keys=True))
'''
r=send(source,timeout=120)
pathlib.Path('outputs/standard3163-stage-transport.log').write_bytes(r.stderr)
assert r.returncode==0,r.stderr.decode()
p=pathlib.Path('outputs/remote-evidence/standard3163-stage-readonly.json');p.write_bytes(r.stdout)
d=json.loads(r.stdout)
for label in ('first','second'):
 s=d[label];print(label,s['epoch'],'trainer',s['trainer'].get('size'),'retriever',s['retriever'].get('size'));print([(x['pid'],x['command'][:95],x['wchan']) for x in s['processes']]);print(s['trainer'].get('tail','')[-1800:])
print('saved',p)
