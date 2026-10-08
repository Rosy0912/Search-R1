from pathlib import Path
import json,hashlib,os
root=Path('/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008')
helper=root/'source/training-871dc46/opsd_ablation_v1'
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
def read(p,expected=None):
 s=p.stat();raw=p.read_bytes();t=p.stat();assert (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)==(t.st_dev,t.st_ino,t.st_size,t.st_mtime_ns,t.st_ctime_ns)
 h=hashlib.sha256(raw).hexdigest();assert expected is None or h==expected
 return json.loads(raw),dict(path=str(p),sha256=h,bytes=len(raw),device=s.st_dev,inode=s.st_ino,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns),raw.decode()
cfg,cf,_=read(root/'group-C1/receipts/authorized-launch-20261009/config.json','13fdbd51cc290bc6d23b0577c67bf8160a17292fab6f315fbfb07ea3f6ed3310')
assert cfg['run_id']=='a1a56efd32e5456d8c6813454975e38f' and cfg['variant']=='C1'
m,mp,_=read(helper/'source-manifest.json','8b22e616d470d79022cc6a0b07dcb0774df21b8e35ebd7dbcaa9b19150236a41')
files=[]
for name,ref in m['files'].items():
 p=helper/name;s=p.stat();raw=p.read_bytes();t=p.stat();assert (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)==(t.st_dev,t.st_ino,t.st_size,t.st_mtime_ns,t.st_ctime_ns)
 assert hashlib.sha256(raw).hexdigest()==ref['sha256'] and len(raw)==ref['bytes'];files.append(dict(path=str(p),sha256=ref['sha256'],bytes=len(raw)))
assert next(x for x in files if x['path'].endswith('/audit_state.py'))['sha256']=='7bfde1c08c421ba50fafa936c39a53305bc0966c06fdb8a58514708630bafe8a'
cps=[]
for n in [1,5]:
 v,vp,raw=read(root/f'group-C1/runs/checkpoints/update-{n:08d}-outer-00000/manifest.json')
 assert v['optimizer_updates']==n and v['outer_index']==0 and v['run_id']==cfg['run_id']
 cps.append(dict(expected_updates=n,expected_outer=0,proof=vp,raw_manifest=raw))
print(json.dumps(dict(accepted=True,config=cf,helper_manifest=mp,helper_files=files,checkpoints=cps)))
