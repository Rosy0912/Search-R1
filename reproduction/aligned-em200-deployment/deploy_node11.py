"""Build a small immutable install request; use existing shared source/assets/packages."""
import base64,json,hashlib,argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
d=Path(__file__).parent/'cohost-node11'
names=['formal.sbatch','formal_worker.sh','formal_inner.sh','formal_driver.py','formal_service_proof.py','bind_retrieval_proof.py','ray_cli_compat.py','bootstrap.py','formal_admission.py','submit_formal.py','resource_watch.py']
payload={n:{'sha256':hashlib.sha256((d/n).read_bytes()).hexdigest(),'base64':base64.b64encode((d/n).read_bytes()).decode()} for n in names}
receiver=r'''
import os,json,hashlib,base64,subprocess,datetime,re,sys,stat
from pathlib import Path
if sys.flags.optimize or os.environ.get('PYTHONOPTIMIZE'):raise RuntimeError('optimized receiver forbidden')
assert os.uname().nodename=='h100-11'
r=Path('/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-20261009');old=r.parent/'bapo-baseline-aligned-em200-20261009';source='080ebbc46b89d61567cd33e1a5e2f1b40afd8b84'
assert r.resolve()==r
for n in ['runs','logs','inputs','tools','tools/deployment','tools/deployment/cohost-node11','scheduler-head','cache','lib','lib/driver']:
 p=r/n
 if p.is_symlink():raise RuntimeError('symlink install directory')
 p.mkdir(parents=True,exist_ok=True)
 if p.resolve()!=p or not p.is_dir():raise RuntimeError('noncanonical install directory')
repo=r/'repo'
if not repo.exists():subprocess.run(['git','clone','--shared','--no-checkout',str(old/'repo'),str(repo)],check=True,capture_output=True,text=True,timeout=45);subprocess.run(['git','-C',str(repo),'checkout','--detach',source],check=True,capture_output=True,text=True,timeout=45)
assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==source
assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip()
def write(p,raw):
 if p.parent.resolve()!=p.parent or p.is_symlink():raise RuntimeError('noncanonical write path')
 if p.exists():
  if not stat.S_ISREG(p.stat().st_mode):raise RuntimeError('nonregular install file')
  assert p.read_bytes()==raw,str(p)
 else:
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
for n,x in PAYLOAD.items():
 raw=base64.b64decode(x['base64']);assert hashlib.sha256(raw).hexdigest()==x['sha256']
 write(r/'tools/resource_watch.py' if n=='resource_watch.py' else r/'tools/deployment/cohost-node11'/n,raw)
ld=subprocess.check_output(['/sbin/ldconfig','-p'],text=True)
paths=re.findall(r'libcuda\.so\.1\s+[^\n]*=>\s+(\S+)',ld);assert paths
for n in ['libcuda.so','libcuda.so.1']:
 p=r/'lib/driver'/n;t=Path(paths[0]).resolve()
 if p.is_symlink():assert p.resolve()==t
 else:assert not p.exists();p.symlink_to(t)
base=Path('/mnt/shared/workspace/training_data_node_5/shaoshitong/search-r1-rl')
model=base/'bapo-fresh200-safety-20261006/assets/qwen2.5-3b-instruct-aa8e725';common=base/'bapo-e1-instruct-300-stabilityfix-r2-20261005/common-data';assets=base/'bapo/assets-root/assets'
assert all(p.is_dir() for p in [model,common,assets])
inputs={'data':old/'inputs/baseline-aligned-data.json','order':old/'inputs/baseline-observed-order.json','retrieval':old/'runs/hardware-aligned-em200-13e07285c97044fd97974d94f53fea1b-3345/retrieval-golden.json'}
def relocate(v):
 if isinstance(v,str):
  if v.startswith(str(old)):return str(r)+v[len(str(old)):]
  if v.startswith('/dataset/'):return '/mnt/shared/workspace/training_data_node_5'+v[len('/dataset'):]
  return v
 if isinstance(v,list):return [relocate(x)for x in v]
 if isinstance(v,dict):return {k:relocate(x)for k,x in v.items()}
 return v
b=dict(source=source,root=str(r),host='h100-11',scope='SHARED_ASSET_PATH_BINDINGS_NOT_GPU_ACCEPTANCE',model=str(model),common=str(common),retrieval_assets=str(assets),tools={},parent_inputs={})
for name,p in inputs.items():
 raw=p.read_bytes();target=r/'inputs'/('baseline-aligned-'+name+'.json')
 value=json.loads(raw);new=json.dumps(relocate(value),sort_keys=True,indent=2).encode() if name=='retrieval' else raw
 write(target,new);b[name]={'path':str(target),'sha256':hashlib.sha256(new).hexdigest()};b['parent_inputs'][name]={'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'scope':'original accepted content; path relocation not node11 GPU acceptance'}
for n,x in PAYLOAD.items():b['tools']['tools/resource_watch.py' if n=='resource_watch.py' else n]=x['sha256']
write(r/'inputs/node11-bindings.json',json.dumps(b,sort_keys=True,indent=2).encode())
env=dict(os.environ,PYTHONPATH=str(r/'tools')+':'+str(r/'retriever-gpu-lib')+':'+str(repo),CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1');env.pop('LD_LIBRARY_PATH',None)
p=subprocess.run([str(r/'env/bin/python'),'-B',str(r/'tools/deployment/cohost-node11/formal_admission.py'),'verify'],env=env,capture_output=True,text=True,timeout=45)
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'root':str(r),'source':source,'installed_tools':b['tools'],'bindings_sha256':hashlib.sha256((r/'inputs/node11-bindings.json').read_bytes()).hexdigest(),'admission':{'exit':p.returncode,'stdout':p.stdout,'stderr':p.stderr},'scope':'ACTUAL_NODE11_SHARED_ASSET_CPU_ADMISSION_NOT_GPU_OR_TRAINING_PASS'}))
raise SystemExit(p.returncode)
'''
Path(a.output).write_text('PAYLOAD='+repr(payload)+'\n'+receiver)
