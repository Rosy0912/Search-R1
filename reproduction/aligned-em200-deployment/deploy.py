"""Install only this fixed BAPO source; no Slurm, GPU, training or old state."""
from pathlib import Path
import argparse,base64,datetime,hashlib,json,os,shutil,subprocess,sys
ROOT=Path('/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-baseline-aligned-em200-20261009')
SOURCE='080ebbc46b89d61567cd33e1a5e2f1b40afd8b84'
OLD=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-controller-initfix-20261006')
DEP=Path('/dataset/shaoshitong/search-r1-rl/bapo')
TASK='BAPO_BASELINE_ALIGNED_EM200_20261009'
def require(ok,msg):
 if not ok:raise RuntimeError(msg)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def git(path,*args):return subprocess.check_output(['git','-C',str(path),*args],text=True).strip()
def safe_dir(path):
 require(not path.is_symlink(),'directory symlink forbidden: '+str(path))
 path.mkdir(exist_ok=True)
 require(path.is_dir() and path.resolve()==path,'directory identity changed: '+str(path))

def verify_topology():
 for name in ['repo','env','env/bin','env/lib','env/lib/python3.10','assets','assets-root','runs','cache','tools','scheduler-head','logs','incoming']:
  path=ROOT/name;require(path.is_dir() and not path.is_symlink() and path.resolve()==path,'own directory topology changed: '+name)
 expected={'env/lib/python3.10/site-packages':DEP/'env/lib/python3.10/site-packages', 'retriever-gpu-lib':DEP/'retriever-gpu-lib','lib':DEP/'lib','assets-root/assets':DEP/'assets-root/assets','assets/qwen2.5-3b-instruct-aa8e725':Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-safety-20261006/assets/qwen2.5-3b-instruct-aa8e725')}
 for name,target in expected.items():
  path=ROOT/name;require(path.is_symlink() and path.resolve(strict=True)==target.resolve(strict=True),'borrowed link changed: '+name)
 for name in ['python','python3','python3.10']:
  path=ROOT/'env/bin'/name;require(path.is_symlink() and path.resolve(strict=True)==(DEP/'env/bin/python').resolve(strict=True),'interpreter link changed')
 require(not (ROOT/'env/pyvenv.cfg').is_symlink() and sha(ROOT/'env/pyvenv.cfg')==sha(DEP/'env/pyvenv.cfg'),'venv configuration changed')
 require(not (ROOT/'assets-root/assets.json').is_symlink() and sha(ROOT/'assets-root/assets.json')==sha(DEP/'assets-root/assets.json'),'asset configuration changed')

def prepare():
 require(sys.platform=='linux' and os.uname().nodename=='h100-5','wrong target host')
 mount=json.loads(subprocess.check_output(['findmnt','-T','/mnt/shared','-J'],text=True))['filesystems'][0]
 require(mount['fstype']=='nfs4' and mount['source']=='10.32.0.8:/mnt/data/sihe-share5','unverified shared output storage')
 require(shutil.disk_usage('/mnt/shared').free>400*1024**3,'insufficient shared storage')
 require(not ROOT.is_symlink() and ROOT.resolve()==ROOT,'root/ancestor symlink forbidden')
 marker=ROOT/'.aligned-em200-deployment.json';identity={'task':TASK,'source':SOURCE,'root':str(ROOT)}
 if ROOT.exists():
  require(not marker.is_symlink() and marker.is_file() and json.loads(marker.read_text())==identity,'unknown existing target directory')
 else:
  ROOT.mkdir(parents=True,exist_ok=False)
  with marker.open('x')as f:json.dump(identity,f,sort_keys=True)
 require(ROOT.resolve()==ROOT,'canonical shared root changed')
 safe_dir(ROOT/'incoming');safe_dir(ROOT/'logs')
 return mount

def upload(number,encoded,digest):
 prepare();require(type(number)is int and 0<=number<16,'invalid source part')
 raw=base64.b64decode(encoded,validate=True);require(hashlib.sha256(raw).hexdigest()==digest,'source part hash mismatch')
 target=ROOT/'incoming'/f'source.part{number:02d}'
 require(not target.is_symlink(),'source part symlink forbidden')
 if target.exists():require(sha(target)==digest,'existing source part differs')
 else:
  with target.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
 return {'part':number,'sha256':digest,'bytes':len(raw)}

def setup(count,bundle_sha):
 mount=prepare();require(type(count)is int and 0<count<=16,'bad source part count')
 parts=[ROOT/'incoming'/f'source.part{i:02d}'for i in range(count)]
 for part in parts:require(part.is_file() and not part.is_symlink(),'source assembly part is missing/nonregular/symlink')
 bundle=ROOT/'incoming'/'source.bundle';raw=b''.join(part.read_bytes()for part in parts)
 require(hashlib.sha256(raw).hexdigest()==bundle_sha,'whole source bundle mismatch')
 require(not bundle.is_symlink(),'source bundle symlink forbidden')
 if bundle.exists():require(sha(bundle)==bundle_sha,'existing source bundle differs')
 else:
  with bundle.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
 record=ROOT/'logs'/'source-isolation.json'
 require(not record.is_symlink(),'deployment record symlink forbidden')
 if record.exists():
  prior=json.loads(record.read_text());require(prior.get('task')==TASK and prior.get('root')==str(ROOT) and prior.get('scope')=='SOURCE_DEPLOYMENT_ONLY_NO_GPU_NO_TRAINING_NO_OLD_RECEIPT','prior deployment identity differs');verify_topology();require(prior['source']==SOURCE and prior['bundle_sha256']==bundle_sha,'prior deployment mismatch')
  require(git(ROOT/'repo','rev-parse','HEAD')==SOURCE and not git(ROOT/'repo','status','--porcelain'),'deployed source changed')
  return prior
 old_before=git(OLD/'repo','rev-parse','HEAD');require(old_before=='60f566b3ddabc88ce9de56ead5ad30a3d6795dd2','source dependency identity changed')
 require(not (ROOT/'repo').exists(),'partial source setup requires inspection, not overwrite')
 subprocess.run(['git','clone','--no-hardlinks','--no-checkout',str(OLD/'repo'),str(ROOT/'repo')],check=True)
 subprocess.run(['git','-C',str(ROOT/'repo'),'fetch',str(bundle),'HEAD'],check=True)
 subprocess.run(['git','-C',str(ROOT/'repo'),'checkout','--detach',SOURCE],check=True)
 require(git(ROOT/'repo','rev-parse','HEAD')==SOURCE and not git(ROOT/'repo','status','--porcelain'),'fixed source not clean')
 require(not (ROOT/'env').exists(),'unknown environment prefix')
 env=ROOT/'env';(env/'bin').mkdir(parents=True);(env/'lib/python3.10').mkdir(parents=True)
 shutil.copy2(DEP/'env/pyvenv.cfg',env/'pyvenv.cfg')
 for name in ['python','python3','python3.10']:(env/'bin'/name).symlink_to((DEP/'env/bin/python').resolve())
 (env/'lib/python3.10/site-packages').symlink_to(DEP/'env/lib/python3.10/site-packages',target_is_directory=True)
 for name in ['retriever-gpu-lib','lib']:(ROOT/name).symlink_to(DEP/name,target_is_directory=True)
 (ROOT/'assets-root').mkdir();(ROOT/'assets-root/assets').symlink_to(DEP/'assets-root/assets',target_is_directory=True)
 shutil.copy2(DEP/'assets-root/assets.json',ROOT/'assets-root/assets.json')
 model=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-safety-20261006/assets/qwen2.5-3b-instruct-aa8e725')
 require(model.is_dir() and not model.is_symlink(),'official model mirror unavailable')
 (ROOT/'assets').mkdir();(ROOT/'assets/qwen2.5-3b-instruct-aa8e725').symlink_to(model,target_is_directory=True)
 for name in ['runs','cache','tools','scheduler-head']:(ROOT/name).mkdir()
 verify_topology()
 require(git(OLD/'repo','rev-parse','HEAD')==old_before,'old source unexpectedly changed')
 value={'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'task':TASK,'root':str(ROOT),'source':SOURCE,'bundle_sha256':bundle_sha,'mount':mount,'own_env_prefix':str(env),'shared_readonly_dependencies':str(DEP/'env/lib/python3.10/site-packages'),'official_model':str(model),'old_source_unchanged':old_before,'scope':'SOURCE_DEPLOYMENT_ONLY_NO_GPU_NO_TRAINING_NO_OLD_RECEIPT'}
 with record.open('x')as f:json.dump(value,f,indent=2)
 return value
def install_tools(files):
 prepare();record=ROOT/'logs/source-isolation.json'
 require(record.is_file() and not record.is_symlink(),'source setup not complete or record redirected')
 value=json.loads(record.read_text())
 require(value.get('task')==TASK and value.get('root')==str(ROOT) and value.get('source')==SOURCE and value.get('scope')=='SOURCE_DEPLOYMENT_ONLY_NO_GPU_NO_TRAINING_NO_OLD_RECEIPT','source setup identity differs')
 verify_topology()
 require(git(ROOT/'repo','rev-parse','HEAD')==SOURCE and not git(ROOT/'repo','status','--porcelain'),'fixed source changed before tool installation')
 target=ROOT/'tools/deployment';safe_dir(target)
 result=[]
 for name,encoded,digest in files:
  rel=Path(name);require(not rel.is_absolute() and all(part not in ['.','..']for part in rel.parts) and len(rel.parts)<=3,'invalid tool relative path')
  path=target/rel
  current=target
  for part in rel.parts[:-1]:current=current/part;safe_dir(current)
  require(not path.is_symlink() and path.resolve()==path,'tool path redirected')
  raw=base64.b64decode(encoded,validate=True);require(hashlib.sha256(raw).hexdigest()==digest,'tool bytes differ')
  if path.exists():require(path.is_file() and sha(path)==digest,'existing bound tool differs')
  else:
   with path.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
  result.append({'path':str(path),'sha256':digest,'bytes':len(raw)})
 return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--parts',type=int,required=True);p.add_argument('--bundle-sha',required=True);a=p.parse_args();print(json.dumps(setup(a.parts,a.bundle_sha)))
