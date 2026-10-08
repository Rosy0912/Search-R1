"""Create only this fresh200 root; dependencies/assets remain read-only."""
from pathlib import Path
import json,hashlib,os,subprocess,sys,shutil
ROOT=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-safety-20261006')
OLD=Path('/dataset/shaoshitong/search-r1-rl/bapo')
COMMON=Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-stabilityfix-r2-20261005')
def require(ok,msg):
 if not ok:raise RuntimeError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
require(sys.platform=='linux' and os.uname().nodename=='h100-5','wrong deployment host')
require(ROOT.is_dir() and not ROOT.is_symlink() and not (ROOT/'repo').exists(),'unknown/repeated source deployment forbidden')
source=(ROOT/'source-expected.txt').read_text().strip()
for name in ('logs','cache','runs','assets-root','scheduler-head'): (ROOT/name).mkdir(exist_ok=True)
require(not (ROOT/'env').exists(),'new environment already exists')
env=ROOT/'env';(env/'bin').mkdir(parents=True);(env/'lib/python3.10').mkdir(parents=True)
shutil.copy2(OLD/'env/pyvenv.cfg',env/'pyvenv.cfg')
for name in ('python','python3','python3.10'): (env/'bin'/name).symlink_to((OLD/'env/bin/python').resolve())
(env/'lib/python3.10/site-packages').symlink_to(OLD/'env/lib/python3.10/site-packages',target_is_directory=True)
for name in ('retriever-gpu-lib','lib'): (ROOT/name).symlink_to(OLD/name,target_is_directory=True)
(ROOT/'assets-root/assets').symlink_to(OLD/'assets-root/assets',target_is_directory=True)
shutil.copy2(OLD/'assets-root/assets.json',ROOT/'assets-root/assets.json')
subprocess.run(['git','clone','--branch','codex/bapo-fresh200-safety',str(ROOT/'source.bundle'),str(ROOT/'repo')],check=True)
require(subprocess.check_output(['git','-C',str(ROOT/'repo'),'rev-parse','HEAD'],text=True).strip()==source,'source mismatch')
require(not subprocess.check_output(['git','-C',str(ROOT/'repo'),'status','--porcelain'],text=True).strip(),'source dirty')
record=dict(source_commit=source,root=str(ROOT),source_bundle_sha256=sha(ROOT/'source.bundle'),
 own_env_prefix=str(env),shared_dependency_site_packages=str((env/'lib/python3.10/site-packages').resolve()),
 shared_read_only_retrieval_assets=str((ROOT/'assets-root/assets').resolve()),
 shared_model=str(COMMON/'assets/qwen2.5-3b-instruct-aa8e725'),shared_data=str(COMMON/'common-data'),
 old_source_unchanged=subprocess.check_output(['git','-C',str(OLD/'repo'),'rev-parse','HEAD'],text=True).strip(),
 scope='SOURCE_ENV_DEPLOYMENT_ONLY_NO_GPU_NO_LEGACY_RECEIPT')
with (ROOT/'logs/isolation-setup.json').open('x')as f:json.dump(record,f,indent=2)
print(json.dumps(record))
