"""Finish only the validated failed source clone; retain original archive/setup/env.
No jobs, GPU actions, environment recreation, old receipts or shared writes.
"""
import argparse,ast,hashlib,io,json,os,subprocess,sys,tarfile
from pathlib import Path
ROOT=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-controller-initfix-20261006')
OLD=Path('/dataset/shaoshitong/search-r1-rl/bapo')
COMMON=Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-stabilityfix-r2-20261005')
MODEL=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-safety-20261006/assets/qwen2.5-3b-instruct-aa8e725')
SOURCE='60f566b3ddabc88ce9de56ead5ad30a3d6795dd2'
BRANCH='codex/bapo-fresh200-controller-initfix'
BUNDLE_SHA='08affedc9988b34e3d679b7e7fa88498af1feff44bad445b110a24be05649503'
TAR_SHA='b05f8ef29736c428eec73303ca246ca5cf8fd0af082cce065f91772c0ede75a1'
SETUP_SHA='efefa0b2e6f41696ec2d7dc58656bf9cfa077b580913aa5277ad2f9cb22c1395'
FAILED_BRANCH='codex/bapo-fresh200-controller-initfix-policy'
def require(ok,message):
 if not ok:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def bundle_heads(bundle):return subprocess.check_output(['git','bundle','list-heads',str(bundle)],text=True).strip().splitlines()
def validate_bundle(bundle):
 require(sha(bundle)==BUNDLE_SHA,'Exact preserved bundle required')
 require(bundle_heads(bundle)==[SOURCE+' refs/heads/'+BRANCH],'Exact real frozen branch/ref required')
def setup_branch(raw):
 tree=ast.parse(raw)
 calls=[n for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='run'and n.args and isinstance(n.args[0],ast.List)]
 commands=[]
 for call in calls:
  entries=call.args[0].elts
  if len(entries)>=4 and all(isinstance(n,ast.Constant)for n in entries[:4]):
   prefix=[n.value for n in entries[:4]]
   if prefix[:3]==['git','clone','--branch']:commands.append(prefix[3])
 require(len(commands)==1,'One explicit source clone branch required');return commands[0]
def validate_archive(root):
 raw=(root/'source-deployment.tar').read_bytes();require(hashlib.sha256(raw).hexdigest()==TAR_SHA,'Preserved original deployment archive required')
 with tarfile.open(fileobj=io.BytesIO(raw),mode='r:')as tar:
  members=tar.getmembers();require(len(members)==4 and {m.name for m in members}=={'source.bundle','source-expected.txt','setup-isolated.py','source-deployment-manifest.json'}and all(m.isfile()for m in members),'Original archive inventory required')
  files={m.name:tar.extractfile(m).read()for m in members}
 for name,value in files.items():require((root/name).is_file()and not(root/name).is_symlink()and (root/name).read_bytes()==value,'Original deployed bytes changed '+name)
 manifest=json.loads(files['source-deployment-manifest.json']);require(manifest['source_commit']==SOURCE,'Original source manifest differs')
 for name,digest in manifest['files'].items():require(sha(root/name)==digest,'Original source file hash differs')
 require(sha(root/'setup-isolated.py')==SETUP_SHA and setup_branch(files['setup-isolated.py'])==FAILED_BRANCH,'Exact documented failed setup required')
 require((root/'source-expected.txt').read_bytes()==(SOURCE+'\n').encode(),'Original expected source differs');validate_bundle(root/'source.bundle')
def validate_existing(root):
 require(root.is_dir()and not root.is_symlink()and root.resolve()==root,'Canonical existing own ROOT required')
 expected={'source-deployment.tar','source.bundle','source-expected.txt','setup-isolated.py','source-deployment-manifest.json','logs','cache','runs','assets-root','scheduler-head','env','retriever-gpu-lib','lib','assets'}
 require({p.name for p in root.iterdir()}==expected,'Exact pre-clone directory inventory required; foreign contents or repository refused')
 require(not(root/'repo').exists()and not(root/'repo').is_symlink()and not(root/'logs/isolation-setup.json').exists(),'Clone/isolation metadata must be absent')
 for name in ['logs','cache','runs','assets-root','scheduler-head','env','assets']:require((root/name).is_dir()and not(root/name).is_symlink(),'Existing own directory required')
 for name in ['runs','cache','scheduler-head']:require(not any((root/name).iterdir()),'Pre-clone run/cache/scheduler directories must be empty')
 require({p.name for p in (root/'env').iterdir()}=={'bin','lib','pyvenv.cfg'}and {p.name for p in (root/'env/bin').iterdir()}=={'python','python3','python3.10'},'Exact existing own env layout required')
 require({p.name for p in (root/'assets').iterdir()}=={'qwen2.5-3b-instruct-aa8e725'}and {p.name for p in (root/'assets-root').iterdir()}=={'assets','assets.json'},'Exact existing official/retrieval aliases required')
 env=root/'env';require((env/'pyvenv.cfg').read_bytes()==(OLD/'env/pyvenv.cfg').read_bytes(),'Existing own env config differs')
 for name in ['python','python3','python3.10']:require((env/'bin'/name).is_symlink()and (env/'bin'/name).resolve()==(OLD/'env/bin/python').resolve(),'Existing own interpreter alias differs')
 for name,target in [('env/lib/python3.10/site-packages',OLD/'env/lib/python3.10/site-packages'),('retriever-gpu-lib',OLD/'retriever-gpu-lib'),('lib',OLD/'lib'),('assets-root/assets',OLD/'assets-root/assets'),('assets/qwen2.5-3b-instruct-aa8e725',MODEL)]:
  p=root/name;require(p.is_symlink()and p.resolve()==target.resolve(),'Existing logical read-only alias differs '+name)
 require((root/'assets-root/assets.json').read_bytes()==(OLD/'assets-root/assets.json').read_bytes(),'Existing retrieval asset metadata differs')
 require(MODEL.is_dir()and not MODEL.is_symlink(),'Official manifest-only mirror required')
 validate_archive(root)
def finish(root):
 validate_existing(root)
 subprocess.run(['git','clone','--branch',BRANCH,str(root/'source.bundle'),str(root/'repo')],check=True)
 require(subprocess.check_output(['git','-C',str(root/'repo'),'rev-parse','HEAD'],text=True).strip()==SOURCE,'Recovered source HEAD differs')
 require(not subprocess.check_output(['git','-C',str(root/'repo'),'status','--porcelain'],text=True).strip(),'Recovered source dirty')
 record=dict(source_commit=SOURCE,root=str(root),source_bundle_sha256=sha(root/'source.bundle'),own_env_prefix=str(root/'env'),shared_dependency_site_packages=str((root/'env/lib/python3.10/site-packages').resolve()),shared_read_only_retrieval_assets=str((root/'assets-root/assets').resolve()),shared_model=str(MODEL),official_model_alias=str(root/'assets/qwen2.5-3b-instruct-aa8e725'),model_asset_scope='OFFICIAL_MANIFEST_FILES_ONLY_NOT_CHECKPOINT; LOGICAL_READ_ONLY_REHASHED_BEFORE_ADMISSION',shared_data=str(COMMON/'common-data'),old_source_unchanged=subprocess.check_output(['git','-C',str(OLD/'repo'),'rev-parse','HEAD'],text=True).strip(),scope='SOURCE_ENV_DEPLOYMENT_ONLY_NO_GPU_NO_LEGACY_RECEIPT',recovery_scope='ADDITIVE_CLONE_FINISH_EXISTING_VALIDATED_ENV_UNCHANGED',original_failed_setup_sha256=SETUP_SHA,original_deployment_archive_sha256=TAR_SHA,actual_branch=BRANCH,recovery_script_sha256=sha(__file__))
 with (root/'logs/isolation-setup.json').open('x')as f:json.dump(record,f,indent=2);f.flush();os.fsync(f.fileno())
 fd=os.open(root/'logs',os.O_RDONLY);os.fsync(fd);os.close(fd)
 print(json.dumps(record))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--execute-worker',action='store_true');parser.add_argument('--bundle',type=Path);args=parser.parse_args()
 require(not sys.flags.optimize,'Optimized interpreter forbidden')
 if args.execute_worker:
  require(sys.platform=='linux'and os.uname().nodename=='h100-5'and sys.prefix==str(ROOT/'env'),'Actual own worker environment required');finish(ROOT)
 else:
  require(args.bundle is not None,'Local bundle precheck requires --bundle');validate_bundle(args.bundle);print(json.dumps(dict(scope='LOCAL_EXACT_BUNDLE_BRANCH_CHECK_ONLY',source_commit=SOURCE,branch=BRANCH,mutated=False)))
