import pathlib,subprocess,shutil,hashlib,json
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-20261004');old=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo')
assert root.is_dir()and not root.is_symlink()
source=(root/'source-expected.txt').read_text().strip();assert len(source)==40
for name in ('tools','logs','cache','runs','assets','assets-root'): (root/name).mkdir(exist_ok=True)
for name in ('env','retriever-gpu-lib','lib'):
 p=root/name
 if not p.exists():p.symlink_to(old/name,target_is_directory=True)
 assert p.is_symlink()and p.resolve()==(old/name).resolve()
a=root/'assets-root';shared=a/'assets'
if not shared.exists():shared.symlink_to(old/'assets-root/assets',target_is_directory=True)
assert shared.is_symlink()and shared.resolve()==(old/'assets-root/assets').resolve()
p=a/'assets.json';original=old/'assets-root/assets.json'
if p.exists():assert p.read_bytes()==original.read_bytes()
else:shutil.copyfile(original,p)
repo=root/'repo'
if not repo.exists():subprocess.run(['git','clone','--branch','codex/bapo-e1-instruct-300',str(root/'source.bundle'),str(repo)],check=True)
assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==source
assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True)
record={'root':str(root),'source_commit':source,'source_bundle_sha256':hashlib.sha256((root/'source.bundle').read_bytes()).hexdigest(),'read_only_shared_targets':{name:str((root/name).resolve())for name in ('env','retriever-gpu-lib','lib')},'retrieval_assets':str(shared.resolve()),'assets_manifest_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'new_model_root':str(root/'assets/qwen2.5-3b-instruct-aa8e725'),'common_data_root':str(root/'common-data'),'old_source_unchanged':subprocess.check_output(['git','-C',str(old/'repo'),'rev-parse','HEAD'],text=True).strip()}
p=root/'logs/isolation-setup.json'
if p.exists():assert json.loads(p.read_text())==record
else:p.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
