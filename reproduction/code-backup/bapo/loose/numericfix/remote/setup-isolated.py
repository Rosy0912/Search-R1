import pathlib,subprocess,shutil,hashlib,json
old=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo');root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-rewardfix-numeric');source='1cceda2f7e201de3f68c876bb9a3b895563c20ce'
assert root.is_dir() and not root.is_symlink()
for name in ('tools','logs','cache','runs','prepared','assets-root'): (root/name).mkdir(exist_ok=True)
for name in ('env','retriever-gpu-lib','lib'):
 p=root/name
 if not p.exists():p.symlink_to(old/name,target_is_directory=True)
 assert p.resolve()==(old/name).resolve()
a=root/'assets-root'; shared=a/'assets'
if not shared.exists():shared.symlink_to(old/'assets-root/assets',target_is_directory=True)
assert shared.resolve()==(old/'assets-root/assets').resolve()
for src,dst in [(old/'assets-root/assets.json',a/'assets.json'),(old/'logs/derived-assets.sha256',root/'logs/derived-assets.sha256')]:
 if dst.exists():assert dst.read_bytes()==src.read_bytes()
 else:shutil.copy2(src,dst)
data=a/'data/paper-grpo'
if not data.exists():shutil.copytree(old/'assets-root/data/paper-grpo',data)
assert not data.is_symlink()
for p in sorted(data.rglob('*')):
 if p.is_file():
  assert not p.is_symlink(); q=old/'assets-root/data/paper-grpo'/p.relative_to(data);assert hashlib.sha256(p.read_bytes()).digest()==hashlib.sha256(q.read_bytes()).digest()
repo=root/'repo'
if not repo.exists():subprocess.run(['git','clone','--branch','reproduce/bapo',str(root/'source.bundle'),str(repo)],check=True)
assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==source
assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True)
record={'source_commit':source,'isolated_root':str(root),'source_bundle_sha256':hashlib.sha256((root/'source.bundle').read_bytes()).hexdigest(),'shared_read_only_targets':{k:str((root/k).resolve()) for k in ('env','retriever-gpu-lib','lib')},'shared_assets_only':str(shared.resolve()),'copied_assets_manifest_sha256':hashlib.sha256((a/'assets.json').read_bytes()).hexdigest(),'copied_paper_data_files':{str(p.relative_to(a)):hashlib.sha256(p.read_bytes()).hexdigest() for p in data.rglob('*') if p.is_file()},'old_source_unchanged':subprocess.check_output(['git','-C',str(old/'repo'),'rev-parse','HEAD'],text=True).strip(),'forbidden_writes':'old repo/tools/logs/runs/receipts/env/assets; no old prepare/bootstrap/install scripts'}
(root/'logs/isolation-setup.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
