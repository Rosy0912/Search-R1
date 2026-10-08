import pathlib,os,json,shutil,hashlib,time
old=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-20261004');root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-statefix-20261004');base=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo')
assert old.is_dir() and not old.is_symlink();assert root.is_dir() and not root.is_symlink()
assert not (root/"repo").exists() and not (root/"scheduler-head").exists()
for name in ('tools','logs','cache','runs','assets','common-data','assets-root','evidence-tools'):(root/name).mkdir(exist_ok=True)
for name in ('env','retriever-gpu-lib','lib'):assert (root/name).is_symlink() and (root/name).resolve()==(base/name).resolve()
assert (root/'assets-root/assets').is_symlink() and (root/'assets-root/assets').resolve()==(base/'assets-root/assets').resolve()
shutil.copy2(old/'assets-root/assets.json',root/'assets-root/assets.json')
records=[]
for directory in ('common-data','assets/qwen2.5-3b-instruct-aa8e725'):
 target=root/directory;target.mkdir(parents=True,exist_ok=True)
 for src in (old/directory).rglob('*'):
  assert not src.is_symlink()
  dst=target/src.relative_to(old/directory)
  if src.is_dir():dst.mkdir(parents=True,exist_ok=True);continue
  assert src.is_file() and src.stat().st_dev==root.stat().st_dev
  dst.parent.mkdir(parents=True,exist_ok=True)
  if dst.exists():assert dst.is_file() and not dst.is_symlink() and dst.stat().st_ino==src.stat().st_ino and dst.stat().st_dev==src.stat().st_dev
  else:os.link(src,dst)
  records.append({'source':str(src),'destination':str(dst),'bytes':dst.stat().st_size,'device':dst.stat().st_dev,'inode':dst.stat().st_ino,'immutable_asset_reuse':True})
record={'status':'NEW_ROOT_INPUT_PREPARATION_ONLY_NOT_SOURCE_DEPLOYED_OR_JOB_SUBMITTED','created_epoch':time.time(),'old_root_preserved':str(old),'new_root':str(root),'files':records}
(root/'logs/input-reuse-preparation.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record))
