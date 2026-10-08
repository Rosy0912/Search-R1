import pathlib,tarfile,json,hashlib,os
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300')
target=root/'candidates/tree-cbb74d0-a42c2f86';assert not target.exists()
archive=root/'transfers/tree-cbb74d0-a42c2f86.tar.gz';pending=target.with_name(target.name+'.incoming-'+str(os.getpid()));pending.mkdir(parents=True)
with tarfile.open(archive) as t:
 for member in t.getmembers():
  name=pathlib.PurePosixPath(member.name)
  assert not name.is_absolute() and '..' not in name.parts and member.isfile()
 t.extractall(pending)
m=json.loads((pending/'manifest.json').read_bytes());assert hashlib.sha256((pending/'manifest.json').read_bytes()).hexdigest()=='a42c2f86d860836aab2919c277653c57a6daa3d659523244c1f8335d373ecc50'
for file,expected in m['files'].items():assert hashlib.sha256((pending/file).read_bytes()).hexdigest()==expected,file
for file,expected in m['external_files'].items():assert hashlib.sha256(pathlib.Path(file).read_bytes()).hexdigest()==expected,file
os.rename(pending,target);print(json.dumps(dict(status='IMMUTABLE_SOURCE_PACKAGE_SHA_VERIFIED_NOT_SUBMITTED',path=str(target),files=len(m['files']),code=m['code_commit'])))
