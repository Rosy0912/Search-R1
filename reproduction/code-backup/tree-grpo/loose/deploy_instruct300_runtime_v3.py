import pathlib,tarfile,json,hashlib,os
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300')
target=root/'candidates/tree-f994af3-aa9e3161';assert not target.exists()
archive=root/'transfers/tree-f994af3-aa9e3161.tar.gz';pending=target.with_name(target.name+'.incoming-'+str(os.getpid()));pending.mkdir(parents=True)
with tarfile.open(archive) as t:
 for member in t.getmembers():
  name=pathlib.PurePosixPath(member.name)
  assert not name.is_absolute() and '..' not in name.parts and member.isfile()
 t.extractall(pending)
m=json.loads((pending/'manifest.json').read_bytes());assert hashlib.sha256((pending/'manifest.json').read_bytes()).hexdigest()=='aa9e3161ee8a211e6e311d2a87be9d456371f0f569d0e0c9c1703e3a575ba69f'
for file,expected in m['files'].items():assert hashlib.sha256((pending/file).read_bytes()).hexdigest()==expected,file
for file,expected in m['external_files'].items():assert hashlib.sha256(pathlib.Path(file).read_bytes()).hexdigest()==expected,file
os.rename(pending,target);print(json.dumps(dict(status='IMMUTABLE_SOURCE_PACKAGE_SHA_VERIFIED_NOT_SUBMITTED',path=str(target),files=len(m['files']),code=m['code_commit'])))
