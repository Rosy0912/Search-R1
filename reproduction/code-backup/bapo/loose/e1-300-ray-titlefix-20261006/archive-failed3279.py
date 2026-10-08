import pathlib,tarfile,hashlib,json,base64,os
r=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-ray-titlefix-20261006');run=r/'runs/bapo-instruct300-hardware-3279';files=[]
for p in run.rglob('*'):
 parts=p.relative_to(run).parts
 if any(x in ('runtime','cache','tmp','__pycache__')for x in parts):continue
 assert not p.is_symlink()
 if p.is_file():files.append(p)
files+=list((r/'tools').iterdir());files+=[r/'source-expected.txt']
files+=[p for p in (r/'logs').iterdir()if p.is_file() and not p.name.startswith('failure3279-')]
for p in files:assert p.is_file()and not p.is_symlink()
inventory={str(p.relative_to(r)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}for p in files}
a=r/'logs/failure3279-raw.tar.gz'
assert not a.exists()
with tarfile.open(a,'w:gz')as t:
 for p in files:t.add(p,arcname=str(p.relative_to(r)),recursive=False)
for p in files:assert inventory[str(p.relative_to(r))]['sha256']==hashlib.sha256(p.read_bytes()).hexdigest()
raw=a.read_bytes();print(json.dumps({'status':'FAILED3279_RAW_PRESERVED_NOT_GPU_ACCEPTANCE','files':inventory,'archive':{'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'base64':base64.b64encode(raw).decode()}}))
