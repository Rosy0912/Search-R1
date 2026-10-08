import pathlib,tarfile,hashlib,subprocess,json,os
SOURCE='733097feeae371b4efd8fd48940c46ff36da839b'
ROOT=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-precision-r4-20261004')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def decode_bundle(path,expected):
 result={}
 with tarfile.open(path,'r:gz')as t:
  members=t.getmembers();assert len(members)==len(expected)and{m.name for m in members}==set(expected)
  for m in members:
   q=pathlib.PurePosixPath(m.name);assert m.isfile()and not q.is_absolute()and '..'not in q.parts and all(x not in ('','.','..')for x in q.parts)
   b=t.extractfile(m).read();assert hashlib.sha256(b).hexdigest()==expected[m.name]
   result[m.name]=(b,m.mode&0o777)
 return result
EXPECTED={'submit.sh': 'a7ceac2352fca8a5cfa0d9d00a78fd03e24700a2a13c7aea4224871f4f481a26', 'hardware.sbatch': 'c900761d7ad85fceff9dbba4525abdef234ab201fad9e0d910ccd338e6b7ce7d', 'formal300.sbatch': '45a9634f51c9b9ccaec1275e73d42944559dbc489c0af75782f4d4d31bab3a67', 'watch-terminal.py': '15cfe2686cf9eea8d464c962e726dd6108a3a91cae70b7102adf50577b373e92', 'approved-launch-hashes.json': 'e15577a1bd0071a152a2f9c40e1631a7d81ae6d70a5ddfe189a95f5ec9818e71'}
ARCHIVE_SHA='98b476c8b6b1d664d29c5289ff2dad24294a42c77f23ac7f68f3e3579dba9c74'
p=ROOT/'approved-head-deploy.tar.gz';d=ROOT/'scheduler-head'
assert ROOT.is_dir()and not ROOT.is_symlink()and ROOT.resolve()==ROOT
assert not d.exists()and not d.is_symlink(),'Existing head directory: reconcile without replacing or resubmitting'
assert sha(p)==ARCHIVE_SHA and not p.is_symlink()
payload=decode_bundle(p,EXPECTED)
assert len(payload)==5 and set(payload)=={'submit.sh','hardware.sbatch','formal300.sbatch','watch-terminal.py','approved-launch-hashes.json'}
assert all('/'not in x for x in payload)
bindings=json.loads(payload['approved-launch-hashes.json'][0]);assert bindings['root']==str(ROOT)and bindings['source_commit']==SOURCE
for name in ('submit.sh','hardware.sbatch','formal300.sbatch','watch-terminal.py'):assert bindings[name]==EXPECTED[name]
# The entire exact archive has been verified before creating the fresh durable D.
d.mkdir()
for name,(b,mode)in payload.items():
 with(d/name).open('xb')as f:f.write(b)
 (d/name).chmod(mode);assert sha(d/name)==EXPECTED[name]
print(json.dumps({'status':'APPROVED_FRESH_HEAD_DURABLE_TOOLS_PREPARED_NO_SUBMISSION','path':str(d),'files':EXPECTED,'archive_sha256':ARCHIVE_SHA}),flush=True)
