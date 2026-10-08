import pathlib,tarfile,hashlib,subprocess,json,os
SOURCE='733097feeae371b4efd8fd48940c46ff36da839b'
ROOT=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-precision-r5-20261004')
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
EXPECTED={'submit.sh': '83f5083f1e2d5d4b6630d6c7bd7762f14898f6348ed433759d42f2c24c029907', 'hardware.sbatch': '069631588f4665ec0c873ed8f6b19333f2efe858794a024fc076f99dbd1277ee', 'formal300.sbatch': 'f49136220f5ea206a855a92ff118aecfddbb2f2909fda9bfd4ce90a49aa960f7', 'watch-terminal.py': 'a72aced457880e8956a346a9f4d55e3f4ed0113ca20f25af278e319e102ae2ff', 'approved-launch-hashes.json': '3e64e1257b1b76ce70107c3a4b72b206a34c9c78af113cd5ddce939f3a9f47c8'}
ARCHIVE_SHA='389c5c3f5838aeca28ff30c89fc67585d3644c5093545915021227ef22bbc9b5'
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
