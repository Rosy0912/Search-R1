import pathlib,tarfile,hashlib,subprocess,json,os
SOURCE='733097feeae371b4efd8fd48940c46ff36da839b'
ROOT=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-monitor-r6-20261004')
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
EXPECTED={'approved-launch-hashes.json': 'f2bc5d167b0044f27fe27548575908f1ee8d43a8d9aa5be8725ae37ffea19a6e', 'formal300.sbatch': 'fd7b8f89f3fedd9ddb9e60ecd4339f1f5367d136a7df5a6ba1c86c94264aff30', 'hardware.sbatch': '9ff9eabcfa9eb7fbb977dc1d4b03adf0c12775550d083dd182c68ddc0f35def4', 'submit.sh': '92b422c74a98c28bd78de86d97b877c9dfa93855dd01f35f859f839ca2a28edc', 'watch-terminal.py': 'e63e8cb7c426c64facc7ba24d8d0d34600eece51287b8e6dd9ee741081c1c5ac'}
ARCHIVE_SHA='8a26fb81fa09a54a231a90194943ca247f96e024fc3b712649cd728f4756164a'
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
