import pathlib,sys,tarfile,io,hashlib,json,subprocess
ROOT=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-stabilityfix-r2-20261005')
EXPECTED={'submit.sh': '34fdb8380fba79ca426e7080261e222a6083701d2ae9d0c06cdfc99e766c1e30', 'hardware.sbatch': '6900d9c63ef631c29b148b68acc1e0fbbbed9d2b938dfc0a490a66afc3976e1f', 'formal300.sbatch': '1f9014c24748a7cb44779284fe3b499e110f1188cb396fe55edec9e1118a5dd8', 'approved-launch-hashes.json': '95ba14b77a5c5f3ad4ca17bd98d3b2a3fa4dc865bfb2edb708ddb130d2d928ca'}
ARCHIVE_SHA='483ca450af2326bfbe39d28d75c7437805d6c476a0721040e34bc575234529fa'
KIND='head'
raw=sys.stdin.buffer.read();assert hashlib.sha256(raw).hexdigest()==ARCHIVE_SHA
if KIND=='head':
 if not ROOT.exists():ROOT.mkdir(parents=True)
assert ROOT.is_dir() and not ROOT.is_symlink() and ROOT.resolve()==ROOT
if KIND=='worker':
 assert subprocess.check_output(['git','-C',str(ROOT/'repo'),'rev-parse','HEAD'],text=True).strip()=='39cdcdd1404213b897bb0de99cc52d72707682f0'
 assert not subprocess.check_output(['git','-C',str(ROOT/'repo'),'status','--porcelain'],text=True)
 assert not list((ROOT/'runs').iterdir())
 record=json.loads((ROOT/'logs/cpu-gate.json').read_text());assert record['passed']==305 and record['exit_code']==record['skipped']==record['failed']==0
 destination=ROOT
else:
 destination=ROOT/'scheduler-head';assert not destination.exists()
payload={}
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz')as bundle:
 members=bundle.getmembers();assert len(members)==len(EXPECTED) and {m.name for m in members}==set(EXPECTED)
 for m in members:
  q=pathlib.PurePosixPath(m.name);assert m.isfile() and not q.is_absolute() and '..'not in q.parts
  data=bundle.extractfile(m).read();assert hashlib.sha256(data).hexdigest()==EXPECTED[m.name]
  payload[m.name]=(data,m.mode)
for name,(data,mode)in payload.items():
 p=destination/name;assert destination in p.parents
 if p.exists()or p.is_symlink():assert p.is_file()and not p.is_symlink()and hashlib.sha256(p.read_bytes()).hexdigest()==EXPECTED[name]
 if KIND=='worker':assert p.parent.resolve()==p.parent
if KIND=='head':destination.mkdir()
for name,(data,mode)in payload.items():
 p=destination/name;p.parent.mkdir(parents=True,exist_ok=True)
 if not p.exists():p.open('xb').write(data)
 p.chmod(mode);assert hashlib.sha256(p.read_bytes()).hexdigest()==EXPECTED[name]
print(json.dumps({'status':'VERIFIED_DEPLOYMENT_NO_SUBMISSION','kind':KIND,'files':len(payload),'root':str(ROOT)}))
