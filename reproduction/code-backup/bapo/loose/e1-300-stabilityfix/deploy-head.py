import pathlib,sys,tarfile,io,hashlib,json,subprocess
ROOT=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-stabilityfix-20261005')
EXPECTED={'submit.sh': 'ac7d1aba890e129d627f1db3336970c8b27e9054b4f24d43446744dd923b1dc4', 'hardware.sbatch': '84daee9831a3871fc9237c9de831bf07d9283c238c75c8b8b192284d85f0b800', 'formal300.sbatch': '0f88dec1489ea7acb373b485554d08e3a62c9454bee7a13395390500bbf135f9', 'approved-launch-hashes.json': 'e4d3706876a8d91be5c8f823979ed32b19c5c639837631c6d0e125f5bbdfa685'}
ARCHIVE_SHA='648769cca2738432997872104859cc5388b7900442a3fcf22bf51d35afa24456'
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
