import os,json,subprocess,datetime,hashlib,tarfile
from pathlib import Path
assert os.uname().nodename=='h100-11'
r=Path('/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v3-20261009');b=json.loads((r/'scheduler-head/formal-submit-receipt.json').read_text());assert str(b['job'])=='3353';run=Path(b['run']);out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job':3353,'logs':{},'commands':{},'files':[]}
for args in [['scontrol','show','job','3353','-o'],['nvidia-smi','-q','-d','ECC,PAGE_RETIREMENT,ROW_REMAPPER'],['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader'],['df','-B1',str(r),'/dataset']]:
 p=subprocess.run(args,capture_output=True,text=True,timeout=20);out['commands'][' '.join(args)]={'exit':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
(r/'evidence').mkdir(exist_ok=True)
archive=r/'evidence/formal3353-terminal-small.tar';assert not archive.exists()
with tarfile.open(archive,'w')as tar:
 for p in run.rglob('*'):
  if p.is_file() and not p.is_symlink() and p.suffix not in ('.pt','.safetensors','.bin') and p.stat().st_size<20_000_000:
   raw=p.read_bytes();out['files'].append({'path':str(p.relative_to(run)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()});tar.add(p,arcname=str(p.relative_to(run)),recursive=False)
for n in ['driver-time.log','train.log','original-exit-code.txt','cleanup-exit-code.txt','cleanup.json','formal-supervisor.log']:
 p=run/n
 if p.exists():out['logs'][n]=p.read_text()[-40000:]
out['archive']={'path':str(archive),'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}
print(json.dumps(out))
