"""Copy only verified official bytes into this task's new assets directory."""
from pathlib import Path
import json,hashlib,os,shutil
R=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-safety-20261006')
S=Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-stabilityfix-r2-20261005/assets/qwen2.5-3b-instruct-aa8e725')
D=R/'assets/qwen2.5-3b-instruct-aa8e725'
def require(ok,msg):
 if not ok:raise RuntimeError(msg)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
require(os.uname().nodename=='h100-5' and R.is_dir() and not R.is_symlink(),'wrong target/root')
m=S/'VERIFIED-MODEL-MANIFEST.json';require(sha(m)=='7f53ce6c010ac076b1ec482031bea7c9e6b8d0a096a61b880fe8ad4971bc9e44','manifest differs')
doc=json.loads(m.read_text());require(doc['repo']=='Qwen/Qwen2.5-3B-Instruct' and doc['revision']=='aa8e72537993ba99e69dfaafa59ed015b17504d1','wrong official source')
require(not D.exists() and not D.is_symlink() and not (R/'assets').is_symlink(),'never overwrite/redirect a prior model isolation attempt')
D.mkdir(parents=True,mode=0o700);records=[]
for entry in doc['files']:
 name=entry['file'];require(Path(name).name==name,'invalid member')
 source=S/name;target=D/name
 require(source.is_file() and not source.is_symlink() and source.stat().st_size==entry['bytes'] and sha(source)==entry['sha256'],'official source bytes differ')
 with source.open('rb')as inp,target.open('xb')as out:shutil.copyfileobj(inp,out,8388608);out.flush();os.fsync(out.fileno())
 require(target.stat().st_size==entry['bytes'] and sha(target)==entry['sha256'],'copied official bytes differ')
 records.append(dict(file=name,bytes=entry['bytes'],sha256=entry['sha256']))
with m.open('rb')as inp,(D/m.name).open('xb')as out:shutil.copyfileobj(inp,out)
require(sha(D/m.name)==sha(m),'copied manifest differs')
require({p.name for p in D.iterdir()}=={e['file'] for e in doc['files']}|{m.name},'copied inventory is not exactly official manifest')
require(all(p.is_file() and not p.is_symlink() and p.resolve().parent==D.resolve() for p in D.iterdir()),'copied inventory escaped own directory')
record=dict(source=str(S),destination=str(D),files=records,model_manifest_sha256=sha(m),retained_original_inventory=[p.name for p in S.iterdir()],scope='VERIFIED_OFFICIAL_BYTES_COPY_NO_MODEL_UPDATE')
with (R/'logs/model-isolation.json').open('x')as f:json.dump(record,f,indent=2)
print(json.dumps(record))
