"""Head-only exact diagnostic metadata mirror. Never submits or runs a GPU probe."""
import argparse,hashlib,io,json,os,sys,tarfile
from pathlib import Path
ROOT=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-safety-r5-20261006/probability-diagnostic-3307-20261006')
SOURCE='aec5e057093b4c491bfb0a47abd7605d98981b56'
def require(ok,message):
 if not ok:raise ValueError(message)
def sha(blob):return hashlib.sha256(blob).hexdigest()
def validate(blob,expected_tar,expected_manifest):
 require(sha(blob)==expected_tar,'Exact independently reviewed archive required')
 with tarfile.open(fileobj=io.BytesIO(blob),mode='r:gz')as archive:
  members=archive.getmembers();names=[m.name for m in members]
  require(len(names)==len(set(names)),'Duplicate archive members refused')
  require(all(m.isfile()and not Path(m.name).is_absolute()and '..'not in Path(m.name).parts for m in members),'Regular canonical archive files required')
  raw={m.name:archive.extractfile(m).read()for m in members}
 require(set(raw)=={'source-expected.txt','logs/tools-manifest.json'}|{'tools/'+n for n in json.loads(raw['logs/tools-manifest.json'])['files']},'Only exact tool/source metadata may be mirrored')
 require(raw['source-expected.txt'].decode().strip()==SOURCE and sha(raw['logs/tools-manifest.json'])==expected_manifest,'Fixed source/manifest required')
 manifest=json.loads(raw['logs/tools-manifest.json']);require(manifest['source_commit']==SOURCE and len(manifest['files'])==12,'Exact diagnostic inventory required')
 for name,digest in manifest['files'].items():require(sha(raw['tools/'+name])==digest,'Diagnostic tool hash mismatch')
 return raw
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--expected-archive-sha',required=True);p.add_argument('--expected-manifest-sha',required=True);args=p.parse_args()
 require(not sys.flags.optimize and sys.platform=='linux'and os.uname().nodename=='h100-16','Actual scheduler head, unoptimized interpreter required')
 raw=validate(args.archive.read_bytes(),args.expected_archive_sha,args.expected_manifest_sha)
 require(not ROOT.exists(),'Existing diagnostic head root refused; never merge old receipts')
 ROOT.mkdir()
 for name in ['tools','logs','runs','scheduler-head']:(ROOT/name).mkdir()
 for name,blob in raw.items():
  path=ROOT/name
  with path.open('xb')as f:f.write(blob);f.flush();os.fsync(f.fileno())
  if name.startswith('tools/')and name.endswith(('.sh','.sbatch')):path.chmod(0o700)
 print(json.dumps({'scope':'ZERO_ADAM_DIAGNOSTIC_HEAD_METADATA_ONLY','root':str(ROOT),'source_commit':SOURCE,'tools_manifest_sha256':args.expected_manifest_sha,'gpu_submitted':False,'approval_published':False}))
