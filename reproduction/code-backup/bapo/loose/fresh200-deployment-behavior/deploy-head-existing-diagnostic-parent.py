"""Exclusive production metadata addition beside the exact existing numeric diagnostic.
Default checks the fixed archive only. --execute-head is restricted to actual h100-16.
Never writes diagnostic files, submits jobs, or publishes approval.
"""
import argparse,hashlib,io,json,os,sys,tarfile
from pathlib import Path
ROOT=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-behavior-20261006')
SOURCE='90755a561abd239fbb23882939410e643bb913ae'
CHILD='numeric-diagnostic-3310-20261006'
DIAG_SHA='8a79b017d526de2cb9ff78f3e701e2572a5da0029545956d985394b72d492830'
TOOLS_SHA='ec1e92369e03b141dc6a528e795a9459bc331e4296cfc65f42d895c2db0f0256'
TAR_SHA='e7e7c89394cccf079523986efc50b1e399e6f06c15c94b0dcbad885145ca7a98'
def require(ok,message):
 if not ok:raise ValueError(message)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def validate_parent(root):
 require(root.is_dir()and not root.is_symlink()and root.resolve()==root,'Canonical existing own parent required')
 require({p.name for p in root.iterdir()}=={CHILD},'Only exact diagnostic child allowed; production metadata or foreign child refused')
 child=root/CHILD
 require(child.is_dir()and not child.is_symlink(),'Ordinary diagnostic directory required')
 require(all(not p.is_symlink()for p in child.rglob('*')),'Diagnostic symlinks refused')
 require((child/'source-expected.txt').read_bytes()==(SOURCE+'\n').encode(),'Existing diagnostic source differs')
 raw=(child/'logs/tools-manifest.json').read_bytes();require(sha(raw)==DIAG_SHA,'Existing diagnostic manifest differs')
 manifest=json.loads(raw);require(manifest['source_commit']==SOURCE and len(manifest['files'])==13,'Exact diagnostic source and inventory required')
 require({p.name for p in (child/'tools').iterdir()}==set(manifest['files']),'Exact diagnostic tool names required')
 for name,digest in manifest['files'].items():
  p=child/'tools'/name;require(p.is_file()and sha(p.read_bytes())==digest,'Diagnostic tool bytes differ')
def validate_archive(blob):
 require(sha(blob)==TAR_SHA,'Fixed production archive required')
 with tarfile.open(fileobj=io.BytesIO(blob),mode='r:')as tar:
  members=tar.getmembers();require(len({m.name for m in members})==len(members),'Duplicate archive member')
  require(all(m.isfile()and not Path(m.name).is_absolute()and '..'not in Path(m.name).parts for m in members),'Regular safe archive members required')
  files={m.name:tar.extractfile(m).read()for m in members}
 raw=files['logs/tools-manifest.json'];require(sha(raw)==TOOLS_SHA,'Production manifest differs')
 manifest=json.loads(raw);require(manifest['source_commit']==SOURCE and len(manifest['files'])==21,'Exact production source and 21 tools required')
 require(set(files)=={'logs/tools-manifest.json'}|{'tools/'+n for n in manifest['files']},'Only production metadata allowed')
 for name,digest in manifest['files'].items():require(sha(files['tools/'+name])==digest,'Production tool hash differs')
 return files
def deploy(root,files):
 validate_parent(root)
 for name in ['logs','tools','scheduler-head']:(root/name).mkdir(mode=0o700)
 files=dict(files);files['source-expected.txt']=(SOURCE+'\n').encode()
 for name,blob in files.items():
  with (root/name).open('xb')as f:f.write(blob);f.flush();os.fsync(f.fileno())
 for name in ['logs','tools','scheduler-head','']:
  fd=os.open(root/name,os.O_RDONLY);os.fsync(fd);os.close(fd)
 return dict(scope='HEAD_CODE_DEPLOYMENT_NO_LAUNCH_PERMISSION',source_commit=SOURCE,tools_manifest_sha256=TOOLS_SHA,files=21,diagnostic_untouched=True,gpu_submitted=False,approval_published=False)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--execute-head',action='store_true');args=p.parse_args()
 require(not sys.flags.optimize,'Optimized interpreter forbidden')
 files=validate_archive(args.archive.read_bytes())
 if args.execute_head:
  require(sys.platform=='linux'and os.uname().nodename=='h100-16','Actual scheduler head required');print(json.dumps(deploy(ROOT,files)))
 else:print(json.dumps(dict(scope='LOCAL_FIXED_ARCHIVE_CHECK_ONLY',source_commit=SOURCE,files=21,mutated=False)))
