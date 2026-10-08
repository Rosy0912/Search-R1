from pathlib import Path
import shlex,subprocess,json
B=Path(__file__).resolve().parents[1];R=Path(__file__).resolve().parent
code="""
import io,tarfile,pathlib,hashlib,sys,json,os
if sys.flags.optimize: raise RuntimeError('optimized Python forbidden for scoped deployment')
raw=sys.stdin.buffer.read();assert hashlib.sha256(raw).hexdigest()=='f08a6d2834fadb879080cf39f33db1b70a8bda93b2cffeefcc5e797c14222045'
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-safety-r4-20261006')
assert os.uname().nodename=='h100-5' and root.is_dir() and not root.is_symlink()
assert (root/'source-expected.txt').read_text().strip()=='5121a8d9349eadb5a81119cc4573165338e041bd'
assert not (root/'tools').exists() and not (root/'logs/tools-manifest.json').exists()
(root/'tools').mkdir(mode=0o700)
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:') as tar:
 for item in tar.getmembers():
  p=pathlib.PurePosixPath(item.name)
  assert item.isfile() and ((len(p.parts)==2 and p.parts[0]=='tools') or item.name=='logs/tools-manifest.json') and '..' not in p.parts
  with (root/item.name).open('xb')as f:f.write(tar.extractfile(item).read())
manifest=json.loads((root/'logs/tools-manifest.json').read_text())
assert set(manifest['files'])=={p.name for p in (root/'tools').iterdir() if p.is_file()}
assert all(hashlib.sha256((root/'tools'/name).read_bytes()).hexdigest()==value for name,value in manifest['files'].items())
print(json.dumps(dict(source_commit=manifest['source_commit'],tools_manifest_sha256=hashlib.sha256((root/'logs/tools-manifest.json').read_bytes()).hexdigest(),files=len(manifest['files']),scope='CODE_DEPLOYMENT_NO_LAUNCH_PERMISSION')))
"""
with (B/'evidence/fresh200-r4-tools-deployment.actual.log').open('x')as log:
 result=subprocess.run(['python3',str(B/'h100_exec.py'),'--socks','--timeout','90','--stdin-file',str(R/'tools-deployment.tar'),'python3 -B -c '+shlex.quote(code)],stdout=log,stderr=subprocess.STDOUT)
print(json.dumps(dict(exit_code=result.returncode)));raise SystemExit(result.returncode)
