from pathlib import Path
import shlex,subprocess,json
B=Path(__file__).resolve().parents[1];R=Path(__file__).resolve().parent
code="""
import io,tarfile,pathlib,hashlib,sys,json,os
if sys.flags.optimize: raise RuntimeError('optimized Python forbidden for scoped deployment')
raw=sys.stdin.buffer.read();assert hashlib.sha256(raw).hexdigest()=='e7e7c89394cccf079523986efc50b1e399e6f06c15c94b0dcbad885145ca7a98'
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-behavior-20261006')
assert os.uname().nodename=='h100-16' and not root.exists() and not root.is_symlink()
root.mkdir(mode=0o700)
(root/'logs').mkdir(mode=0o700)
(root/'scheduler-head').mkdir(mode=0o700)
(root/'source-expected.txt').write_text('90755a561abd239fbb23882939410e643bb913ae' + chr(10))
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
print(json.dumps(dict(source_commit=manifest['source_commit'],tools_manifest_sha256=hashlib.sha256((root/'logs/tools-manifest.json').read_bytes()).hexdigest(),files=len(manifest['files']),scope='HEAD_CODE_DEPLOYMENT_NO_LAUNCH_PERMISSION')))
"""
with (B/'evidence/fresh200-behavior-head-code-deployment.actual.log').open('x')as log:
 result=subprocess.run(['python3',str(B/'h100_exec.py'),'--socks','--node','h100-16','--timeout','90','--stdin-file',str(R/'tools-deployment.tar'),'python3 -B -c '+shlex.quote(code)],stdout=log,stderr=subprocess.STDOUT)
print(json.dumps(dict(exit_code=result.returncode)));raise SystemExit(result.returncode)
