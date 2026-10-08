"""Scoped source transfer via the user's existing H100 route."""
from pathlib import Path
import shlex,subprocess,json
B=Path(__file__).resolve().parents[1]
R=Path(__file__).resolve().parent
code="""
import io,tarfile,pathlib,hashlib,sys,json,os,subprocess
if sys.flags.optimize: raise RuntimeError('optimized Python forbidden for scoped deployment')
raw=sys.stdin.buffer.read()
assert hashlib.sha256(raw).hexdigest()=='277d01d3008c8d88d0d772ff98b54b8536bf38c9e7448428409242698c301507'
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-safety-r3-20261006')
assert os.uname().nodename=='h100-5' and not root.exists() and not root.is_symlink()
root.mkdir(mode=0o700)
(root/'source-deployment.tar').write_bytes(raw)
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:') as tar:
 names={'source.bundle','source-expected.txt','setup-isolated.py','source-deployment-manifest.json'}
 assert set(tar.getnames())==names and len(tar.getmembers())==4
 for item in tar.getmembers():
  assert item.isfile() and item.name in names
  with (root/item.name).open('xb') as dest:dest.write(tar.extractfile(item).read())
manifest=json.loads((root/'source-deployment-manifest.json').read_text())
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==value for name,value in manifest['files'].items())
subprocess.run(['python3','-B',str(root/'setup-isolated.py')],check=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
"""
command='python3 -B -c '+shlex.quote(code)
with (B/'evidence/fresh200-r3-source-deployment.actual.log').open('x')as log:
 result=subprocess.run(['python3',str(B/'h100_exec.py'),'--socks','--timeout','120','--stdin-file',str(R/'source-deployment.tar'),command],stdout=log,stderr=subprocess.STDOUT)
print(json.dumps({'exit_code':result.returncode,'log':str(B/'evidence/fresh200-r3-source-deployment.actual.log')}))
raise SystemExit(result.returncode)
