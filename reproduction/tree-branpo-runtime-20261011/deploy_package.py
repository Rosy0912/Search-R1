"""Default dryrun; explicit execute copies code package only, never submits."""
import argparse,json,shlex,subprocess,tarfile,tempfile
from pathlib import Path
from ssh_retained18 import route
p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--target',choices=['h100-10'],default='h100-10');p.add_argument('--execute',action='store_true');a=p.parse_args()
receipt=json.loads((a.package/'package-build.json').read_text());remote=receipt['remote_package']
commands=[route('test ! -e '+shlex.quote(remote),a.target),route('mkdir -p '+shlex.quote(remote),a.target)]
print(json.dumps(dict(dryrun=not a.execute,remote_package=remote,commands=commands,slurm_submitted=False)))
if a.execute:
 subprocess.run(commands[0],check=True);subprocess.run(commands[1],check=True)
 with tempfile.TemporaryFile() as archive:
  with tarfile.open(fileobj=archive,mode='w') as tar:
   for f in sorted(a.package.rglob('*')):
    if f.is_file():tar.add(f,arcname=str(f.relative_to(a.package)),recursive=False)
  archive.seek(0);subprocess.run(route('tar -xf - -C '+shlex.quote(remote),a.target),stdin=archive,check=True)
 # Target Python verifies source bytes without loading tensor/library assets.
 verifier="import hashlib,json,pathlib; p=pathlib.Path("+repr(remote)+"); m=json.loads((p/'manifest.json').read_text()); assert hashlib.sha256((p/'manifest.json').read_bytes()).hexdigest()=="+repr(receipt['manifest_sha256'])+"; assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==h for n,h in m['files'].items()); print('DEPLOYED_CODE_BYTES_VERIFIED')"
 subprocess.run(route('python3 -c '+shlex.quote(verifier),a.target),check=True)
