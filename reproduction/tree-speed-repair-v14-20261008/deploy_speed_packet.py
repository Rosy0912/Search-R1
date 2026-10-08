"""Clone verified own3339 V13 code on18 and deliver only V14 speed code deltas.

Inline compressed code avoids the stalled nested-SSH stdin stream. No model,
data, index, environment or tensor checkpoint crosses this client.
"""
import base64
import hashlib
import io
import json
from pathlib import Path
import shlex
import tarfile
from datetime import datetime, timezone
import ssh_retained18 as ssh

HERE = Path(__file__).resolve().parent
ROOT = '/dataset/shaoshitong/search-r1-rl/tree-grpo'


def main():
    spec = json.loads((HERE/'deployment-v14.json').read_text())
    previous = json.loads((HERE/'immutable-v13-control-preserved/deployment.json').read_text())
    package = Path(spec['local_package'])
    old_manifest = json.loads((Path(previous['local_package'])/'manifest.json').read_text())
    manifest = json.loads((package/'manifest.json').read_text())
    changed = [name for name, digest in manifest['files'].items()
               if old_manifest['files'].get(name) != digest] + ['manifest.json']
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz') as stream:
        for name in changed:
            stream.add(package/name, arcname='instruct300/candidates/'+Path(spec['package']).name+'/'+name)
        for local, key in [('human-authority-v14.json', 'authority_path'),
                           ('direct-release-v14.json', 'release_path'),
                           ('formal-speed-v14.sbatch', 'script_path')]:
            stream.add(HERE/local, arcname=spec[key][len(ROOT)+1:])
    raw = buffer.getvalue()
    if len(raw) > 1000000:
        raise RuntimeError('Unexpectedly large code-only delta')
    digest = hashlib.sha256(raw).hexdigest()
    shared = ROOT+'/.node-migration/speed-v14-code-delta-'+digest[:16]
    parts = []
    for index, offset in enumerate(range(0, len(raw), 16384)):
        part = raw[offset:offset+16384]
        path = shared+'/part-'+str(index)
        source = "from pathlib import Path\nimport base64,hashlib,json\n"
        source += 'p=Path('+repr(path)+');raw=base64.b64decode('+repr(base64.b64encode(part).decode())+')\n'
        source += "p.parent.mkdir(parents=True,exist_ok=True)\nif p.exists():\n assert p.read_bytes()==raw\nelse:\n with p.open('xb') as stream:stream.write(raw)\n"
        source += "print(json.dumps({'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}))\n"
        result = ssh.run('python3 -c '+shlex.quote(source), timeout=60)
        if result.returncode:
            raise RuntimeError('Code delta part outcome requires readback: '+result.stderr)
        receipt = json.loads(result.stdout)
        if receipt['sha256'] != hashlib.sha256(part).hexdigest():
            raise RuntimeError('Shared code part readback differs')
        parts.append(path)
    payload = dict(spec=spec, previous=previous, archive_parts=parts,
                   archive_sha256=digest)
    source = '''from pathlib import Path
import json,hashlib,base64,tarfile,io,shutil,time
payload=PAYLOAD
spec=payload['spec'];previous=payload['previous'];root=Path(ROOT)
source=Path(previous['package']);target=Path(spec['package'])
assert not target.exists() and not Path(spec['once_intent']).exists()
for name in ['authority_path','release_path','script_path']:assert not Path(spec[name]).exists()
old=(source/'manifest.json').read_bytes()
assert hashlib.sha256(old).hexdigest()==previous['manifest_sha']
manifest=json.loads(old)
for name,digest in manifest['files'].items():
 p=Path(name);assert not p.is_absolute() and '..' not in p.parts
 src=source/p;assert src.is_file() and not src.is_symlink()
 assert hashlib.sha256(src.read_bytes()).hexdigest()==digest,name
shutil.copytree(source,target,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
raw=b''.join(Path(p).read_bytes() for p in payload['archive_parts']);archive_nbytes=len(raw)
assert hashlib.sha256(raw).hexdigest()==payload['archive_sha256']
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as archive:
 for item in archive.getmembers():
  p=Path(item.name);assert not p.is_absolute() and '..' not in p.parts
  assert item.isfile() and not item.issym() and not item.islnk()
 archive.extractall(root)
raw=(target/'manifest.json').read_bytes()
assert hashlib.sha256(raw).hexdigest()==spec['manifest_sha']
manifest=json.loads(raw)
for name,digest in manifest['files'].items():assert hashlib.sha256((target/name).read_bytes()).hexdigest()==digest,name
for name,digest in manifest['external_files'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
for path,sha in [('authority_path','authority_sha'),('release_path','release_sha'),('script_path','script_sha')]:
 assert hashlib.sha256(Path(spec[path]).read_bytes()).hexdigest()==spec[sha],path
assert (root/'python-dev/usr/include/python3.10/Python.h').is_file()
print(json.dumps({'status':'FROZEN_V14_SPEED_CODE_DELTA_DEPLOYED','observed_epoch':time.time(),
 'manifest_sha256':spec['manifest_sha'],'files_verified':len(manifest['files']),
 'archive_bytes':archive_nbytes,'training_submitted':False,'assets_transferred_via_client':False}))
'''
    source = source.replace('payload=PAYLOAD', 'payload='+repr(payload), 1)
    source = source.replace('root=Path(ROOT)', 'root=Path('+repr(ROOT)+')', 1)
    result = ssh.run('python3 -c '+shlex.quote(source), timeout=100)
    receipt = dict(observed_at=datetime.now(timezone.utc).isoformat(),
                   returncode=result.returncode, stdout=result.stdout, stderr=result.stderr,
                   changed_code_files=len(changed), inline_archive_bytes=len(raw),
                   archive_sha256=payload['archive_sha256'])
    (HERE.parents[1]/'outputs/tree-fresh200/RETAINED18_SPEED_V14_PACKET_DEPLOYMENT_20261008.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
