"""Build a tiny code-only overlay on the retained published A checkout.

The effective runtime files are byte-pinned to Git. No model/data is transferred.
"""
import argparse,base64,gzip,hashlib,io,json,subprocess,tarfile
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
paths=subprocess.check_output(['git','ls-tree','-r','--name-only',commit],text=True).splitlines()
new=[name for name in paths if name.startswith(('release/recursive_bran_a/ablation150_', 'search_r1/baselines/recursive_ablation.py')) or name=='docs/recursive-mechanism-ablation-plan-20261014.md']
required=[name for name in paths if name.startswith(('search_r1/','verl/','reproduction/'))]
required+=new+['release/recursive_bran_a/h100_migration/reuse_retriever.py','release/recursive_bran_a/h100_migration/ray_cli_compat.py']
manifest={name:hashlib.sha256(subprocess.check_output(['git','show',f'{commit}:{name}'])).hexdigest() for name in sorted(set(required))}
archive=io.BytesIO()
with tarfile.open(fileobj=archive,mode='w:gz') as tar:
 for name in new:
  raw=subprocess.check_output(['git','show',f'{commit}:{name}']);member=tarfile.TarInfo(name);member.size=len(raw);member.mode=0o755 if name.endswith('.sh') else 0o644;tar.addfile(member,io.BytesIO(raw))
request=dict(commit=commit,files=manifest,overlay_archive=base64.b64encode(archive.getvalue()).decode())
encoded=base64.b64encode(gzip.compress(json.dumps(request).encode())).decode()
source='''import base64,gzip,hashlib,io,json,shutil,tarfile,time
from pathlib import Path
request=json.loads(gzip.decompress(base64.b64decode(PAYLOAD)))
b=Path('/dataset/shaoshitong/recursive-ablation150-20261014');receipt=b/'deployment.json'
if receipt.exists():
 old=json.loads(receipt.read_text());assert old['commit']==request['commit']
 for name,sha in request['files'].items():assert hashlib.sha256((b/'source'/name).read_bytes()).hexdigest()==sha,name
 print(json.dumps(old));raise SystemExit(0)
assert not (b/'source').exists(), 'Inspect any unreceipted source before retrying'
b.mkdir(parents=True,exist_ok=True);dest=b/('source-pending-'+request['commit'][:8])
assert not dest.exists(), 'Inspect incomplete own overlay before retrying'
original=Path('/dataset/shaoshitong/recursive-bran-a-h100-20261012/source-71d4e69')
shutil.copytree(original,dest,symlinks=True,ignore=shutil.ignore_patterns('.git','__pycache__','*.pyc'))
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(request['overlay_archive'])),mode='r:gz') as tar:
 for m in tar:
  path=Path(m.name);assert not path.is_absolute() and '..' not in path.parts and m.isfile()
  target=dest/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(tar.extractfile(m).read());target.chmod(m.mode)
for name,sha in request['files'].items():assert hashlib.sha256((dest/name).read_bytes()).hexdigest()==sha,name
dest.rename(b/'source')
record=dict(time=time.time(),commit=request['commit'],source=str(b/'source'),code_overlay_bytes=len(base64.b64decode(request['overlay_archive'])),
 file_count=len(request['files']),all_manifest_sha_equal=True,manifest_scope='All effective search_r1/verl/reproduction core and new ablation sources plus used H100 helpers',
 node_local_base=str(original),original_runs_modified=False)
receipt.write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
'''.replace('PAYLOAD',repr(encoded))
Path(a.output).write_text(source)
print(json.dumps(dict(commit=commit,request_bytes=len(source.encode()),overlay_bytes=len(archive.getvalue()),verified_runtime_files=len(manifest))))
