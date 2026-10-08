PINS={'archive_sha256': '62f66a12f1511a175f0af6d0524bfad0d33744ae65cf882c752562fd7706a879', 'archive_bytes': 573843, 'remote_package': '/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300/candidates/current-fresh200-chain-mask-v8-9693f0c', 'code_commit': '9693f0c846fab483a03e1d70ce1109aeddc5e118', 'manifest_sha256': '336a0a60df561fd3fcf50e7fdc03cee344a0b33e9d916193c7b4a4716b1b6a12', 'remote_extract_destination': '/dataset/shaoshitong/search-r1-rl/tree-grpo/instruct300'}
import io,tarfile,hashlib,json,socket,sys
from pathlib import Path
assert socket.gethostname().split('.')[0]=='h100-9'
raw=sys.stdin.buffer.read(PINS['archive_bytes']+1)
assert len(raw)==PINS['archive_bytes'] and hashlib.sha256(raw).hexdigest()==PINS['archive_sha256']
root=Path(PINS['remote_extract_destination']);candidate=Path(PINS['remote_package'])
assert not candidate.exists() and not candidate.is_symlink()
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as bundle:
 members=bundle.getmembers()
 for item in members:
  rel=Path(item.name)
  assert not rel.is_absolute() and '..' not in rel.parts and item.isfile()
  target=root/rel
  assert not target.exists() and not target.is_symlink()
 for item in members:
  target=root/item.name;target.parent.mkdir(parents=True,exist_ok=True)
  with target.open('xb') as out:out.write(bundle.extractfile(item).read())
  target.chmod(item.mode & 0o777)
assert hashlib.sha256((candidate/'manifest.json').read_bytes()).hexdigest()==PINS['manifest_sha256']
print(json.dumps(dict(status='CODE_ONLY_CURRENT_CHAIN_V8_STAGED',code_commit=PINS['code_commit'],archive_sha256=PINS['archive_sha256'],file_count=len(members),node='h100-9',remote_package=str(candidate))))
