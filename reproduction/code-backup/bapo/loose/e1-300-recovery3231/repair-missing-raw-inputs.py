"""Add missing immutable manifest provenance inputs via same-node hardlinks."""
from pathlib import Path
import os,json,hashlib,subprocess
ROOT=Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-recovery-20261005');OLD=Path('/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct-300-stabilityfix-r2-20261005');src=OLD/'common-data/raw';dst=ROOT/'common-data/raw'
assert ROOT==ROOT.resolve()and not ROOT.is_symlink()and ROOT.is_dir()and ROOT/'common-data'==(ROOT/'common-data').resolve()and not (ROOT/'common-data').is_symlink()
assert os.uname().nodename=='h100-5' and src.is_dir()and src==src.resolve()and not dst.exists()
assert (ROOT/'source-expected.txt').read_text().strip()=='7de6494bff08f35ffeefbf23ed9a9e1b73e5612e'
assert not list((ROOT/'runs').iterdir()),'Do not mutate inputs for a live run'
assert hashlib.sha256((ROOT/'common-data/manifest.json').read_bytes()).hexdigest()=='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
items=[];dst.mkdir()
for p in sorted(src.rglob('*')):
 assert not p.is_symlink()and p==p.resolve()
 q=dst/p.relative_to(src)
 if p.is_dir():q.mkdir(exist_ok=True)
 else:
  assert p.is_file()and not q.exists();os.link(p,q);before=p.stat();after=q.stat();assert (before.st_dev,before.st_ino,before.st_size)==(after.st_dev,after.st_ino,after.st_size)
  items.append(dict(source=str(p),target=str(q),bytes=after.st_size,same_inode=True))
print(json.dumps(dict(status='MISSING_RAW_MANIFEST_INPUTS_RESTORED_SAME_NODE_IMMUTABLE_HARDLINKS_NOT_TRAINING',files=items,source_unchanged=True,manifest_unchanged=True),indent=2))
