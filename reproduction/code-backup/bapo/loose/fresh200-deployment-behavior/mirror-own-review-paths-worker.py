from pathlib import Path
import os,hashlib,json
R=Path('/dataset/shaoshitong/search-r1-rl/bapo-fresh200-behavior-20261006');S=R/'reviews';D=R/'logs/reviews'
assert os.uname().nodename=='h100-5' and R.resolve()==R and not S.is_symlink()
D.mkdir(exist_ok=False)
bindings={}
for p in S.iterdir():
 assert p.is_file() and not p.is_symlink();raw=p.read_bytes()
 with (D/p.name).open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
 assert hashlib.sha256((D/p.name).read_bytes()).digest()==hashlib.sha256(raw).digest();bindings[p.name]=hashlib.sha256(raw).hexdigest()
print(json.dumps(dict(scope='EXACT_OWN_REVIEW_COPIES_TO_EXISTING_REVIEWED_LOOKUP_PATH_NO_PERMISSION',files=bindings)))
