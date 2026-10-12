import json,hashlib,os
from pathlib import Path
r=Path('/dataset/shaoshitong/search-r1-rl/bapo-aligned-cohost-node11-v6-20261009');oldrun=r/'runs/formal-aligned-em200-ef141268a5694dabace114936d380acc-3356'
src=r/'evidence/partial-resume-bindings3356-v2.json';d=json.loads(src.read_bytes());closure=d['original_closure'];assert all(oldrun in Path(x['path']).parents for x in closure)
for report in sorted(oldrun.rglob('native-initial*.json')):
 v=json.loads(report.read_bytes());p=Path(v['path']);assert oldrun in p.parents and p.resolve()==p and p.is_file()and not p.is_symlink()
 entry=dict(path=str(p),bytes=p.stat().st_size,sha256=v['sha256']);assert not any(x['path']==str(p)for x in closure);closure.append(entry)
assert len([x for x in closure if Path(x['path']).name.startswith('native-initial')and x['path'].endswith('.pt')])==9
d['original_closure']=closure;d['parent_bindings_sha256']=hashlib.sha256(src.read_bytes()).hexdigest();p=r/'evidence/partial-resume-bindings3356-v3.json'
with p.open('x')as f:json.dump(d,f,sort_keys=True,allow_nan=False);f.flush();os.fsync(f.fileno())
print(json.dumps({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'files':len(closure),'native_initial_tensor_bindings':9,'large_files_hashed_here':False}))
