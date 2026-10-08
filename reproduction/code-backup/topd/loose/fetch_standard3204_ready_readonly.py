import sys,json,hashlib,base64
from pathlib import Path
sys.path.insert(0,'work/g1-native-cpu-deploy-v6');from stream_transport import send
code=r'''
import pathlib,json,hashlib,base64
root=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd/logs/r1-standard32b-profile2-v12/evidence');paths=[root/'actor-resource-audit'/('%s-ready-rank%d.json'%(role,rank)) for role in ('actor','ref') for rank in range(8)]+[root/'global-worker-ready.json'];rows=[]
for p in paths:
 assert not p.is_symlink();a=p.stat();b=p.read_bytes();z=p.stat();assert (a.st_ino,a.st_size,a.st_mtime_ns)==(z.st_ino,z.st_size,z.st_mtime_ns)
 rows.append(dict(path=str(p),size=len(b),sha256=hashlib.sha256(b).hexdigest(),data=base64.b64encode(b).decode()))
print(json.dumps(rows))
'''
r=send(code,attempts=3,retry_safe=True,timeout=90);assert r.returncode==0,(r.stdout,r.stderr)
rows=json.loads(r.stdout);out=Path('outputs/remote-evidence/standard3204-ready-originals');out.mkdir(exist_ok=False)
for row in rows:
 b=base64.b64decode(row.pop('data'));assert len(b)==row['size'] and hashlib.sha256(b).hexdigest()==row['sha256'];(out/Path(row['path']).name).write_bytes(b)
(out/'original-refs.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows))
