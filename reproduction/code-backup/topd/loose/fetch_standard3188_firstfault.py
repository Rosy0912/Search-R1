"""Readonly small actual first-fault evidence alongside complete streaming fetch."""
import pathlib,sys,json,base64,hashlib
sys.dont_write_bytecode=True
sys.path.insert(0,str(pathlib.Path(__file__).parent/'g1-native-cpu-deploy-v6'))
from stream_transport import send
root='/dataset/shaoshitong/search-r1-rl/topd/logs/r1-standard32b-profile2-v8/evidence'
code='import pathlib,json,base64,hashlib\nrows=[]\nfor n in '+repr(['trainer-first-exception.json','trainer.log','terminal-binding.json'])+':\n p=pathlib.Path('+repr(root)+')/n;a=p.stat();assert not p.is_symlink() and a.st_size<4*1024*1024;b=p.read_bytes();z=p.stat();assert (a.st_size,a.st_ino,a.st_mtime_ns)==(z.st_size,z.st_ino,z.st_mtime_ns);rows.append(dict(name=n,path=str(p),size=len(b),sha256=hashlib.sha256(b).hexdigest(),data=base64.b64encode(b).decode()))\nprint(json.dumps(rows))\n'
r=send(code,attempts=3,retry_safe=True,timeout=90)
if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace'))
rows=json.loads(r.stdout);D=pathlib.Path('outputs/remote-evidence/standard-r1-profile3188-firstfault');D.mkdir(exist_ok=False,parents=True)
for row in rows:
 b=base64.b64decode(row.pop('data'),validate=True);assert len(b)==row['size'] and hashlib.sha256(b).hexdigest()==row['sha256'];(D/row['name']).write_bytes(b)
(D/'fetch-refs.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
