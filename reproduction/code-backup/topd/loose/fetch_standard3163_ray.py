import sys,pathlib,json,base64,hashlib
sys.path.insert(0,'work/g1-native-cpu-deploy-v6')
from stream_transport import send
code=r'''
import pathlib,json,base64,hashlib
r=pathlib.Path('/dev/shm/r1-3163');rows=[]
for p in r.rglob('*'):
 if p.is_file() and not p.is_symlink() and 'logs' in p.parts and (p.name in ('raylet.out','raylet.err') or p.name.startswith('python-core-driver') or (p.name.startswith(('worker-','python-core-worker-')) and any(str(x) in p.name for x in [3045587,3045882,3045883,3045884,3045885,3045886,3045887,3045888]))):
  b=p.read_bytes();assert len(b)<4*1024**2;rows.append(dict(path=str(p),name=p.name,size=len(b),sha256=hashlib.sha256(b).hexdigest(),data=base64.b64encode(b).decode()))
print(json.dumps(rows))
'''
r=send(code,timeout=180);assert r.returncode==0,r.stderr.decode();rows=json.loads(r.stdout);out=pathlib.Path('outputs/remote-evidence/standard-r1-profile3163-ray');out.mkdir(exist_ok=True)
for row in rows:
 b=base64.b64decode(row.pop('data'));assert len(b)==row['size'] and hashlib.sha256(b).hexdigest()==row['sha256'];p=out/row['name'];assert not p.exists();p.write_bytes(b)
(out/'raw-refs.json').write_text(json.dumps(rows,indent=2)+'\n');print('ACTUAL_FULL_RAY_LOGS',len(rows),sum(x['size'] for x in rows))
