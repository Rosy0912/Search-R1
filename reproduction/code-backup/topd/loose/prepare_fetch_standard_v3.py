import sys,json,pathlib,base64,hashlib
sys.path.insert(0,'work/g1-native-cpu-deploy-v6')
from stream_transport import send
source="""
import sys
sys.dont_write_bytecode=True
import pathlib,json,subprocess,hashlib,base64,os
R=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd');d=R/'logs/r1-standard32b-profile2-v3'
if not d.exists():
 subprocess.run([str(R/'env/bin/python'),str(R/'tools/search-r1-32b-grpo100-tools-v3/prepare.py'),'--existing-asset-config',str(R/'logs/g1-initial-static-cpu-3102/recipe/initial-only-static-config.json'),'--mode','profile2','--directory',str(d)],check=True,stdout=sys.stderr,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
paths=[R/'logs/r1-32b-source-cpu-3157.log',R/'logs/r1-32b-source-cpu-3157/pytest.log',R/'logs/r1-32b-source-cpu-3157/result.json',R/'logs/r1-standard32b-sourcecpu-v3-submission.json',R/'logs/r1-standard32b-sourcecpu-3157-observations.jsonl',d/'static-config.json']
rows=[]
for p in paths:
 b=p.read_bytes();rows.append(dict(path=str(p),name=p.name,size=len(b),sha256=hashlib.sha256(b).hexdigest(),data=base64.b64encode(b).decode()))
print(json.dumps(rows))
"""
r=send(source,attempts=1,timeout=180)
pathlib.Path('outputs/standard-r1-native3157-config-transport.log').write_bytes(r.stderr)
assert r.returncode==0,r.stderr.decode()
rows=json.loads(r.stdout);out=pathlib.Path('outputs/remote-evidence/standard-r1-native3157');out.mkdir(exist_ok=True)
for row in rows:
 b=base64.b64decode(row.pop('data'));assert hashlib.sha256(b).hexdigest()==row['sha256'] and len(b)==row['size'];(out/row['name']).write_bytes(b)
(out/'fetch-refs.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
