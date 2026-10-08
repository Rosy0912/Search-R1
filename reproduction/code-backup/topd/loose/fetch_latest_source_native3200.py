import sys,pathlib,json,base64,hashlib
sys.path.insert(0,'work/g1-native-cpu-deploy-v6')
from stream_transport import send
s="""
import pathlib,json,hashlib,base64
r=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd/logs')
paths=[r/'r1-stream-adamw-cpu-3200.log',r/'r1-latest-source-v4-cpu-3200/result.json',r/'r1-standard32b-sourcecpu-latest-source-v4-native26-submission.json',r/'r1-standard32b-sourcecpu-3200-observations.jsonl']
a=[]
for p in paths:
 b=p.read_bytes();a.append(dict(path=str(p),name=p.name,size=len(b),sha256=hashlib.sha256(b).hexdigest(),data=base64.b64encode(b).decode()))
print(json.dumps(a))
"""
r=send(s,attempts=3,timeout=120,retry_safe=True);assert r.returncode==0,r.stderr.decode()
p=pathlib.Path('outputs/remote-evidence/streamed-adamw-native3200');p.mkdir(exist_ok=True)
a=json.loads(r.stdout)
for x in a:
 b=base64.b64decode(x.pop('data'));assert len(b)==x['size'] and hashlib.sha256(b).hexdigest()==x['sha256'];(p/x['name']).write_bytes(b)
(p/'fetch-refs.json').write_text(json.dumps(a,indent=2)+'\n');print(json.dumps(a,indent=2))
