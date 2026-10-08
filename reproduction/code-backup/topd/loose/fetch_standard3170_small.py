import sys,pathlib,json,base64,hashlib
sys.path.insert(0,'work/g1-native-cpu-deploy-v6')
from stream_transport import send
code=r'''
import pathlib,json,base64,hashlib,time,subprocess
r=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd/logs/r1-standard32b-profile2-v5/evidence');rows=[]
for name in ['train-result.json','owned-cleanup.json','global-worker-ready.json','effective-runtime-config.json','trainer.log','retriever.log','trainer-first-exception.json']:
 p=r/name;b=p.read_bytes();assert len(b)<512*1024;rows.append(dict(name=name,path=str(p),size=len(b),sha256=hashlib.sha256(b).hexdigest(),data=base64.b64encode(b).decode()))
for p in sorted((r/'actor-resource-audit').glob('*ready*.json')):
 b=p.read_bytes();rows.append(dict(name='actor-resource-audit/'+p.name,path=str(p),size=len(b),sha256=hashlib.sha256(b).hexdigest(),data=base64.b64encode(b).decode()))
metadata=[]
for p in sorted(r.glob('*.jsonl')):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 metadata.append(dict(name=p.name,path=str(p),size=p.stat().st_size,sha256=h.hexdigest()))
print(json.dumps(dict(scope='READ_ONLY_REAL3170_SMALL_RAW_WITH_LARGE_REMOTE_SHA_RETAINED',epoch=time.time(),rows=rows,large_metadata=metadata)))
'''
r=send(code,timeout=180);assert r.returncode==0,r.stderr.decode();data=json.loads(r.stdout);out=pathlib.Path('outputs/remote-evidence/standard-r1-profile3170');out.mkdir(exist_ok=True)
for row in data['rows']:
 b=base64.b64decode(row.pop('data'));assert len(b)==row['size'] and hashlib.sha256(b).hexdigest()==row['sha256'];p=out/row['name'];p.parent.mkdir(exist_ok=True);p.write_bytes(b)
(out/'small-fetch-refs.json').write_text(json.dumps(data,indent=2)+'\n');print('REAL_SMALL_RAW_VERIFIED',len(data['rows']))
